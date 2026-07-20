require("dotenv").config();

const fs = require("fs");
const path = require("path");
const express = require("express");
const cors = require("cors");
const OpenAI = require("openai").default;

const app = express();
const PORT = process.env.PORT || 3000;

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const CHAT_MODEL = process.env.CHAT_MODEL || "gpt-5.4-mini";
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";
const CONTACT_FORM_URL = process.env.CONTACT_FORM_URL || "https://usg.usc.edu/contact/";
const CONTACT_EMAIL = process.env.CONTACT_EMAIL || "usg@usc.edu";
// RAG answers are simple grounded extraction — low effort cuts
// time-to-first-token sharply. Crisis generation is left untouched.
const REASONING_EFFORT = process.env.REASONING_EFFORT || "low";

if (!OPENAI_API_KEY) {
  console.error("Set OPENAI_API_KEY in your environment.");
  process.exit(1);
}

const client = new OpenAI({ apiKey: OPENAI_API_KEY });
const KB_PATH = path.join(__dirname, "kb.json");

// Postgres/pgvector retrieval when SUPABASE_DB_URL is set; kb.json fallback
// otherwise so the app keeps working before the env var lands everywhere.
const { Pool } = require("pg");
const pool = process.env.SUPABASE_DB_URL
  ? new Pool({
      connectionString: process.env.SUPABASE_DB_URL,
      max: 5,
      ssl: { rejectUnauthorized: false },
    })
  : null;

// Langfuse tracing, active only when keys are set (LANGFUSE_SECRET_KEY,
// LANGFUSE_PUBLIC_KEY, optional LANGFUSE_BASEURL). Crisis requests are
// deliberately never traced — that content stays out of analytics tools.
let langfuse = null;
if (process.env.LANGFUSE_SECRET_KEY && process.env.LANGFUSE_PUBLIC_KEY) {
  const { Langfuse } = require("langfuse");
  langfuse = new Langfuse();
}

// Two-stage retrieval: RERANK=0 disables the second stage (for A/B evals).
const RERANK = process.env.RERANK !== "0";
const RERANK_MODEL = process.env.RERANK_MODEL || "gpt-5.4-mini";
const RERANK_CANDIDATES = 20;

// ponytail: LLM reranker on the existing OpenAI key; swap this function's
// body for Cohere Rerank (cross-encoder) if a COHERE_API_KEY ever lands
async function rerank(query, candidates, k = 4) {
  if (candidates.length <= k) return candidates;
  const list = candidates
    .map((c, i) => `[${i}] (${c.source_title}) ${c.text.slice(0, 500)}`)
    .join("\n\n");
  try {
    const resp = await client.responses.create({
      model: RERANK_MODEL,
      instructions: `You are a search reranker. Given a query and numbered passages, reply with ONLY a JSON array of the indices of the ${k} passages most relevant to the query, most relevant first.`,
      input: [{ role: "user", content: `Query: ${query}\n\nPassages:\n${list}` }],
    });
    const idx = JSON.parse(resp.output_text.match(/\[[\d,\s]*\]/)[0]);
    const picked = idx
      .filter((i) => Number.isInteger(i) && candidates[i])
      .slice(0, k)
      .map((i) => candidates[i]);
    return picked.length ? picked : candidates.slice(0, k);
  } catch (e) {
    console.warn("rerank failed, using vector order:", e.message);
    return candidates.slice(0, k);
  }
}

async function topChunksDb(queryEmbedding, k = 4) {
  const vec = `[${queryEmbedding.join(",")}]`;
  const { rows } = await pool.query(
    `select source_url, source_title, chunk_index, text,
            source_modified::text as source_modified,
            source_modified_year, evergreen,
            1 - (embedding <=> $1::vector) as score
     from chunks
     order by embedding <=> $1::vector
     limit $2`,
    [vec, k]
  );
  return rows;
}

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "..", "docs")));

function loadKb() {
  if (!fs.existsSync(KB_PATH)) {
    return { chunks: [] };
  }

  const raw = fs.readFileSync(KB_PATH, "utf-8");
  const parsed = JSON.parse(raw);
  return parsed && Array.isArray(parsed.chunks) ? parsed : { chunks: [] };
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function norm(a) {
  return Math.sqrt(dot(a, a)) || 1;
}

function cosineSimilarity(a, b) {
  return dot(a, b) / (norm(a) * norm(b));
}

async function embed(text) {
  const resp = await client.embeddings.create({
    model: EMBED_MODEL,
    input: text,
  });
  return resp.data[0].embedding;
}

// Deterministic staleness gate — code decides, never the model.
// A source is stale if it's not evergreen and was last modified before the
// current year. Missing dates are treated as fresh (pre-date-gate kb.json
// has no dates; gating on "unknown" would flag everything).
function assessFreshness(sources, now = new Date()) {
  const year = now.getFullYear();
  const stale = sources.filter(
    (s) => !s.evergreen && s.source_modified_year && s.source_modified_year < year
  );
  if (!stale.length) return { stale, notice: "" };
  const oldest = Math.min(...stale.map((s) => s.source_modified_year));
  return {
    stale,
    notice:
      `\n\n---\n⚠️ Some of this answer comes from pages last updated in ${oldest}, ` +
      `so details may have changed. To confirm, [contact USG](${CONTACT_FORM_URL}) ` +
      `or email ${CONTACT_EMAIL}.`,
  };
}

function topChunks(queryEmbedding, chunks, k = 4) {
  return chunks
    .map((chunk) => ({
      ...chunk,
      score: cosineSimilarity(queryEmbedding, chunk.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

const CRISIS_REPLIES = [
  "I’m really sorry you’re hurting this much. Please call or text 988 right now and reach out to someone you trust to stay with you.",
  
  "You don’t have to carry this alone tonight. Please contact 988 now and tell a trusted person nearby how you’re feeling.",
  
  "I’m really glad you said something. Please call or text 988 right now and be with someone you trust if you can.",
  
  "I’m sorry you’re in this much pain. Please reach out to 988 now and let someone close to you know you need support.",
];

const CRISIS_SOURCES = [
  {
    source_title: "Health and Wellness Resources",
    source_url: "https://healthandwellness.usc.edu/",
  },
  {
    source_title: "988 Suicide & Crisis Lifeline",
    source_url: "https://988lifeline.org/",
  },
];

function pickCrisisReply(message) {
  const chars = String(message || "");
  const sum = [...chars].reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
  return CRISIS_REPLIES[Math.abs(sum) % CRISIS_REPLIES.length];
}

function hasCrisisKeywords(message) {
  const text = String(message || "").toLowerCase();

  const patterns = [
    /\bsuicid(e|al)\b/i,
    /\bself[-\s]?harm\b/i,
    /\bend my life\b/i,
    /\btake my life\b/i,
    /\bkill myself\b/i,
    /\bhurt myself\b/i,
    /\boverdose\b/i,
    /\bnot want to be here\b/i,
    /\b(don'?t|do not) want to (be here|live|exist)\b/i,
    /\bwant to die\b/i,
  ];

  return patterns.some((re) => re.test(text));
}

function hasDisallowedAssistantLanguage(text) {
  const badPatterns = [
    /\bi can help\b/i,
    /\bi[' ]?m here if you want to talk\b/i,
    /\blet me know\b/i,
    /\bwhat would you like to do next\b/i,
    /\bhow can i help\b/i,
    /\bi can support you\b/i,
  ];

  return badPatterns.some((re) => re.test(String(text || "")));
}

function historySuggestsCrisis(history) {
  const combined = history
    .map((m) => m?.content || "")
    .join(" ")
    .toLowerCase();

  const patterns = [
    /\bkill myself\b/i,
    /\bwant to die\b/i,
    /\bsuicid(e|al)\b/i,
    /\bdon'?t want to live\b/i,
    /\bno reason to live\b/i,
    /\blife is pointless\b/i,
  ];

  return patterns.some((re) => re.test(combined));
}

async function isCrisisMessage(message) {
  if (hasCrisisKeywords(message)) return true;

  try {
    const mod = await client.moderations.create({
      model: "omni-moderation-latest",
      input: message,
    });

    const result = mod.results?.[0];
    return Boolean(
      result?.flagged &&
      (
        result?.categories?.["self-harm"] ||
        result?.categories?.["self-harm/instructions"] ||
        result?.categories?.["self-harm/intent"]
      )
    );
  } catch (err) {
    console.warn("Moderation failed, using keyword fallback:", err.message);
    return hasCrisisKeywords(message);
  }
}

const CRISIS_INSTRUCTIONS = `
You are responding to a user who may be in emotional crisis or considering self-harm.

Your tone should feel:
- warm
- calm
- grounded
- human
- emotionally present

Do not sound robotic, clinical, overly formal, or like a policy document.

Keep responses SHORT.
Usually 2-5 sentences maximum.

Always:
- acknowledge the pain directly
- encourage contacting 988
- encourage reaching out to a trusted real person nearby

Only mention 911 or emergency rooms if the user may be in immediate danger.

Do NOT:
- overwhelm the user with long lists
- give excessive bullet points
- sound repetitive
- act like a therapist
- encourage emotional dependency on the chatbot
- say things like "I'm always here for you"

If the user continues expressing hopelessness, treat follow-up messages as part of the same emotional conversation even if the wording is vague.

Examples:
- "what's the point"
- "nothing matters"
- "why should I"
- "i dont want to do this anymore"

should be understood as emotional continuation, not a new factual topic.

Responses should feel compassionate and direct, not scripted.
`;

// Returns the complete, guardrail-checked crisis payload, or null if the
// message is not a crisis. Always generated in full before anything is
// shown — the streaming endpoint must never stream this path.
async function crisisPayload(message, history) {
  const crisis = (await isCrisisMessage(message)) || historySuggestsCrisis(history);
  if (!crisis) return null;

  const crisisResponse = await client.responses.create({
    model: CHAT_MODEL,
    instructions: CRISIS_INSTRUCTIONS,
    input: [
      {
        role: "user",
        content: [{ type: "input_text", text: message }],
      },
    ],
  });

  const answer = crisisResponse.output_text || pickCrisisReply(message);
  return {
    crisis: true,
    answer: hasDisallowedAssistantLanguage(answer) ? pickCrisisReply(message) : answer,
    sources: CRISIS_SOURCES,
  };
}

// Shared RAG prep: embed -> retrieve -> rerank -> prompt + deduped sources.
// Throws err with .status when the KB is empty.
async function ragPrepare(message, trace) {
  let t = new Date();
  const queryEmbedding = await embed(message);
  trace?.span({
    name: "embed",
    startTime: t,
    endTime: new Date(),
    metadata: { model: EMBED_MODEL },
  });

  t = new Date();
  const kCandidates = RERANK ? RERANK_CANDIDATES : 4;
  let matches;
  if (pool) {
    matches = await topChunksDb(queryEmbedding, kCandidates);
  } else {
    matches = topChunks(queryEmbedding, loadKb().chunks, kCandidates);
  }
  if (!matches.length) {
    const err = new Error("Knowledge base is empty. Run ingestion first.");
    err.status = 400;
    throw err;
  }
  trace?.span({
    name: "retrieve",
    startTime: t,
    endTime: new Date(),
    output: matches.map((m) => ({
      url: m.source_url,
      chunk: m.chunk_index,
      score: m.score,
    })),
    metadata: { backend: pool ? "pgvector" : "kb.json", k: kCandidates },
  });

  if (RERANK) {
    t = new Date();
    matches = await rerank(message, matches, 4);
    trace?.span({
      name: "rerank",
      startTime: t,
      endTime: new Date(),
      output: matches.map((m) => ({ url: m.source_url, chunk: m.chunk_index })),
      metadata: { model: RERANK_MODEL },
    });
  }

  const today = new Date().toISOString().slice(0, 10);
  const context = matches
    .map(
      (chunk, idx) =>
        `[${idx + 1}] ${chunk.source_title}` +
        (chunk.source_modified ? ` (page last updated: ${chunk.source_modified.slice(0, 10)})` : "") +
        `\n${chunk.source_url}\n${chunk.text}`
    )
    .join("\n\n");

  const instructions = `
        Today's date is ${today}.

        Answer the user's question using the provided context AND the recent conversation history.

        Each context source shows when its page was last updated. When answering about time-sensitive things (elections, events, results, deadlines), mention how current the information is, e.g. "as of the page's last update in March 2026...". Do not decide whether information is outdated beyond citing these dates.

        If the user's message is short, vague, emotional, or dependent on previous messages, interpret it in conversational context before assuming it is a brand-new factual question.

        Keep the conversation natural and context-aware.

        Use the knowledge base for factual USC/USG information. State only facts found in the context — do not fill gaps from general knowledge, and do not attribute to a resource anything the context does not say about it.

        If the knowledge base truly does not contain enough information, say so plainly.

        Be concise, accurate, and human.
      `;

  const uniqueSourcesMap = new Map();
  for (const chunk of matches) {
    const key = chunk.source_url;
    const existing = uniqueSourcesMap.get(key);
    if (!existing || chunk.score > existing.score) {
      uniqueSourcesMap.set(key, {
        source_title: chunk.source_title,
        source_url: chunk.source_url,
        score: chunk.score,
        source_modified: chunk.source_modified || null,
        source_modified_year: chunk.source_modified_year || null,
        evergreen: Boolean(chunk.evergreen),
      });
    }
  }
  const sources = Array.from(uniqueSourcesMap.values());
  const { notice } = assessFreshness(sources);
  const userContent = `Context:\n${context}\n\nCurrent user message:\n${message}`;

  return { instructions, userContent, sources, notice };
}

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

app.post("/api/chat", async (req, res) => {
  try {
    const message = (req.body.message || "").trim();
    const history = Array.isArray(req.body.history)
      ? req.body.history.slice(-8)
      : [];
    if (!message) {
      return res.status(400).json({ error: "Missing message." });
    }

    // Crisis check and retrieval run concurrently; the crisis verdict still
    // gates the response — nothing is sent until it resolves.
    const trace = langfuse?.trace({ name: "chat", input: message });
    const ragPromise = ragPrepare(message, trace).catch((e) => e);
    const crisis = await crisisPayload(message, history);
    if (crisis) return res.json(crisis);

    const rag = await ragPromise;
    if (rag instanceof Error) throw rag;
    const { instructions, userContent, sources, notice } = rag;

    const t = new Date();
    const response = await client.responses.create({
      model: CHAT_MODEL,
      instructions,
      input: [...history, { role: "user", content: userContent }],
      reasoning: { effort: REASONING_EFFORT },
    });

    trace?.generation({
      name: "generate",
      model: CHAT_MODEL,
      startTime: t,
      endTime: new Date(),
      input: userContent,
      output: response.output_text || "",
      usage: {
        input: response.usage?.input_tokens,
        output: response.usage?.output_tokens,
      },
    });

    const answer = (response.output_text || "") + notice;
    trace?.update({ output: answer });
    langfuse?.flushAsync().catch(() => {});

    res.json({
      answer,
      sources,
      crisis: false,
    });
  } catch (err) {
    console.error(err);
    res.status(err.status || 500).json({ error: err.message || "Server error" });
  }
});

app.post("/api/chat/stream", async (req, res) => {
  try {
    const message = (req.body.message || "").trim();
    const history = Array.isArray(req.body.history) ? req.body.history.slice(-8) : [];
    if (!message) {
      return res.status(400).json({ error: "Missing message." });
    }

    res.set({
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    });
    res.flushHeaders();
    const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);

    // Crisis answers are generated and guardrail-checked in full, then sent
    // as one event — never token-streamed. The check runs concurrently with
    // retrieval but always resolves before the first token goes out.
    const trace = langfuse?.trace({ name: "chat-stream", input: message });
    const ragPromise = ragPrepare(message, trace).catch((e) => e);
    const crisis = await crisisPayload(message, history);
    if (crisis) {
      send(crisis);
      send({ done: true });
      return res.end();
    }

    const rag = await ragPromise;
    if (rag instanceof Error) throw rag;
    const { instructions, userContent, sources, notice } = rag;

    const abort = new AbortController();
    req.on("close", () => abort.abort());

    const t = new Date();
    const stream = await client.responses.create(
      {
        model: CHAT_MODEL,
        instructions,
        input: [...history, { role: "user", content: userContent }],
        reasoning: { effort: REASONING_EFFORT },
        stream: true,
      },
      { signal: abort.signal }
    );

    let full = "";
    let usage = null;
    for await (const event of stream) {
      if (event.type === "response.output_text.delta") {
        full += event.delta;
        send({ delta: event.delta });
      } else if (event.type === "response.completed") {
        usage = event.response?.usage;
      }
    }

    trace?.generation({
      name: "generate",
      model: CHAT_MODEL,
      startTime: t,
      endTime: new Date(),
      input: userContent,
      output: full,
      usage: { input: usage?.input_tokens, output: usage?.output_tokens },
    });
    trace?.update({ output: full + notice });
    langfuse?.flushAsync().catch(() => {});

    send({ done: true, sources, notice, crisis: false });
    res.end();
  } catch (err) {
    if (err.name === "AbortError") return res.end();
    console.error(err);
    if (res.headersSent) {
      res.write(`data: ${JSON.stringify({ error: err.message || "Server error" })}\n\n`);
      return res.end();
    }
    res.status(err.status || 500).json({ error: err.message || "Server error" });
  }
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

module.exports = {
  app,
  client,
  CHAT_MODEL,
  RERANK,
  rerank,
  embed,
  loadKb,
  topChunks,
  topChunksDb,
  assessFreshness,
  hasCrisisKeywords,
  pool,
};