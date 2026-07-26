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
  // Query-aware snippet: center on the first query-term hit so evidence
  // deeper in the chunk is visible to the reranker (a name at char 502
  // was invisible to a plain 500-char head slice).
  const terms = query.toLowerCase().split(/\W+/).filter((w) => w.length > 3);
  const snippet = (text, len = 500) => {
    const hay = text.toLowerCase();
    let pos = -1;
    for (const t of terms) {
      const i = hay.indexOf(t);
      if (i !== -1 && (pos === -1 || i < pos)) pos = i;
    }
    if (pos <= len / 2) return text.slice(0, len);
    const start = Math.max(0, pos - Math.floor(len / 2));
    return text.slice(start, start + len);
  };
  const list = candidates
    .map(
      (c, i) =>
        `[${i}] (${c.source_title}` +
        (c.source_modified ? `, updated ${String(c.source_modified).slice(0, 10)}` : "") +
        `) ${snippet(c.text)}`
    )
    .join("\n\n");
  try {
    const resp = await client.responses.create({
      model: RERANK_MODEL,
      instructions: `You are a search reranker. Today is ${new Date().toISOString().slice(0, 10)}. Given a query and numbered passages, reply with ONLY a JSON array of the indices of the ${k} passages most relevant to the query, most relevant first. For queries about the current state of things (who holds a role, what resources or processes exist now), prefer recently-updated pages over old dated posts; for queries about a specific past event or announcement, the dated post from that time is the right answer.`,
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

// Render sits behind a proxy; needed for per-IP rate limiting.
app.set("trust proxy", 1);

// Browser callers: the USG site, the GitHub Pages widget host, localhost.
// Non-browser clients send no Origin header and are unaffected (CORS is a
// browser mechanism, not API auth — rate limiting below covers the rest).
const ALLOWED_ORIGINS = (
  process.env.CORS_ORIGINS ||
  "https://usg.usc.edu,https://arsalanghogari.github.io,https://arsalanghogari.com,https://www.arsalanghogari.com"
).split(",");
app.use(
  cors({
    origin: (origin, cb) =>
      cb(
        null,
        !origin ||
          ALLOWED_ORIGINS.includes(origin) ||
          /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)
      ),
  })
);

app.use(express.json({ limit: "50kb" }));

const rateLimit = require("express-rate-limit");
app.use(
  "/api/",
  rateLimit({
    windowMs: 60_000,
    max: Number(process.env.RATE_LIMIT_PER_MIN) || 20,
    standardHeaders: true,
    legacyHeaders: false,
  })
);

app.use(express.static(path.join(__dirname, "..", "docs")));

const MAX_MESSAGE_CHARS = 2000;

// Shared request validation for both chat endpoints.
// Returns {message, history} or {error}.
function parseChatBody(body) {
  const message = (body?.message || "").trim();
  if (!message) return { error: "Missing message." };
  if (message.length > MAX_MESSAGE_CHARS) {
    return { error: `Message too long (max ${MAX_MESSAGE_CHARS} characters).` };
  }
  const history = (Array.isArray(body.history) ? body.history : [])
    .filter(
      (m) =>
        m &&
        (m.role === "user" || m.role === "assistant") &&
        typeof m.content === "string" &&
        m.content.length <= MAX_MESSAGE_CHARS * 2
    )
    .slice(-8);
  // anonymous widget-generated session id — groups traces into
  // conversations in Langfuse, no PII
  const sessionId =
    typeof body.sessionId === "string" && /^[\w-]{8,64}$/.test(body.sessionId)
      ? body.sessionId
      : null;
  return { message, history, sessionId };
}

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

// ---- Agentic routing: KB retrieval vs live events calendar ---------------

const WP_EVENTS_API =
  process.env.WP_EVENTS_API || "https://usg.usc.edu/wp-json/tribe/events/v1";
const ROUTER_MODEL = process.env.ROUTER_MODEL || "gpt-5.4-mini";
const BROWSER_UA =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";

const AGENT_TOOLS = [
  {
    type: "function",
    name: "search_knowledge_base",
    strict: true,
    description:
      "Search the USG website knowledge base: departments, branches, funding policies, resources, elections, press releases, people and rosters, and anything that already happened (past meetings, outcomes, results, announcements).",
    parameters: {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"],
      additionalProperties: false,
    },
  },
  {
    type: "function",
    name: "get_project_tracker",
    strict: true,
    description:
      "Fetch the LIVE USG project tracker (Legislative Branch dashboard): current projects, how many are active, their statuses, committees, and collaborators. Use for questions about what USG is working on now, project counts, or the status of an initiative. Never for department descriptions or past events.",
    parameters: {
      type: "object",
      properties: {},
      required: [],
      additionalProperties: false,
    },
  },
  {
    type: "function",
    name: "get_upcoming_events",
    strict: true,
    description:
      "Fetch live UPCOMING events from the USG events calendar. Use only for future/scheduled happenings — what's coming up, when something meets next, this week's events. Never for past meetings or their outcomes.",
    parameters: {
      type: "object",
      properties: {
        search: {
          type: ["string", "null"],
          description: "Optional keyword filter, e.g. 'senate' or 'wellness'",
        },
      },
      required: ["search"],
      additionalProperties: false,
    },
  },
];

// One forced tool call decides the path. Falls back to KB on any failure —
// the bot must keep answering even if routing breaks.
async function routeTool(message, history, trace) {
  const t = new Date();
  try {
    const resp = await client.responses.create({
      model: ROUTER_MODEL,
      instructions: "Route the user's question to exactly one tool.",
      input: [...history, { role: "user", content: message }],
      tools: AGENT_TOOLS,
      tool_choice: "required",
    });
    const call = (resp.output || []).find((o) => o.type === "function_call");
    let args = {};
    try {
      args = JSON.parse(call?.arguments || "{}");
    } catch {}
    const route = {
      tool:
        { get_upcoming_events: "events", get_project_tracker: "tracker" }[call?.name] || "kb",
      args,
    };
    trace?.update({ tags: [route.tool] }); // dashboard slicing by route
    trace?.span({
      name: "route",
      startTime: t,
      endTime: new Date(),
      // tool name only — args can echo the raw user message, and this
      // span may belong to a trace that turns out to be a crisis request
      output: { tool: route.tool },
      metadata: { model: ROUTER_MODEL },
    });
    return route;
  } catch (e) {
    console.warn("router failed, defaulting to kb:", e.message);
    return { tool: "kb", args: {} };
  }
}

// Live-events counterpart of ragPrepare: same return shape, so both
// endpoints treat the two paths identically.
async function eventsPrepare(message, args, trace) {
  const t = new Date();
  const today = new Date().toISOString().slice(0, 10);

  async function fetchEvents(search) {
    const params = new URLSearchParams({ per_page: "10", start_date: today });
    if (search) params.set("search", search);
    const r = await fetch(`${WP_EVENTS_API}/events?${params}`, {
      headers: { "User-Agent": BROWSER_UA },
    });
    if (!r.ok) return [];
    return ((await r.json()).events || []).map((e) => ({
      title: e.title,
      start: e.start_date,
      end: e.end_date,
      venue: e.venue?.venue || null,
      url: e.url,
    }));
  }

  let events = [];
  try {
    events = await fetchEvents(args?.search);
    if (!events.length && args?.search) events = await fetchEvents(null);
  } catch (e) {
    console.warn("events fetch failed:", e.message);
  }
  trace?.span({
    name: "get_upcoming_events",
    startTime: t,
    endTime: new Date(),
    output: events,
    metadata: { search: args?.search || null },
  });

  const instructions = `
        Today's date is ${today}.

        Answer using the live events list below — it was fetched from the USG events calendar just now and is current. Give dates, times, and locations plainly, and interpret relative words like "next" or "this week" against today's date.

        If the list is empty, say the calendar lookup found nothing (or the calendar couldn't be reached) and point the user to the calendar page instead of guessing.

        Be concise, accurate, and human.
      `;
  const userContent = `Live upcoming USG events (fetched just now):\n${JSON.stringify(events, null, 1)}\n\nCurrent user message:\n${message}`;
  const sources = [
    {
      source_title: "USG Events Calendar",
      source_url: "https://usg.usc.edu/calendar/",
      score: 1,
      source_modified: null,
      source_modified_year: null,
      evergreen: true,
    },
  ];
  return { instructions, userContent, sources, notice: "" };
}

const TRACKER_CSV_URL =
  process.env.TRACKER_CSV_URL ||
  "https://docs.google.com/spreadsheets/d/e/2PACX-1vTpDRrRtY-6BCNoe6psBsBa_7rkf_lTj1upbOeFHLM_J1fyRLSOULRcFBvcsFleXtNSPRJlAg5Avo9I/pub?output=csv";

const TRACKER_STATUS = {
  1: "Planning", 2: "In Progress", 3: "Almost Done",
  4: "Postponed/Paused", 5: "Completed", 6: "Ongoing",
};
const TRACKER_COMMITTEE = {
  1: "Executive", 2: "Advocacy", 3: "Senate/Legislative", 4: "Academic Affairs",
  5: "Accessibility", 6: "Affordability & Basic Needs",
  7: "Campus Infrastructure & Sustainability", 8: "External Affairs",
  9: "Wellness", 10: "Senate Bill",
};

// Minimal RFC-4180 CSV parse (quoted fields, escaped quotes).
function parseCsv(text) {
  const rows = [];
  let row = [], field = "", inQ = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQ) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQ = false;
      } else field += c;
    } else if (c === '"') inQ = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows;
}

// Live project-tracker counterpart of ragPrepare. The static page HTML
// carries stale snapshot numbers (the widget renders live), so project
// questions must come from the tracker's own published Google Sheet.
async function trackerPrepare(message, trace) {
  const t = new Date();
  let projects = [];
  try {
    const r = await fetch(TRACKER_CSV_URL, { redirect: "follow" });
    if (r.ok) {
      projects = parseCsv(await r.text())
        .slice(1)
        .filter((row) => row[0] && row[5])
        .map((row) => ({
          title: row[0].trim(),
          description: (row[1] || "").trim(),
          status: TRACKER_STATUS[Number(row[5])] || row[5],
          committee: TRACKER_COMMITTEE[Number(row[3])] || row[3],
        }));
    }
  } catch (e) {
    console.warn("tracker fetch failed:", e.message);
  }

  const byStatus = {};
  for (const p of projects) byStatus[p.status] = (byStatus[p.status] || 0) + 1;
  const summary =
    `Total projects on the tracker: ${projects.length} ` +
    `(the dashboard labels this count "Active Projects"). By status: ` +
    JSON.stringify(byStatus);
  const list = projects
    .map((p) => `- ${p.title} [${p.status}, ${p.committee}] ${p.description}`)
    .join("\n");

  trace?.span({
    name: "get_project_tracker",
    startTime: t,
    endTime: new Date(),
    output: { count: projects.length, byStatus },
  });

  const instructions = `
        Today's date is ${new Date().toISOString().slice(0, 10)}.

        Answer using the live USG project tracker data below — it was fetched just now from the tracker's own data source and is current. Be precise about statuses: the dashboard's "Active Projects" number is the total count of listed projects, which includes completed ones; give the by-status breakdown when the user asks about active work.

        If the list is empty, say the tracker couldn't be reached and point the user to the Legislative Branch page instead of guessing.

        Be concise, accurate, and human.
      `;
  const userContent = `Live USG project tracker (fetched just now):\n${summary}\n\nProjects:\n${list}\n\nCurrent user message:\n${message}`;
  const sources = [
    {
      source_title: "USG Legislative Branch — Live Project Tracker",
      source_url: "https://usg.usc.edu/legislative-branch/",
      score: 1,
      source_modified: null,
      source_modified_year: null,
      evergreen: true,
    },
  ];
  return { instructions, userContent, sources, notice: "" };
}

// LangGraph state machine: route -> (kb | events | tracker) -> END, shared by both
// endpoints. Node bodies are the same functions as before — the graph is
// orchestration only. KB prep still runs speculatively: the route node
// kicks it off before awaiting the router, so routing adds ~no latency on
// the common KB path; a wasted embed+rerank on the events path costs
// fractions of a cent.
const { StateGraph, Annotation, START, END } = require("@langchain/langgraph");

const AgentState = Annotation.Root({
  message: Annotation(),
  history: Annotation(),
  trace: Annotation(),
  route: Annotation(),
  ragPromise: Annotation(),
  prepared: Annotation(),
});

const agentGraph = new StateGraph(AgentState)
  .addNode("router", async (s) => ({
    // property order matters: start the speculative KB prep, THEN await
    // the router
    ragPromise: ragPrepare(s.message, s.trace).catch((e) => e),
    route: await routeTool(s.message, s.history, s.trace),
  }))
  .addNode("kb", async (s) => {
    const rag = await s.ragPromise;
    if (rag instanceof Error) throw rag;
    return { prepared: rag };
  })
  .addNode("events", async (s) => ({
    prepared: await eventsPrepare(s.message, s.route.args, s.trace),
  }))
  .addNode("tracker", async (s) => ({
    prepared: await trackerPrepare(s.message, s.trace),
  }))
  .addEdge(START, "router")
  .addConditionalEdges("router", (s) => s.route.tool, {
    kb: "kb",
    events: "events",
    tracker: "tracker",
  })
  .addEdge("kb", END)
  .addEdge("events", END)
  .addEdge("tracker", END)
  .compile();

async function agentPrepare(message, history, trace) {
  const out = await agentGraph.invoke({ message, history, trace });
  return out.prepared;
}

// One source per URL, keeping the best-scoring chunk's metadata.
function dedupSources(matches) {
  const byUrl = new Map();
  for (const chunk of matches) {
    const existing = byUrl.get(chunk.source_url);
    if (!existing || chunk.score > existing.score) {
      byUrl.set(chunk.source_url, {
        source_title: chunk.source_title,
        source_url: chunk.source_url,
        score: chunk.score,
        source_modified: chunk.source_modified || null,
        source_modified_year: chunk.source_modified_year || null,
        evergreen: Boolean(chunk.evergreen),
      });
    }
  }
  return Array.from(byUrl.values());
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

  const sources = dedupSources(matches);
  const { notice } = assessFreshness(sources);
  const userContent = `Context:\n${context}\n\nCurrent user message:\n${message}`;

  return { instructions, userContent, sources, notice };
}

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

// Thumbs up/down from the widget -> Langfuse score on the answer's trace.
app.post("/api/feedback", (req, res) => {
  const { traceId, value, comment } = req.body || {};
  if (!langfuse || typeof traceId !== "string" || ![0, 1].includes(value)) {
    return res.json({ ok: false });
  }
  langfuse.score({
    // stable id -> re-submissions (changed mind, added comment) upsert the
    // same score instead of double-counting
    id: `fb-${traceId}`,
    traceId,
    name: "user_feedback",
    value,
    comment: typeof comment === "string" ? comment.slice(0, 1000) : undefined,
  });
  langfuse.flushAsync().catch(() => {});
  res.json({ ok: true });
});

// Which DB is this instance actually talking to? (diagnostic — host is
// masked to a prefix, no credentials)
app.get("/health/db", async (_req, res) => {
  if (!pool) return res.json({ pool: false });
  try {
    const host = new URL(process.env.SUPABASE_DB_URL).hostname;
    const { rows } = await pool.query(
      "select current_database() as db, (select count(*)::int from chunks) as chunks"
    );
    res.json({ pool: true, host_prefix: host.slice(0, 12), ...rows[0] });
  } catch (e) {
    res.json({ pool: true, error: e.message });
  }
});

// Buffers span/generation/update calls until a real Langfuse trace exists.
// Lets retrieval run (and instrument itself) concurrently with the crisis
// check while the trace — with input + sessionId, which Langfuse only
// honors at creation — is created strictly after the crisis verdict.
// Crisis requests: buffer is discarded, zero events reach analytics.
function deferredTrace() {
  let real = null;
  const q = [];
  const fw = (m) => (o) => (real ? real[m](o) : q.push([m, o]));
  return {
    span: fw("span"),
    generation: fw("generation"),
    update: fw("update"),
    attach(t) {
      real = t;
      for (const [m, o] of q) t[m](o);
      q.length = 0;
    },
    get id() {
      return real ? real.id : null;
    },
  };
}

app.post("/api/chat", async (req, res) => {
  try {
    const { message, history, sessionId, error } = parseChatBody(req.body);
    if (error) {
      return res.status(400).json({ error });
    }

    // Crisis check and retrieval run concurrently; the crisis verdict still
    // gates the response — nothing is sent until it resolves.
    // Spans buffer locally; the real trace (with input + sessionId, which
    // Langfuse only honors at creation) is created after the crisis check.
    const trace = langfuse ? deferredTrace() : null;
    const ragPromise = agentPrepare(message, history, trace).catch((e) => e);
    const crisis = await crisisPayload(message, history);
    if (crisis) {
      // count-only marker; the buffered spans are discarded
      langfuse?.trace({ name: "crisis", tags: ["crisis"] });
      langfuse?.flushAsync().catch(() => {});
      return res.json(crisis);
    }
    trace?.attach(
      langfuse.trace({ name: "chat", input: message, sessionId: sessionId || undefined })
    );

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
      traceId: trace?.id || null,
    });
  } catch (err) {
    console.error(err);
    res.status(err.status || 500).json({ error: err.message || "Server error" });
  }
});

app.post("/api/chat/stream", async (req, res) => {
  try {
    const { message, history, sessionId, error } = parseChatBody(req.body);
    if (error) {
      return res.status(400).json({ error });
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
    // Deferred like /api/chat: buffered spans, trace created post-verdict.
    const trace = langfuse ? deferredTrace() : null;
    const ragPromise = agentPrepare(message, history, trace).catch((e) => e);
    const crisis = await crisisPayload(message, history);
    if (crisis) {
      langfuse?.trace({ name: "crisis", tags: ["crisis"] }); // count-only
      langfuse?.flushAsync().catch(() => {});
      send(crisis);
      send({ done: true });
      return res.end();
    }
    trace?.attach(
      langfuse.trace({ name: "chat-stream", input: message, sessionId: sessionId || undefined })
    );

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

    send({ done: true, sources, notice, crisis: false, traceId: trace?.id || null });
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
    console.log(
      pool
        ? "retrieval: pgvector (SUPABASE_DB_URL set)"
        : "retrieval: kb.json fallback — SUPABASE_DB_URL NOT set; kb.json is no longer committed, so production MUST have the env var"
    );
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
  cosineSimilarity,
  dedupSources,
  parseChatBody,
  routeTool,
  eventsPrepare,
  agentGraph,
  pool,
};