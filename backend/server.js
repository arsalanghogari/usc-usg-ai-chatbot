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

if (!OPENAI_API_KEY) {
  console.error("Set OPENAI_API_KEY in your environment.");
  process.exit(1);
}

const client = new OpenAI({ apiKey: OPENAI_API_KEY });
const KB_PATH = path.join(__dirname, "kb.json");

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "..", "frontend")));

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

    const crisis =
      await isCrisisMessage(message) ||
      historySuggestsCrisis(history);

    if (crisis) {
      const crisisResponse = await client.responses.create({
        model: CHAT_MODEL,
        instructions: CRISIS_INSTRUCTIONS,
        input: [
          {
            role: "user",
            content: [
              {
                type: "input_text",
                text: message,
              },
            ],
          },
        ],
      });

      const answer = crisisResponse.output_text || pickCrisisReply(message);

      if (hasDisallowedAssistantLanguage(answer)) {
        return res.json({
          crisis: true,
          answer: pickCrisisReply(message),
          sources: CRISIS_SOURCES,
        });
      }

      return res.json({
        crisis: true,
        answer,
        sources: CRISIS_SOURCES,
      });
    }

    const kb = loadKb();
    if (!kb.chunks.length) {
      return res.status(400).json({ error: "kb.json is empty. Run ingestion first." });
    }

    const queryEmbedding = await embed(message);
    const matches = topChunks(queryEmbedding, kb.chunks, 4);

    const context = matches
      .map(
        (chunk, idx) =>
          `[${idx + 1}] ${chunk.source_title}\n${chunk.source_url}\n${chunk.text}`
      )
      .join("\n\n");

    const response = await client.responses.create({
      model: CHAT_MODEL,
      instructions:
        "Answer the user's question using only the provided context. If the context is insufficient, say so plainly. Be concise and accurate.",
      input: `Context:\n${context}\n\nUser question:\n${message}`,
    });

    const uniqueSourcesMap = new Map();

    for (const chunk of matches) {
      const key = chunk.source_url;
      const existing = uniqueSourcesMap.get(key);

      if (!existing || chunk.score > existing.score) {
        uniqueSourcesMap.set(key, {
          source_title: chunk.source_title,
          source_url: chunk.source_url,
          score: chunk.score,
        });
      }
    }

    res.json({
      answer: response.output_text || "",
      sources: Array.from(uniqueSourcesMap.values()),
      crisis: false,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || "Server error" });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});