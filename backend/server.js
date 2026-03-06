// backend/server.js
// Usage: set OPENAI_API_KEY then `node server.js`
// Exposes POST /chat that uses kb.json

require('dotenv').config();
const fs = require('fs');
const path = require('path');
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const OpenAI = require('openai').default;

const app = express();
app.use(cors());
app.use(bodyParser.json());

const OPENAI_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_KEY) {
  console.error("Set OPENAI_API_KEY in env");
  process.exit(1);
}
const client = new OpenAI({ apiKey: OPENAI_KEY });

const KB_PATH = path.join(__dirname, 'kb.json');

function loadKb() {
  if (!fs.existsSync(KB_PATH)) return { chunks: [] };
  return JSON.parse(fs.readFileSync(KB_PATH, 'utf8'));
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

const SENSITIVE_KEYWORDS = ['suicide','kill myself','self harm','assault','rape','emergency'];

function detectSensitive(text) {
  const t = text.toLowerCase();
  for (const k of SENSITIVE_KEYWORDS) if (t.includes(k)) return true;
  return false;
}

function checkCitations(answer) {
  return answer.includes('[source:');
}

app.post('/chat', async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: 'no message provided' });

    if (detectSensitive(message)) {
      return res.json({
        answer: "This looks urgent or sensitive. Please contact emergency services or your counseling center. If you are in immediate danger call local emergency services.",
        sources: [],
        flags: { fallback: true, sensitive: true }
      });
    }

    const kb = loadKb();
    if (!kb.chunks || kb.chunks.length === 0) {
      return res.json({ answer: "KB empty. Run ingestion.", sources: [], flags: { fallback: true }});
    }

    // 1) embed query
    const embResp = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: message
    });
    const qvec = embResp.data[0].embedding;

    // 2) score
    const scored = kb.chunks.map(c => ({ c, score: cosine(qvec, c.embedding) }));
    scored.sort((a,b) => b.score - a.score);
    const TOP_K = 4;
    const top = scored.slice(0, TOP_K).filter(s => s.score > 0.50);

    if (top.length === 0) {
      // fallback
      const allSources = Array.from(new Set(kb.chunks.map(x => x.source_url)))
        .map(url => ({ title: kb.chunks.find(c=>c.source_url===url).source_title, url }));
      return res.json({
        answer: "I couldn't find an answer in the approved pages. Try checking these pages directly.",
        sources: allSources,
        flags: { fallback: true }
      });
    }

    // 3) construct context
    const contextText = top.map(t => `---\n${t.c.text}\n[source:${t.c.source_title} | ${t.c.source_url}]\n`).join('\n');

    const system = `You are a helpful assistant. Answer using ONLY the context sections below. Do NOT invent facts. For any factual claim include a citation tag like [source:TITLE | URL]. Keep answers concise (1-3 sentences).`;

    const userPrompt = `Context:\n${contextText}\n\nUser question: "${message}"\n\nAnswer:`;

    const gen = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: system },
        { role: "user", content: userPrompt }
      ],
      max_tokens: 300,
      temperature: 0.0
    });

    const answer = gen.choices[0].message.content.trim();
    const hasCitation = checkCitations(answer);
    const finalAnswer = hasCitation ? answer : (answer + "\n\nNote: I couldn't find an explicit citation; please check the source pages.");

    const unique = {};
    top.forEach(t => { unique[t.c.source_url] = { title: t.c.source_title, url: t.c.source_url }; });
    const sources = Object.values(unique);

    res.json({ answer: finalAnswer, sources, flags: { fallback: !hasCitation } });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'server error', details: err.message });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));