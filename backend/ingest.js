// backend/ingest.js
// Usage: set OPENAI_API_KEY in env, then `node ingest.js`
// Produces kb.json in the same folder.

require('dotenv').config();
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const cheerio = require('cheerio');
const OpenAI = require('openai').default;

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
if (!process.env.OPENAI_API_KEY) {
  console.error("Set OPENAI_API_KEY environment variable.");
  process.exit(1);
}

// ====== Edit this: replace with your 5 page URLs/titles ======
const PAGES = [
  { url: "https://usg.usc.edu/blog/2026/03/03/february-24th-2026-senate-press-release/", title: "Feb 24, 2026 Senate Press Release" },
  { url: "https://usg.usc.edu/blog/2025/10/13/fall-2025-senate-special-elections-result/", title: "Fall 2025 Senate Special Elections Result" },
  { url: "https://usg.usc.edu/resources/health-and-wellness/", title: "Health and Wellness Resources" },
  { url: "https://usg.usc.edu/executive-branch/", title: "USG Executive Branch" },
  { url: "https://usg.usc.edu/branches/programming/", title: "USG Programming Branch" }
];
// ===========================================================

const OUT = path.join(__dirname, 'kb.json');

function chunkText(text, maxChars = 800) {
  const sentences = text.split(/(?<=[.?!])\s+/);
  const chunks = [];
  let cur = "";
  for (const s of sentences) {
    if ((cur + " " + s).length > maxChars) {
      if (cur.trim()) chunks.push(cur.trim());
      cur = s;
    } else {
      cur = (cur + " " + s).trim();
    }
  }
  if (cur.trim()) chunks.push(cur.trim());
  return chunks;
}

async function fetchPageText(url) {
  const r = await axios.get(url, { timeout: 15000 });
  const $ = cheerio.load(r.data);
  // Prefer main/article, fall back to body
  let sel = $('main').text().trim();
  if (!sel) sel = $('article').text().trim();
  if (!sel) sel = $('body').text().trim();
  return sel.replace(/\s+/g, ' ').trim();
}

async function embed(text) {
  const resp = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: text
  });
  return resp.data[0].embedding;
}

(async () => {
  const chunks = [];
  for (const p of PAGES) {
    console.log("Fetching", p.url);
    try {
      const text = await fetchPageText(p.url);
      if (!text || text.length < 20) {
        console.warn("No text extracted from", p.url);
        continue;
      }
      const parts = chunkText(text, 900);
      for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        console.log(`Embedding ${p.title} chunk ${i+1}/${parts.length}`);
        const embedding = await embed(part);
        chunks.push({
          source_url: p.url,
          source_title: p.title,
          chunk_index: i,
          text: part,
          embedding
        });
      }
    } catch (e) {
      console.error("Error fetching/embedding", p.url, e.message);
    }
  }

  const data = { ingested_at: new Date().toISOString(), chunks };
  fs.writeFileSync(OUT, JSON.stringify(data, null, 2));
  console.log("Wrote", OUT, "with", chunks.length, "chunks");
})();