const { test, before, after } = require("node:test");
const assert = require("node:assert");
const http = require("node:http");

// Mock the OpenAI API with a local server, then point the SDK at it via
// OPENAI_BASE_URL. Set everything before requiring server.js (dotenv does
// not override pre-set vars).
const mock = http.createServer((req, res) => {
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    res.setHeader("Content-Type", "application/json");
    if (req.url.includes("/embeddings")) {
      const n = Array.isArray(JSON.parse(body).input) ? JSON.parse(body).input.length : 1;
      res.end(
        JSON.stringify({
          data: Array.from({ length: n }, () => ({ embedding: Array(1536).fill(0.1) })),
        })
      );
    } else if (req.url.includes("/moderations")) {
      res.end(JSON.stringify({ results: [{ flagged: false, categories: {} }] }));
    } else if (req.url.includes("/responses")) {
      res.end(
        JSON.stringify({
          id: "resp_mock",
          object: "response",
          status: "completed",
          output: [
            {
              type: "message",
              role: "assistant",
              content: [{ type: "output_text", text: "MOCK ANSWER", annotations: [] }],
            },
          ],
          output_text: "MOCK ANSWER",
          usage: { input_tokens: 1, output_tokens: 1 },
        })
      );
    } else {
      res.statusCode = 404;
      res.end("{}");
    }
  });
});

let app, api;

before(async () => {
  // kb.json is not committed (Postgres is the store of record), so CI has
  // neither file nor DB — write a tiny fixture for the fallback path.
  const fs = require("node:fs");
  const path = require("node:path");
  const kbPath = path.join(__dirname, "..", "kb.json");
  if (!fs.existsSync(kbPath)) {
    fs.writeFileSync(
      kbPath,
      JSON.stringify({
        ingested_at: "2026-01-01T00:00:00Z",
        chunks: [0, 1].map((i) => ({
          source_url: "https://usg.usc.edu/branches/funding/",
          source_title: "USG Funding Department",
          chunk_index: i,
          text: `Funding fixture chunk ${i}: apply for USG funding via EngageSC.`,
          source_modified: "2026-06-09T17:18:08Z",
          source_modified_year: 2026,
          evergreen: true,
          embedding: Array(1536).fill(0.1),
        })),
      })
    );
  }

  await new Promise((r) => mock.listen(0, r));
  process.env.OPENAI_API_KEY = "test-key";
  process.env.OPENAI_BASE_URL = `http://localhost:${mock.address().port}/v1`;
  process.env.SUPABASE_DB_URL = "";
  process.env.LANGFUSE_SECRET_KEY = "";
  process.env.LANGFUSE_PUBLIC_KEY = "";
  process.env.RERANK = "0"; // deterministic single-stage retrieval in tests

  app = require("../server.js").app;
  const srv = app.listen(0);
  api = `http://localhost:${srv.address().port}`;
});

after(() => {
  mock.close();
  process.exit(0); // node:test keeps the express handle otherwise
});

async function chat(payload) {
  const r = await fetch(`${api}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return { status: r.status, body: await r.json() };
}

test("normal question returns answer + sources from kb.json", async () => {
  const { status, body } = await chat({ message: "How do I get funding?" });
  assert.equal(status, 200);
  assert.equal(body.crisis, false);
  assert.ok(body.answer.startsWith("MOCK ANSWER"));
  assert.ok(Array.isArray(body.sources) && body.sources.length > 0);
  assert.ok(body.sources[0].source_url.startsWith("https://"));
});

test("crisis keywords route to crisis payload with 988 sources", async () => {
  const { status, body } = await chat({ message: "I want to kill myself" });
  assert.equal(status, 200);
  assert.equal(body.crisis, true);
  assert.ok(body.sources.some((s) => s.source_url.includes("988")));
});

test("empty and oversized messages are rejected", async () => {
  assert.equal((await chat({ message: "" })).status, 400);
  assert.equal((await chat({ message: "x".repeat(2001) })).status, 400);
});

test("stream endpoint emits deltas or full events and a done event", async () => {
  const r = await fetch(`${api}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "How do I get funding?" }),
  });
  assert.equal(r.status, 200);
  assert.match(r.headers.get("content-type"), /text\/event-stream/);
  const text = await r.text();
  const events = text
    .split("\n\n")
    .filter((p) => p.startsWith("data: "))
    .map((p) => JSON.parse(p.slice(6)));
  assert.ok(events.some((e) => e.done));
});
