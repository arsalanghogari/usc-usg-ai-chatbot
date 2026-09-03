// Retrieval-endpoint integration tests with a stubbed OpenAI client that
// CAPTURES request bodies, so we can assert what actually reaches the model:
// sibling-chunk expansion, source dedup, staleness notice, history threading.
//
// Fixture geometry (RERANK=0, all embeddings equal, stable sort):
//   [ old-blog c0, funding c0..c4 ]  — retrieval takes the first k=4,
// so funding c3/c4 are NOT retrieved and only sibling expansion can put
// them in the prompt. The old blog (2023, non-evergreen) must trigger the
// staleness notice.

const { test, before, after } = require("node:test");
const assert = require("node:assert");
const http = require("node:http");

const captured = { responses: [] };

const mock = http.createServer((req, res) => {
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    res.setHeader("Content-Type", "application/json");
    if (req.url.includes("/embeddings")) {
      const n = Array.isArray(JSON.parse(body).input) ? JSON.parse(body).input.length : 1;
      return res.end(
        JSON.stringify({ data: Array.from({ length: n }, () => ({ embedding: Array(1536).fill(0.1) })) })
      );
    }
    if (req.url.includes("/moderations")) {
      return res.end(JSON.stringify({ results: [{ flagged: false, categories: {} }] }));
    }
    if (req.url.includes("/responses")) {
      captured.responses.push(JSON.parse(body));
      return res.end(
        JSON.stringify({
          id: "resp_mock",
          object: "response",
          status: "completed",
          output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "MOCK ANSWER", annotations: [] }] }],
          output_text: "MOCK ANSWER",
          usage: { input_tokens: 1, output_tokens: 1 },
        })
      );
    }
    res.statusCode = 404;
    res.end("{}");
  });
});

const FUNDING_URL = "https://usg.usc.edu/branches/funding/";
const OLD_BLOG_URL = "https://usg.usc.edu/blog/2023/10/16/funding-resources/";

let app, api, srv;

before(async () => {
  const fs = require("node:fs");
  const path = require("node:path");
  const kbPath = path.join(__dirname, "retrieval-fixture-kb.json");
  const chunk = (url, title, i, text, year, evergreen) => ({
    source_url: url,
    source_title: title,
    chunk_index: i,
    text,
    source_modified: `${year}-06-09T00:00:00Z`,
    source_modified_year: year,
    evergreen,
    embedding: Array(1536).fill(0.1),
  });
  fs.writeFileSync(
    kbPath,
    JSON.stringify({
      chunks: [
        chunk(OLD_BLOG_URL, "Funding Resources (2023)", 0, "OLD-BLOG-MARKER: maximum of $8k a year.", 2023, false),
        ...[0, 1, 2, 3, 4].map((i) =>
          chunk(FUNDING_URL, "Funding", i, `FUNDING-CHUNK-${i}-MARKER: funding policy part ${i}.`, 2026, true)
        ),
      ],
    })
  );
  process.env.KB_PATH = kbPath;
  await new Promise((r) => mock.listen(0, r));
  process.env.OPENAI_API_KEY = "test-key";
  process.env.OPENAI_BASE_URL = `http://localhost:${mock.address().port}/v1`;
  process.env.SUPABASE_DB_URL = "";
process.env.ROSTER_PUB_URL = ""; // no roster fetch in tests
  process.env.LANGFUSE_SECRET_KEY = "";
  process.env.LANGFUSE_PUBLIC_KEY = "";
  process.env.RERANK = "0";

  app = require("../server.js").app;
  srv = app.listen(0);
  api = `http://localhost:${srv.address().port}`;
});

after(() => {
  require("node:fs").unlinkSync(require("node:path").join(__dirname, "retrieval-fixture-kb.json"));
  srv.closeAllConnections?.();
  srv.close();
  mock.closeAllConnections?.();
  mock.close();
});

async function chat(payload) {
  const r = await fetch(`${api}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return { status: r.status, body: await r.json() };
}

test("sibling expansion puts non-retrieved chunks of a picked page into the prompt", async () => {
  captured.responses.length = 0;
  const { status, body } = await chat({ message: "How do I get funding?" });
  assert.equal(status, 200);
  assert.equal(body.crisis, false);
  const gen = captured.responses.at(-1);
  const prompt = JSON.stringify(gen.input);
  // k=4 retrieval can only have picked old-blog + funding 0-2; chunks 3 and 4
  // reach the prompt only via expandSiblings
  assert.ok(prompt.includes("FUNDING-CHUNK-3-MARKER"), "sibling chunk 3 missing from prompt");
  assert.ok(prompt.includes("FUNDING-CHUNK-4-MARKER"), "sibling chunk 4 missing from prompt");
  assert.ok(prompt.includes("OLD-BLOG-MARKER"), "retrieved old-blog chunk missing from prompt");
});

test("sources are deduped to one entry per page", async () => {
  const { body } = await chat({ message: "How do I get funding?" });
  const urls = body.sources.map((s) => s.source_url);
  assert.deepEqual([...new Set(urls)].sort(), urls.slice().sort(), "duplicate source urls");
  assert.ok(urls.includes(FUNDING_URL));
  assert.ok(urls.includes(OLD_BLOG_URL));
});

test("stale non-evergreen source appends the dated staleness notice", async () => {
  const { body } = await chat({ message: "What is the funding maximum?" });
  assert.ok(body.answer.includes("⚠️"), "staleness notice missing");
  assert.ok(body.answer.includes("2023"), "notice does not name the stale year");
});

test("conversation history is threaded through to generation", async () => {
  captured.responses.length = 0;
  const history = [
    { role: "user", content: "HISTORY-TURN-MARKER how do I get funding?" },
    { role: "assistant", content: "Apply on EngageSC." },
  ];
  await chat({ message: "What about travel?", history });
  const gen = captured.responses.at(-1);
  const inputStr = JSON.stringify(gen.input);
  assert.ok(inputStr.includes("HISTORY-TURN-MARKER"), "history turn not passed to the model");
});

test("feedback endpoint rejects malformed ratings", async () => {
  for (const payload of [
    {},
    { traceId: "t1" }, // no value
    { traceId: "t1", value: 5 }, // out of range
    { traceId: "t1", value: "up" }, // wrong type
    { traceId: 42, value: 1 }, // traceId wrong type
  ]) {
    const r = await fetch(`${api}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    assert.deepEqual(await r.json(), { ok: false }, `accepted bad payload: ${JSON.stringify(payload)}`);
  }
});
