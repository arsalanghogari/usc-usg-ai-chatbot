const { test } = require("node:test");
const assert = require("node:assert");

// Neutralize external services before requiring the app (dotenv won't
// override pre-set vars).
process.env.OPENAI_API_KEY = process.env.OPENAI_API_KEY || "test-key";
process.env.SUPABASE_DB_URL = "";
process.env.LANGFUSE_SECRET_KEY = "";
process.env.LANGFUSE_PUBLIC_KEY = "";

const {
  cosineSimilarity,
  hasCrisisKeywords,
  assessFreshness,
  dedupSources,
  parseChatBody,
  parseIcsDate,
} = require("../server.js");

test("cosineSimilarity", () => {
  assert.equal(cosineSimilarity([1, 0], [0, 1]), 0);
  assert.ok(Math.abs(cosineSimilarity([1, 2, 3], [1, 2, 3]) - 1) < 1e-9);
  assert.ok(cosineSimilarity([1, 0], [-1, 0]) < 0);
});

test("hasCrisisKeywords", () => {
  assert.ok(hasCrisisKeywords("I want to kill myself"));
  assert.ok(hasCrisisKeywords("thinking about suicide"));
  assert.ok(hasCrisisKeywords("i dont want to be here anymore"));
  assert.ok(hasCrisisKeywords("I do not want to live"));
  assert.ok(!hasCrisisKeywords("when is the next senate meeting?"));
  assert.ok(!hasCrisisKeywords("this deadline is killing me")); // no false positive
});

test("assessFreshness", () => {
  const now = new Date("2026-07-20");
  const stale = assessFreshness([{ evergreen: false, source_modified_year: 2025 }], now);
  assert.ok(stale.notice.includes("2025"));
  assert.ok(stale.notice.includes("contact"));
  assert.equal(assessFreshness([{ evergreen: true, source_modified_year: 2020 }], now).notice, "");
  assert.equal(assessFreshness([{ evergreen: false, source_modified_year: 2026 }], now).notice, "");
  assert.equal(assessFreshness([{ evergreen: false, source_modified_year: null }], now).notice, "");
});

test("dedupSources keeps best score per url", () => {
  const out = dedupSources([
    { source_url: "a", source_title: "A", score: 0.5, evergreen: false },
    { source_url: "a", source_title: "A", score: 0.9, evergreen: false },
    { source_url: "b", source_title: "B", score: 0.4, evergreen: true },
  ]);
  assert.equal(out.length, 2);
  assert.equal(out.find((s) => s.source_url === "a").score, 0.9);
  assert.equal(out.find((s) => s.source_url === "b").evergreen, true);
});

test("parseChatBody", () => {
  assert.equal(parseChatBody({}).error, "Missing message.");
  assert.equal(parseChatBody({ message: "   " }).error, "Missing message.");
  assert.ok(parseChatBody({ message: "x".repeat(2001) }).error.includes("too long"));
  const ok = parseChatBody({
    message: "hi",
    history: [
      { role: "user", content: "q" },
      { role: "hacker", content: "bad" },
      { role: "assistant", content: 42 },
    ],
  });
  assert.equal(ok.message, "hi");
  assert.deepEqual(ok.history, [{ role: "user", content: "q" }]);
});

test("parseIcsDate", () => {
  // TZID local wall-clock (senate meeting shape)
  const tz = parseIcsDate("DTSTART;TZID=America/Los_Angeles:20250826T190000");
  assert.equal(tz.tod, "7:00 PM");
  assert.ok(tz.display.includes("Tuesday, August 26, 2025, 7:00 PM"));
  // UTC Z shape converts to LA time
  const z = parseIcsDate("DTSTART:20251106T013000Z");
  assert.equal(z.tod, "5:30 PM");
  // all-day
  assert.ok(parseIcsDate("DTSTART;VALUE=DATE:20251106").display.includes("all day"));
  // garbage
  assert.equal(parseIcsDate("DTSTART:nope"), null);
});

test("expandSiblings pulls whole pages in order, keeps picked scores", async () => {
  const { expandSiblings, loadKb } = require("../server.js");
  // kb.json is not committed; CI may have none yet (integration's fixture
  // writes in its own before hook) — write the same tiny fixture if absent.
  const fs = require("node:fs");
  const path = require("node:path");
  const kbPath = path.join(__dirname, "..", "kb.json");
  if (!fs.existsSync(kbPath)) {
    fs.writeFileSync(
      kbPath,
      JSON.stringify({
        chunks: [0, 1].map((i) => ({
          source_url: "https://usg.usc.edu/branches/funding/",
          source_title: "USG Funding Department",
          chunk_index: i,
          text: `Funding fixture chunk ${i}`,
          evergreen: true,
          embedding: Array(1536).fill(0.1),
        })),
      })
    );
  }
  // use any multi-chunk page so the test works on real KB and fixture alike
  const byUrl = {};
  for (const c of loadKb().chunks) (byUrl[c.source_url] ||= []).push(c);
  const page = Object.values(byUrl)
    .find((p) => p.length >= 2)
    .sort((a, b) => a.chunk_index - b.chunk_index);
  const pick = { ...page[page.length - 1], score: 0.91 };
  delete pick.embedding;
  const out = await expandSiblings([pick]);
  assert.equal(out.length, page.length); // whole page came back
  assert.deepEqual(out.map((c) => c.chunk_index), page.map((c) => c.chunk_index));
  assert.equal(out.find((c) => c.chunk_index === pick.chunk_index).score, 0.91);
  assert.ok(out.every((c) => !c.embedding)); // no embeddings leaked into context
});
