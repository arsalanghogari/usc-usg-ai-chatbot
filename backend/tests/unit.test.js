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

test("stripDisallowedUrls", () => {
  const { stripDisallowedUrls } = require("../server.js");
  // USC and calendar hosts pass through
  assert.equal(
    stripDisallowedUrls("see https://usg.usc.edu/funding/ and https://usc.edu/"),
    "see https://usg.usc.edu/funding/ and https://usc.edu/"
  );
  assert.ok(stripDisallowedUrls("https://calendar.google.com/x").includes("calendar.google.com"));
  // everything else is removed, including www. bare links and lookalike hosts
  assert.equal(stripDisallowedUrls("go to https://evil.com/login now"), "go to [link removed] now");
  assert.equal(stripDisallowedUrls("go to www.evil.com/login now"), "go to [link removed] now");
  assert.equal(stripDisallowedUrls("https://usc.edu.evil.com/"), "[link removed]");
  assert.equal(stripDisallowedUrls("https://notusc.edu/"), "[link removed]");
  // stripping happens inside parseChatBody for message and history
  const parsed = parseChatBody({
    message: "click https://evil.com/a",
    history: [{ role: "user", content: "https://evil.com/b" }],
  });
  assert.equal(parsed.message, "click [link removed]");
  assert.equal(parsed.history[0].content, "[link removed]");
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

// The August failure this exists to prevent: a bare name is nearly meaningless
// to an embedding, so dense retrieval ranked prose above the roster page that
// actually lists the person and the bot said they weren't in USG at all.
test("hybrid retrieval surfaces a roster chunk dense ranking misses", () => {
  const { topChunks } = require("../server.js");
  const chunk = (url, i, text, embedding) => ({
    source_url: url, source_title: url, chunk_index: i, text, embedding,
  });
  const corpus = [
    // three chunks the query embedding "likes", none of which name her
    chunk("https://usg.usc.edu/blog/senate-press-release/", 0, "The senate discussed advocacy work at length.", [1, 0]),
    chunk("https://usg.usc.edu/blog/committee-news/", 1, "Committee leadership met to plan advocacy programming.", [0.99, 0.14]),
    chunk("https://usg.usc.edu/blog/older-post/", 2, "More prose about USG advocacy and programming.", [0.98, 0.2]),
    // the roster page: names her once, embeds nowhere near the query
    chunk("https://usg.usc.edu/meet-our-team-legislative/", 3, "Priya Raghunathan, APASA Advocacy Liaison", [0, 1]),
  ];
  const q = [1, 0];

  const dense = topChunks(q, corpus, 3, null);
  assert.ok(!dense.some((c) => c.chunk_index === 3), "fixture invalid: dense already finds the roster");

  const hybrid = topChunks(q, corpus, 3, "who is priya raghunathan");
  assert.ok(hybrid.some((c) => c.chunk_index === 3), "lexical channel did not surface the roster chunk");
  assert.ok(hybrid.every((c) => typeof c.score === "number"));
});

test("rrfFuse ranks by summed reciprocal rank, deduping across channels", () => {
  const { rrfFuse } = require("../server.js");
  const a = { source_url: "u", chunk_index: 1 };
  const b = { source_url: "u", chunk_index: 2 };
  const c = { source_url: "u", chunk_index: 3 };
  // b is second in both channels; a and c each lead one. Agreement wins.
  const out = rrfFuse([[a, b], [c, b]], 3);
  assert.equal(out.length, 3, "the same chunk in both channels must not double up");
  assert.equal(out[0].chunk_index, 2);
});

// "assembly" means a Programming Department community assembly on the USG
// site, so retrieval gets that hint; everything else is passed through.
test("retrievalQuery biases assembly questions, leaves others alone", () => {
  const { retrievalQuery } = require("../server.js");
  assert.match(retrievalQuery("when do the assemblies meet"), /Programming Department/);
  assert.equal(retrievalQuery("who is the USG president"), "who is the USG president");
});
