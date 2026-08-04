// Academic-work guardrail tests: router route (layer 1) and output rail
// (layer 3). The mock answers router calls with a real function_call —
// "essay" in the message routes to redirect_academic_help, otherwise to
// search_knowledge_base — and generation calls with tutor-bait text when the
// message carries the TUTOR-BAIT sentinel, so the rail's replacement path is
// reachable end-to-end.

const { test, before, after } = require("node:test");
const assert = require("node:assert");
const http = require("node:http");

const TUTOR_BAIT = "TUTOR-BAIT";

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
      const parsed = JSON.parse(body);
      if (parsed.tools) {
        // router call: decide from the USER MESSAGE only — the tool
        // descriptions themselves contain the word "essay"
        const userText = (parsed.input || [])
          .filter((m) => m.role === "user")
          .map((m) => (typeof m.content === "string" ? m.content : JSON.stringify(m.content)))
          .join(" ");
        let name = "search_knowledge_base";
        let args = "{}";
        if (userText.includes("essay")) name = "redirect_academic_help";
        else if (userText.includes("harass")) name = "redirect_safety_report";
        else if (userText.includes("housing")) {
          name = "redirect_campus_office";
          args = JSON.stringify({ office: "housing" });
        } else if (userText.includes("mystery office")) {
          name = "redirect_campus_office";
          args = JSON.stringify({ office: "nonexistent_office" });
        }
        return res.end(
          JSON.stringify({
            id: "resp_mock",
            object: "response",
            status: "completed",
            output: [{ type: "function_call", name, arguments: args, call_id: "c1" }],
            usage: { input_tokens: 1, output_tokens: 1 },
          })
        );
      }
      const out = body.includes(TUTOR_BAIT)
        ? "Sure — I can help you brainstorm and outline your essay, then revise the draft."
        : "MOCK ANSWER";
      if (parsed.stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.write(`data: ${JSON.stringify({ type: "response.output_text.delta", delta: out })}\n\n`);
        res.write(
          `data: ${JSON.stringify({ type: "response.completed", response: { usage: { input_tokens: 1, output_tokens: 1 } } })}\n\n`
        );
        res.write("data: [DONE]\n\n");
        return res.end();
      }
      return res.end(
        JSON.stringify({
          id: "resp_mock",
          object: "response",
          status: "completed",
          output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: out, annotations: [] }] }],
          output_text: out,
          usage: { input_tokens: 1, output_tokens: 1 },
        })
      );
    }
    res.statusCode = 404;
    res.end("{}");
  });
});

let server, app, api, srv;

before(async () => {
  const fs = require("node:fs");
  const path = require("node:path");
  const kbPath = path.join(__dirname, "guardrail-fixture-kb.json");
  fs.writeFileSync(
    kbPath,
    JSON.stringify({
      chunks: [
        {
          source_url: "https://usg.usc.edu/branches/funding/",
          source_title: "Funding",
          chunk_index: 0,
          text: "Apply for USG funding via EngageSC.",
          source_modified_year: 2026,
          evergreen: true,
          embedding: Array(1536).fill(0.1),
        },
      ],
    })
  );
  process.env.KB_PATH = kbPath;
  await new Promise((r) => mock.listen(0, r));
  process.env.OPENAI_API_KEY = "test-key";
  process.env.OPENAI_BASE_URL = `http://localhost:${mock.address().port}/v1`;
  process.env.SUPABASE_DB_URL = "";
  process.env.LANGFUSE_SECRET_KEY = "";
  process.env.LANGFUSE_PUBLIC_KEY = "";
  process.env.RERANK = "0";

  server = require("../server.js");
  app = server.app;
  srv = app.listen(0);
  api = `http://localhost:${srv.address().port}`;
});

after(() => {
  require("node:fs").unlinkSync(require("node:path").join(__dirname, "guardrail-fixture-kb.json"));
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

// ---- layer 3 unit: the output rail regex ----------------------------------

test("tutor-offer detector catches offer + academic-noun combinations", () => {
  for (const bad of [
    "I can't write it, but I can help you brainstorm your essay.",
    "Happy to outline your assignment structure.",
    "Let's draft a thesis statement together.",
    "I can help you think through the essay prompt.",
    "We can revise your term paper for clarity.",
  ]) {
    assert.ok(server.hasTutorOfferLanguage(bad), `rail missed: ${bad}`);
  }
});

test("tutor-offer detector stays quiet on normal USG answers", () => {
  for (const ok of [
    "Apply for USG funding via EngageSC by Wednesday 11:59 PM.",
    "The essay contest deadline is Friday.", // noun without an offer
    "I can help you find the funding application.", // offer without academic noun
    server.HOMEWORK_REPLY, // the canned reply must pass its own rail
  ]) {
    assert.ok(!server.hasTutorOfferLanguage(ok), `false positive: ${ok}`);
  }
});

// ---- layer 1: router-level deterministic decline --------------------------

test("essay request routes to the canned referral without generation", async () => {
  const { status, body } = await chat({ message: "Can you help me write my history essay?" });
  assert.equal(status, 200);
  assert.equal(body.crisis, false);
  assert.equal(body.answer, server.HOMEWORK_REPLY);
  assert.ok(body.sources.some((s) => s.source_url.includes("resources-guides")));
});

test("normal question still routes to the KB", async () => {
  const { body } = await chat({ message: "How do I get funding for my club?" });
  assert.equal(body.answer, "MOCK ANSWER");
  assert.ok(body.sources.some((s) => s.source_url.includes("/branches/funding/")));
});

// ---- layer 3 e2e: rail replaces a tutor-offer that slipped through --------

test("output rail replaces a generated tutor-offer with the canned referral", async () => {
  // no "essay" in the user message -> routes kb; generation returns bait
  const { body } = await chat({ message: `Tell me about ${TUTOR_BAIT} funding` });
  assert.equal(body.answer, server.HOMEWORK_REPLY);
  assert.ok(body.sources.some((s) => s.source_url.includes("resources-guides")));
});

// ---- safety guardrail: verified contacts, never generated -----------------

test("safety report routes to verified contacts without generation", async () => {
  const { body } = await chat({ message: "How do I report harassment by another student?" });
  assert.equal(body.crisis, false);
  assert.equal(body.answer, server.SAFETY_REPLY);
  assert.ok(body.answer.includes("(213) 740-4321"), "DPS number missing");
  assert.ok(body.answer.includes("911"), "911 missing");
  assert.ok(body.sources.some((s) => s.source_url.includes("/resources/emergencies/")));
});

test("safety reply passes the other guardrails' own checks", () => {
  assert.ok(!server.hasTutorOfferLanguage(server.SAFETY_REPLY));
  assert.ok(!server.hasDisallowedAssistantLanguage(server.SAFETY_REPLY));
});

// ---- directory guardrail: right office, right link ------------------------

test("campus-office question routes to the office's official site", async () => {
  const { body } = await chat({ message: "How do I apply for USC housing?" });
  assert.equal(body.crisis, false);
  assert.ok(body.answer.includes("https://housing.usc.edu"), "housing link missing");
  assert.ok(body.answer.toLowerCase().includes("not usg"));
  assert.ok(body.sources.some((s) => s.source_url === "https://housing.usc.edu"));
});

test("unknown office enum falls back to the campus resources directory", async () => {
  const { body } = await chat({ message: "Who handles the mystery office thing?" });
  assert.ok(body.answer.includes("https://studentaffairs.usc.edu/campus-resources/"));
});

test("every directory entry has a usc.edu-family url", () => {
  for (const [key, office] of Object.entries(server.DIRECTORY)) {
    assert.match(office.url, /^https:\/\/[a-z.]*usc\.edu/, `suspicious url for ${key}: ${office.url}`);
  }
});

// ---- stream endpoint: replace event, no token drip ------------------------

test("stream endpoint sends the referral as a replace event, not deltas", async () => {
  const r = await fetch(`${api}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "Please help write my essay" }),
  });
  const events = (await r.text())
    .split("\n\n")
    .filter((p) => p.startsWith("data: "))
    .map((p) => JSON.parse(p.slice(6)));
  assert.ok(!events.some((e) => e.delta), "referral must not token-stream");
  const rep = events.find((e) => e.replace);
  assert.equal(rep.answer, server.HOMEWORK_REPLY);
  assert.ok(events.at(-1).done);
});

test("stream output rail retracts streamed tutor-offer via replace event", async () => {
  const r = await fetch(`${api}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: `Tell me about ${TUTOR_BAIT} funding` }),
  });
  const events = (await r.text())
    .split("\n\n")
    .filter((p) => p.startsWith("data: "))
    .map((p) => JSON.parse(p.slice(6)));
  const rep = events.find((e) => e.replace);
  assert.ok(rep, "no replace event after tutor-offer stream");
  assert.equal(rep.answer, server.HOMEWORK_REPLY);
});
