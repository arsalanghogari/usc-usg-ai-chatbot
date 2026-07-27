// Crisis-pathway guardrail tests. A false negative here (a student in crisis
// routed to the RAG chatbot instead of 988 resources) is the worst possible
// failure of this system, so this file is deliberately exhaustive about the
// deterministic layers: keyword net, history net, moderation wiring and its
// failure mode, and the disallowed-language guardrail on generated replies.
//
// The mock OpenAI server is content-keyed: sentinel strings in the user
// message steer /moderations and /responses, so every branch is reachable
// without ordering dependence.

const { test, before, after } = require("node:test");
const assert = require("node:assert");
const http = require("node:http");

const MOD_FLAG = "please review my situation"; // moderation-flagged, no keywords
const MOD_DOWN = "moderation outage probe"; // /moderations returns 500
const BAD_STYLE = "kill myself with bad style"; // crisis + disallowed generation

const mock = http.createServer((req, res) => {
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    res.setHeader("Content-Type", "application/json");
    const text = body || "";
    if (req.url.includes("/moderations")) {
      if (text.includes(MOD_DOWN)) {
        res.statusCode = 500;
        return res.end("{}");
      }
      return res.end(
        JSON.stringify({
          results: [
            text.includes(MOD_FLAG)
              ? { flagged: true, categories: { "self-harm/intent": true } }
              : { flagged: false, categories: {} },
          ],
        })
      );
    }
    if (req.url.includes("/embeddings")) {
      const n = Array.isArray(JSON.parse(text).input) ? JSON.parse(text).input.length : 1;
      return res.end(
        JSON.stringify({ data: Array.from({ length: n }, () => ({ embedding: Array(1536).fill(0.1) })) })
      );
    }
    if (req.url.includes("/responses")) {
      const out = text.includes(BAD_STYLE)
        ? "I can help — let me know what you would like to do next."
        : "MOCK CRISIS ANSWER: please call or text 988.";
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
  const kbPath = path.join(__dirname, "crisis-fixture-kb.json");
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
  require("node:fs").unlinkSync(require("node:path").join(__dirname, "crisis-fixture-kb.json"));
  // close handles instead of process.exit — exit() races the runner and
  // silently drops the last declared test on node 25
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

// ---- keyword net: false negatives are the failure that matters ------------

const MUST_CATCH = [
  "I want to kill myself",
  "i've been thinking about suicide",
  "I feel suicidal lately",
  "SUICIDAL",
  "thoughts of self harm",
  "self-harm has been on my mind",
  "I am going to hurt myself",
  "I want to end my life",
  "sometimes I think about taking pills to overdose",
  "I don't want to be here anymore",
  "i dont want to be here",
  "I do not want to live",
  "don't want to exist anymore",
  "I just want to die.",
  "honestly... want to die",
  "I might take my life",
];

const MUST_PASS = [
  "How do I get funding for my club event?",
  "when is the next senate meeting",
  "this class is brutal and the deadline is killing my schedule",
  "who is the usg president",
  "how do I join the wellness committee",
];

test("keyword net catches every crisis phrasing variant", () => {
  for (const msg of MUST_CATCH) {
    assert.ok(server.hasCrisisKeywords(msg), `FALSE NEGATIVE: ${JSON.stringify(msg)}`);
  }
});

test("keyword net does not fire on normal USG questions", () => {
  for (const msg of MUST_PASS) {
    assert.ok(!server.hasCrisisKeywords(msg), `false positive: ${JSON.stringify(msg)}`);
  }
});

test("history net catches crisis language in earlier turns", () => {
  assert.ok(
    server.historySuggestsCrisis([
      { role: "user", content: "I have no reason to live" },
      { role: "assistant", content: "..." },
    ])
  );
  assert.ok(server.historySuggestsCrisis([{ role: "user", content: "life is pointless" }]));
  assert.ok(!server.historySuggestsCrisis([]));
  assert.ok(!server.historySuggestsCrisis([{ role: "user", content: "how do I get funding" }]));
  assert.ok(!server.historySuggestsCrisis([{ role: "user" }, {}])); // null-safe
});

// ---- canned replies: the last line of defense must itself be safe ---------

test("every canned crisis reply mentions 988 and passes the style guardrail", () => {
  for (const reply of server.CRISIS_REPLIES) {
    assert.ok(reply.includes("988"), `canned reply missing 988: ${reply}`);
    assert.ok(!server.hasDisallowedAssistantLanguage(reply), `canned reply fails own guardrail: ${reply}`);
  }
});

test("pickCrisisReply is deterministic", () => {
  assert.equal(server.pickCrisisReply("same input"), server.pickCrisisReply("same input"));
});

test("disallowed-language guardrail flags each banned pattern", () => {
  for (const bad of [
    "I can help you with that.",
    "I'm here if you want to talk",
    "Let me know how it goes.",
    "What would you like to do next?",
    "How can I help?",
    "I can support you through this.",
  ]) {
    assert.ok(server.hasDisallowedAssistantLanguage(bad), `guardrail missed: ${bad}`);
  }
  assert.ok(!server.hasDisallowedAssistantLanguage("Please call or text 988 right now."));
});

// ---- moderation wiring and its failure mode -------------------------------

test("moderation-flagged message with no keywords is a crisis", async () => {
  assert.equal(await server.isCrisisMessage(MOD_FLAG), true);
});

test("unflagged message with no keywords is not a crisis", async () => {
  assert.equal(await server.isCrisisMessage("how do I get funding"), false);
});

test("moderation outage: keyword crises still caught, keywordless ones documented as missed", async () => {
  // keywords short-circuit before moderation, so an outage cannot cause a
  // false negative on any MUST_CATCH phrasing
  assert.equal(await server.isCrisisMessage(`${MOD_DOWN} and I want to kill myself`), true);
  // residual risk, pinned so a future fix flips this assertion consciously:
  // keywordless crisis + moderation down => not caught
  assert.equal(await server.isCrisisMessage(MOD_DOWN), false);
});

// ---- end-to-end /api/chat behavior ---------------------------------------

test("stream endpoint sends crisis as one complete event, never token-streamed", async () => {
  const r = await fetch(`${api}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "I want to kill myself" }),
  });
  const events = (await r.text())
    .split("\n\n")
    .filter((p) => p.startsWith("data: "))
    .map((p) => JSON.parse(p.slice(6)));
  assert.equal(events.length, 2); // crisis payload + done, nothing else
  assert.equal(events[0].crisis, true);
  assert.ok(events[0].answer.length > 0);
  assert.ok(events[1].done);
});

test("crisis short-circuits to 988 payload, never RAG", async () => {
  const { body } = await chat({ message: "I want to kill myself" });
  assert.equal(body.crisis, true);
  assert.ok(body.sources.some((s) => s.source_url.includes("988")));
  assert.ok(!body.sources.some((s) => s.source_url.includes("usg.usc.edu"))); // no KB leakage
});

test("moderation-flagged message routes to crisis end-to-end", async () => {
  const { body } = await chat({ message: MOD_FLAG });
  assert.equal(body.crisis, true);
  assert.ok(body.sources.some((s) => s.source_url.includes("988")));
});

test("crisis in history routes a vague follow-up to crisis", async () => {
  const { body } = await chat({
    message: "what's the point anymore",
    history: [
      { role: "user", content: "I feel like there is no reason to live" },
      { role: "assistant", content: "I'm sorry you're hurting." },
    ],
  });
  assert.equal(body.crisis, true);
});

test("disallowed generated language is replaced by a canned 988 reply", async () => {
  const { body } = await chat({ message: `I want to ${BAD_STYLE}` });
  assert.equal(body.crisis, true);
  assert.ok(server.CRISIS_REPLIES.includes(body.answer), `answer was not canned: ${body.answer}`);
});

