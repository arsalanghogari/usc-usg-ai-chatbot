#!/usr/bin/env node
// RAG eval harness: retrieval hit-rate + MRR over evals/golden.json,
// optional LLM-judge faithfulness (--judge) through the real /api/chat.
// Usage: node evals/eval.js [--judge] [--k 4]
// Appends one summary line per run to evals/history.jsonl.

const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

require("dotenv").config({ path: path.join(__dirname, "..", "backend", ".env") });
const server = require("../backend/server.js");

const JUDGE = process.argv.includes("--judge");
const K = Number(process.argv[process.argv.indexOf("--k") + 1]) || 4;
const golden = JSON.parse(fs.readFileSync(path.join(__dirname, "golden.json"), "utf-8"));

const { client, CHAT_MODEL } = server;

async function retrieve(q, queryEmbedding, text = null) {
  const kCand = server.RERANK ? 20 : K;
  const cands = server.pool
    ? await server.topChunksDb(queryEmbedding, kCand, text)
    : server.topChunks(queryEmbedding, server.loadKb().chunks, kCand, text);
  return server.RERANK ? server.rerank(q, cands, K) : cands;
}

async function judgeFaithfulness(question, answer, context) {
  const resp = await client.responses.create({
    model: CHAT_MODEL,
    instructions:
      "You are grading a RAG chatbot. Given the retrieved context, the user question, and the bot's answer, decide whether every factual claim in the answer is supported by the context. Ignore hedges, formatting, and offers to help further. Reply with ONLY a JSON object: {\"faithful\": true|false, \"reason\": \"<short>\"}",
    input: [
      {
        role: "user",
        content: `Context:\n${context}\n\nQuestion: ${question}\n\nAnswer: ${answer}`,
      },
    ],
  });
  try {
    return JSON.parse(resp.output_text.match(/\{[\s\S]*\}/)[0]);
  } catch {
    return { faithful: null, reason: "judge output unparseable" };
  }
}

async function main() {
  const results = [];

  for (const item of golden) {
    const expected = Array.isArray(item.expected) ? item.expected : [item.expected];
    // same query the server retrieves with (assembly bias included)
    const rq = server.retrievalQuery(item.q);
    const emb = await server.embed(rq);
    const matches = await retrieve(rq, emb, item.q);
    const urls = [...new Set(matches.map((m) => m.source_url))];
    const rank = urls.findIndex((u) => expected.includes(u)) + 1; // 0 = miss

    const r = { q: item.q, hit: rank > 0, rank, top: urls[0] };

    if (JUDGE) {
      // Generate from the server's own prep and judge against that exact
      // context. Rebuilding it here missed sibling chunks and the live roster,
      // so correct answers got flagged as unsupported (0.50 on 2026-10-20).
      const prep = await server.ragPrepare(item.q);
      const resp = await client.responses.create({
        model: CHAT_MODEL,
        instructions: prep.instructions,
        input: [{ role: "user", content: prep.userContent }],
        reasoning: { effort: server.REASONING_EFFORT },
      });
      const context = prep.userContent.split("</context>")[0].replace("<context>\n", "");
      const verdict = await judgeFaithfulness(item.q, resp.output_text || "", context);
      r.faithful = verdict.faithful;
      r.judge_reason = verdict.reason;
    }

    results.push(r);
    process.stdout.write(r.hit ? "." : "x");
  }
  console.log("\n");

  const n = results.length;
  const hits = results.filter((r) => r.hit).length;
  const mrr = results.reduce((s, r) => s + (r.rank ? 1 / r.rank : 0), 0) / n;
  const judged = results.filter((r) => typeof r.faithful === "boolean");
  const faithful = judged.filter((r) => r.faithful).length;

  const summary = {
    ts: new Date().toISOString(),
    git: execSync("git rev-parse --short HEAD", { cwd: __dirname }).toString().trim(),
    backend: server.pool ? "pgvector" : "kb.json",
    hybrid: server.HYBRID,
    rerank: server.RERANK,
    k: K,
    n,
    hit_rate: +(hits / n).toFixed(3),
    mrr: +mrr.toFixed(3),
    faithfulness: judged.length ? +(faithful / judged.length).toFixed(3) : null,
  };

  console.table([summary]);
  for (const r of results.filter((x) => !x.hit)) {
    console.log(`MISS  ${r.q}\n      got: ${r.top}`);
  }
  for (const r of results.filter((x) => x.faithful === false)) {
    console.log(`UNFAITHFUL  ${r.q}\n      judge: ${r.judge_reason}`);
  }

  fs.appendFileSync(path.join(__dirname, "history.jsonl"), JSON.stringify(summary) + "\n");

  await server.pool?.end();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
