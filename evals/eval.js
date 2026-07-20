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

async function retrieve(q, queryEmbedding) {
  const kCand = server.RERANK ? 20 : K;
  const cands = server.pool
    ? await server.topChunksDb(queryEmbedding, kCand)
    : server.topChunks(queryEmbedding, server.loadKb().chunks, kCand);
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
  let httpBase = null;
  let httpServer = null;

  if (JUDGE) {
    httpServer = server.app.listen(0);
    httpBase = `http://localhost:${httpServer.address().port}`;
  }

  for (const item of golden) {
    const expected = Array.isArray(item.expected) ? item.expected : [item.expected];
    const emb = await server.embed(item.q);
    const matches = await retrieve(item.q, emb);
    const urls = [...new Set(matches.map((m) => m.source_url))];
    const rank = urls.findIndex((u) => expected.includes(u)) + 1; // 0 = miss

    const r = { q: item.q, hit: rank > 0, rank, top: urls[0] };

    if (JUDGE) {
      const resp = await fetch(`${httpBase}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: item.q }),
      });
      const js = await resp.json();
      // judge the model's own words, not the deterministic staleness notice
      const bare = (js.answer || "").split("\n\n---\n⚠️")[0];
      // mirror the server's prompt context exactly (incl. date headers),
      // otherwise the judge flags legitimate "page last updated" citations
      const context = matches
        .map(
          (m, i) =>
            `[${i + 1}] ${m.source_title}` +
            (m.source_modified ? ` (page last updated: ${m.source_modified.slice(0, 10)})` : "") +
            `\n${m.source_url}\n${m.text}`
        )
        .join("\n\n");
      const verdict = await judgeFaithfulness(item.q, bare, context);
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

  if (httpServer) httpServer.close();
  await server.pool?.end();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
