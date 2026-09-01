#!/usr/bin/env node
// Answer accuracy on REAL traffic: pulls chat traces from Langfuse for a date
// range, rebuilds each answer's retrieved sources from its rerank/retrieve
// span, and grades it with the same rubric live-eval.js uses on synthetic
// waves. Langfuse's free tier keeps ~30 days, so run it weekly.
// Usage: node evals/prod-eval.js --from 2026-08-16 --to 2026-08-20 [--judge-model M]
// Sidecar log: evals/prod-eval-log.jsonl

const fs = require("fs");
const path = require("path");
require("dotenv").config({ path: path.join(__dirname, "..", "backend", ".env") });

const { RUBRIC, openai } = require("./live-eval.js");

const arg = (n, d) => {
  const i = process.argv.indexOf("--" + n);
  return i > -1 ? process.argv[i + 1] : d;
};
const FROM = arg("from", null);
const TO = arg("to", FROM);
const CONC = Number(arg("concurrency", 4));
const LOG = path.join(__dirname, "prod-eval-log.jsonl");
if (!FROM) {
  console.error("need --from YYYY-MM-DD [--to YYYY-MM-DD] (dates are Pacific)");
  process.exit(1);
}
// USG's students are in Pacific; Langfuse timestamps are UTC
const PT_OFFSET = "T07:00:00Z";
const fromTs = FROM + PT_OFFSET;
const toTs = new Date(Date.parse(TO + PT_OFFSET) + 864e5).toISOString();

const BASE = process.env.LANGFUSE_BASEURL || "https://us.cloud.langfuse.com";
const auth =
  "Basic " +
  Buffer.from(`${process.env.LANGFUSE_PUBLIC_KEY}:${process.env.LANGFUSE_SECRET_KEY}`).toString("base64");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Langfuse rate-limits hard; every 4xx/5xx here is worth one patient retry
async function lf(p) {
  for (let a = 0; ; a++) {
    const r = await fetch(BASE + p, { headers: { Authorization: auth } });
    if (!r.ok) {
      if (a < 6) {
        await sleep(4000 * (a + 1));
        continue;
      }
      throw new Error(`${p}: HTTP ${r.status}`);
    }
    return r.json();
  }
}

async function pages(pathname, onPage) {
  for (let page = 1; ; page++) {
    const d = await lf(`${pathname}&limit=50&page=${page}`);
    onPage(d.data || []);
    if (!d.data?.length || page >= (d.meta?.totalPages || 1)) return;
    await sleep(1000);
  }
}

// The bot answers from kb.json, so that is what "grounded" is measured against
// (the whole page, not just the chunks that scored — the model saw siblings).
const kbByUrl = {};
for (const ch of JSON.parse(fs.readFileSync(path.join(__dirname, "..", "backend", "kb.json"), "utf-8")).chunks)
  kbByUrl[ch.source_url] = (kbByUrl[ch.source_url] || "") + ch.text + "\n";

(async () => {
  if (!process.env.LANGFUSE_PUBLIC_KEY) {
    console.error("need LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY");
    process.exit(1);
  }
  const traces = [];
  await pages(`/api/public/traces?fromTimestamp=${fromTs}&toTimestamp=${toTs}`, (d) =>
    traces.push(...d.filter((t) => !String(t.sessionId).startsWith("syn-eval"))) // drop synthetic waves
  );
  traces.sort((a, b) => a.timestamp.localeCompare(b.timestamp));

  // one bulk sweep instead of a per-trace fetch: 136 trace GETs gets you 429'd
  const sources = {};
  for (const name of ["retrieve", "rerank"]) // rerank second: it wins where present
    await pages(
      `/api/public/observations?name=${name}&fromStartTime=${fromTs}&toStartTime=${toTs}`,
      (d) => {
        for (const o of d)
          if (Array.isArray(o.output)) sources[o.traceId] = o.output.map((x) => x.url).filter(Boolean);
      }
    );

  console.log(`${FROM}..${TO} (PT): ${traces.length} questions, ${new Set(traces.map((t) => t.sessionId)).size} sessions`);

  const results = [];
  const queue = [...traces];
  const worker = async () => {
    while (queue.length) {
      const t = queue.shift();
      const route = (t.tags || [])[0] || null;
      const urls = [...new Set(sources[t.id] || [])].slice(0, 3);
      let row;
      try {
        const context = urls.length
          ? urls.map((u, i) => `[${i + 1}] ${u}\n${kbByUrl[u] ? kbByUrl[u].slice(0, 14000) : "(page text unavailable)"}`).join("\n\n")
          : "(no sources cited)";
        const out = await openai(
          RUBRIC,
          // the answer is history by the time it is graded: without the ask
          // date the judge fails correct calendar answers as "past events"
          `This exchange happened on ${t.timestamp.slice(0, 10)} — judge it as of that date, not today.\n\n` +
            `Live page text:\n${context}\n\nStudent question: ${t.input}\n\nChatbot answer:\n${t.output || "(the bot returned nothing)"}`
        );
        const v = JSON.parse(out.match(/\{[\s\S]*\}/)[0]);
        row = { ts: t.timestamp, route, q: t.input, value: [0, 1].includes(v.value) ? v.value : null, reason: v.reason, urls };
      } catch (e) {
        row = { ts: t.timestamp, route, q: t.input, value: null, reason: "judge error: " + e.message };
      }
      results.push(row);
      fs.appendFileSync(LOG, JSON.stringify({ traceId: t.id, ...row }) + "\n");
      process.stdout.write(row.value === 1 ? "." : row.value === 0 ? "X" : "?");
    }
  };
  await Promise.all(Array.from({ length: CONC }, worker));

  const rated = results.filter((r) => r.value !== null);
  const up = rated.filter((r) => r.value === 1).length;
  console.log(`\n\nrated=${rated.length}/${results.length} accurate=${up} (${((100 * up) / Math.max(1, rated.length)).toFixed(1)}%)`);

  const byRoute = {};
  for (const r of rated) {
    const b = (byRoute[r.route] ||= [0, 0]);
    b[0] += r.value;
    b[1]++;
  }
  for (const [k, [u, n]] of Object.entries(byRoute).sort((a, b) => b[1][1] - a[1][1]))
    console.log(`  ${k}: ${u}/${n} = ${Math.round((100 * u) / n)}%`);
  for (const r of results.filter((x) => x.value === 0)) console.log(`WRONG [${r.route}] ${r.q}\n      ${r.reason}`);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
