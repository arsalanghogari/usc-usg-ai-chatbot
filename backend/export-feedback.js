#!/usr/bin/env node
// Archives real-user Langfuse traces (question, answer, route, session) and
// user_feedback scores into Postgres before Langfuse's 30-day free-tier
// window drops them. Runs weekly from ingest.yml; idempotent upserts, so
// overlapping windows are harmless. The repo is public, so this user-written
// content lives in the private DB (RLS on, no public API access), never in git.
require("dotenv").config();
const { Pool } = require("pg");

const BASE = process.env.LANGFUSE_BASEURL || "https://us.cloud.langfuse.com";
const auth =
  "Basic " +
  Buffer.from(
    `${process.env.LANGFUSE_PUBLIC_KEY}:${process.env.LANGFUSE_SECRET_KEY}`
  ).toString("base64");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Langfuse rate-limits hard; back off and retry instead of failing the run
async function lf(path) {
  for (let a = 0; ; a++) {
    const r = await fetch(BASE + path, { headers: { Authorization: auth } });
    if (r.ok) return r.json();
    if (a >= 5 || (r.status < 500 && r.status !== 429)) throw new Error(`${path}: HTTP ${r.status}`);
    await sleep(4000 * (a + 1));
  }
}

async function all(path) {
  const out = [];
  for (let page = 1; ; page++) {
    const d = await lf(`${path}&limit=100&page=${page}`);
    out.push(...(d.data || []));
    if (!d.data?.length || page >= (d.meta?.totalPages || 1)) return out;
    await sleep(800);
  }
}

// The widget always sends a sessionId; synthetic waves are tagged syn-eval-*,
// and sessionless traces are scripts (eval.js and friends).
const isReal = (sessionId) => Boolean(sessionId) && !String(sessionId).startsWith("syn-eval");
const text = (v) => (v == null ? null : typeof v === "string" ? v : JSON.stringify(v));

(async () => {
  if (!process.env.LANGFUSE_PUBLIC_KEY || !process.env.SUPABASE_DB_URL) {
    console.error("Need LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY and SUPABASE_DB_URL");
    process.exit(1);
  }
  const pool = new Pool({
    connectionString: process.env.SUPABASE_DB_URL,
    ssl: { rejectUnauthorized: false },
  });
  // RLS with no policies: the public REST API (anon key) sees nothing; this
  // script and the backend connect as postgres, which bypasses RLS.
  await pool.query(`
    create table if not exists trace_archive (
      id text primary key,
      ts timestamptz,
      name text,
      session_id text,
      route text,
      question text,
      answer text
    );
    create table if not exists feedback_archive (
      id text primary key,
      trace_id text,
      ts timestamptz,
      value int,
      comment text,
      question text,
      route text,
      session_id text
    );
    alter table trace_archive enable row level security;
    alter table feedback_archive enable row level security;`);

  // Everything Langfuse still holds; the upsert makes re-archiving a no-op.
  // Crisis traces are count-only markers with no content or session: they are
  // dropped by isReal like any sessionless trace.
  const traces = (await all("/api/public/traces?")).filter((t) => isReal(t.sessionId));
  for (const t of traces) {
    await pool.query(
      `insert into trace_archive (id, ts, name, session_id, route, question, answer)
       values ($1,$2,$3,$4,$5,$6,$7)
       on conflict (id) do update set answer = coalesce(excluded.answer, trace_archive.answer)`,
      [t.id, t.timestamp, t.name, t.sessionId, (t.tags || [])[0] || null, text(t.input), text(t.output)]
    );
  }

  const byTrace = new Map(traces.map((t) => [t.id, t]));
  const scores = await all("/api/public/scores?name=user_feedback");
  let kept = 0;
  for (const s of scores) {
    // trace lookup: from this run's list, else the archive (trace aged out)
    let t = byTrace.get(s.traceId);
    if (!t) {
      const { rows } = await pool.query(
        "select session_id as \"sessionId\", question as input, route from trace_archive where id = $1",
        [s.traceId]
      );
      t = rows[0] && { ...rows[0], tags: [rows[0].route] };
    }
    if (!t || !isReal(t.sessionId)) continue; // synthetic rating or unknown trace
    await pool.query(
      `insert into feedback_archive (id, trace_id, ts, value, comment, question, route, session_id)
       values ($1,$2,$3,$4,$5,$6,$7,$8)
       on conflict (id) do update set value = excluded.value,
         comment = coalesce(excluded.comment, feedback_archive.comment)`,
      [s.id, s.traceId, s.timestamp, s.value, s.comment || null, text(t.input), (t.tags || [])[0] || null, t.sessionId]
    );
    kept++;
  }

  const { rows } = await pool.query(`
    select (select count(*)::int from trace_archive) as traces,
           (select count(distinct session_id)::int from trace_archive) as sessions,
           (select count(*)::int from feedback_archive) as ratings`);
  console.log(
    `langfuse: ${traces.length} real traces, ${kept}/${scores.length} real ratings; ` +
      `archive now ${rows[0].traces} traces, ${rows[0].sessions} sessions, ${rows[0].ratings} ratings`
  );
  await pool.end();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
