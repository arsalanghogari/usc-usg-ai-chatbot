#!/usr/bin/env node
// STANDBY — not currently scheduled (simulations would pollute the archive;
// re-add the feedback-export job to ingest.yml to enable).
// Archives Langfuse user_feedback scores (+ their trace's question, route,
// session) into Postgres before Langfuse's 30-day free-tier window drops
// them. Idempotent upserts. The repo is public, so this user-written
// content lives in the private DB, never in git.
require("dotenv").config();
const { Pool } = require("pg");

const BASE = process.env.LANGFUSE_BASEURL || "https://us.cloud.langfuse.com";
const auth =
  "Basic " +
  Buffer.from(
    `${process.env.LANGFUSE_PUBLIC_KEY}:${process.env.LANGFUSE_SECRET_KEY}`
  ).toString("base64");

async function lf(path) {
  const r = await fetch(BASE + path, { headers: { Authorization: auth } });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

(async () => {
  if (!process.env.LANGFUSE_PUBLIC_KEY || !process.env.SUPABASE_DB_URL) {
    console.error("Need LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY and SUPABASE_DB_URL");
    process.exit(1);
  }
  const pool = new Pool({
    connectionString: process.env.SUPABASE_DB_URL,
    ssl: { rejectUnauthorized: false },
  });
  await pool.query(`
    create table if not exists feedback_archive (
      id text primary key,
      trace_id text,
      ts timestamptz,
      value int,
      comment text,
      question text,
      route text,
      session_id text
    )`);

  const scores = [];
  for (let page = 1; page <= 20; page++) {
    const d = await lf(`/api/public/scores?limit=100&page=${page}&name=user_feedback`);
    scores.push(...(d.data || []));
    if (!d.data?.length || page >= (d.meta?.totalPages || 1)) break;
  }

  let upserted = 0;
  for (const s of scores) {
    let question = null, route = null, session = null;
    try {
      const t = await lf(`/api/public/traces/${s.traceId}`);
      question = typeof t.input === "string" ? t.input : null;
      route = (t.tags || [])[0] || null;
      session = t.sessionId || null;
    } catch {} // trace may have aged out; keep the score anyway
    const r = await pool.query(
      `insert into feedback_archive (id, trace_id, ts, value, comment, question, route, session_id)
       values ($1,$2,$3,$4,$5,$6,$7,$8)
       on conflict (id) do update set value = excluded.value,
         comment = coalesce(excluded.comment, feedback_archive.comment)`,
      [s.id, s.traceId, s.timestamp, s.value, s.comment || null, question, route, session]
    );
    upserted += r.rowCount;
  }
  const { rows } = await pool.query(
    "select count(*)::int as total, count(*) filter (where value = 0)::int as downs from feedback_archive"
  );
  console.log(
    `fetched ${scores.length} scores; upserted ${upserted}; archive now ${rows[0].total} rows (${rows[0].downs} downvotes)`
  );
  await pool.end();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
