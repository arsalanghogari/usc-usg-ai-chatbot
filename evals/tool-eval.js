#!/usr/bin/env node
// Tool-selection accuracy over evals/tools.json.
// Usage: node evals/tool-eval.js
// Appends a summary line to evals/history.jsonl (kind: "tool-selection").

const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

require("dotenv").config({ path: path.join(__dirname, "..", "backend", ".env") });
const server = require("../backend/server.js");

const cases = JSON.parse(fs.readFileSync(path.join(__dirname, "tools.json"), "utf-8"));

async function main() {
  const wrong = [];
  for (const c of cases) {
    const route = await server.routeTool(c.q, []);
    const ok = route.tool === c.tool;
    process.stdout.write(ok ? "." : "x");
    if (!ok) wrong.push({ q: c.q, expected: c.tool, got: route.tool });
  }
  console.log("\n");

  const summary = {
    ts: new Date().toISOString(),
    git: execSync("git rev-parse --short HEAD", { cwd: __dirname }).toString().trim(),
    kind: "tool-selection",
    n: cases.length,
    tool_accuracy: +((cases.length - wrong.length) / cases.length).toFixed(3),
    wrong_tool_rate: +(wrong.length / cases.length).toFixed(3),
  };
  console.table([summary]);
  for (const w of wrong) console.log(`WRONG  ${w.q}\n       expected ${w.expected}, got ${w.got}`);

  fs.appendFileSync(path.join(__dirname, "history.jsonl"), JSON.stringify(summary) + "\n");
  await server.pool?.end();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
