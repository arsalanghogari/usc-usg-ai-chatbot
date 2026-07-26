# Ask USG — Live Synthetic Eval Report

**2026-07-25/26 · 792 rated answers · 6 waves · live production stack**

## Fix trajectory (added 07-26)

| Config | Wave | Positive rate |
|---|---|---|
| Baseline | w1–w4 (526 answers) | **83.7%** |
| + prompt guard, senate schedule line | w5 (133) | 83.5% raw / 85.7% corrected* |
| + Google Calendar ICS events source | — | (verified in prod; no full wave) |
| + sibling-chunk retrieval expansion | w6 (133) | **94.7%** |

\* 3 of w5's downs were the judge flagging the deliberately-injected (and
corpus-verified) senate schedule line; the rubric now exempts it.

The big lever was **sibling-chunk expansion** (rerank picks 4 chunks; roster/
list pages span 8–12, so completeness was structurally impossible). It fixed
the incompleteness cluster and, as a side effect, most "unsupported specifics"
flags — with whole pages in context, the model stops filling gaps.

Remaining 7 downs in w6 are a mix of judge-visibility artifacts and two real
nits: ambiguous-followup handling ("what have *they* accomplished") and
new-vs-current senator disambiguation (blocked on the exec/team page revamp
planned for the school year).

---

*Original baseline report below (waves 1–4).*

## Headline

**83.7% positive across 526 rated answers** (440 👍 / 86 👎, 100% rating coverage).

Stable across waves — w1 85.0%, w2 84.2%, w3 82.0%, w4 83.5% — so this is a real
measurement, not noise. Against the calibration bar: comfortably above the 75%
"healthy" line, 5.3 points short of the 89% target. The gap is ~28 answers, and
the failure clusters below account for nearly all of them.

## How it was run (methodology, so the number is defensible)

- 127-question bank across 17 intents, mirrored from the actual KB coverage
  (funding, elections 2026, senate, exec, committees, resources, join, contact,
  out-of-scope, typos, ambiguous one-worders, stale-info probes, multi-turn).
- 8 persona phrasing styles (club officer, transfer student, casual lowercase,
  typos…) rotated per wave; follow-up turns reuse conversation history exactly
  like the widget does.
- Every session hit the real production `/api/chat` on Render; every rating went
  through the real `/api/feedback` endpoint.
- Ratings are **on merit**: an LLM judge (gpt-5.5, low effort) graded each answer
  against the live usg.usc.edu pages it cited (ingested-snapshot fallback while
  the site was 502ing) on grounding, responsiveness, and usability. Graceful
  declines on out-of-scope questions count as good answers.
- **Data hygiene**: every synthetic trace carries a `syn-eval-*` sessionId in
  Langfuse. Filter `sessionId NOT LIKE 'syn-eval-%'` and your real-user metrics
  are untouched. Synthetic scores can be bulk-deleted later the same way.
- The beta Google Form ("should this ship?") was deliberately left alone — that
  verdict belongs to real humans.

## Per-intent results

| Intent | 👍/n | % | | Intent | 👍/n | % |
|---|---|---|---|---|---|---|
| programming | 8/16 | 50% | | advocacy | 39/44 | 89% |
| committees | 19/28 | 68% | | oos | 29/32 | 91% |
| senate | 34/48 | 71% | | judicial | 10/11 | 91% |
| resources | 24/32 | 75% | | funding | 107/116 | 92% |
| join | 18/24 | 75% | | contact | 12/12 | 100% |
| typo | 12/16 | 75% | | about | 16/16 | 100% |
| stale | 9/12 | 75% | | ambiguous | 20/20 | 100% |
| elections | 46/56 | 82% | | multihop | 14/16 | 88% |
| exec | 23/27 | 85% | | | | |

Funding — the highest-volume real intent — is already at 92%. The damage is
concentrated in senate/committees/programming, i.e. "what's happening right now"
questions.

## Punch list (ranked by expected point gain)

### 1. The calendar hole — fabricated meeting times (~10 downs, ≈2 pts)
"When is the next senate meeting?" is one of the widget's three suggested
questions, and the bot **invents calendar contents** to answer it —
`/calendar` is on the ingest exclude list, so it has literally no data.
Fix: either ingest the calendar feed (or senate-meeting schedule line on the
Senate page), or add a deterministic response for meeting-time questions that
links the calendar instead of guessing. Cheap and high-visibility.

### 2. Hallucinated specifics under thin retrieval (~50 downs total, ≈4–5 pts)
When retrieval comes back weak, the model fills the gap with confident
specifics: invented emails (`usgsots@usc.edu`), GSG office hours and phone
numbers, meeting times/locations, project counts, event lists. One system-prompt
hardening pass fixes most of this: *"only state emails, dates, times, locations,
dollar amounts, and names that appear verbatim in the provided context;
otherwise say you don't have it and link the closest page."* The oos/GSG case
also deserves a canned redirect (bot scored 91% on oos otherwise).

### 3. Retrieval gaps — "not stated" when it is stated (~28 downs, ≈2–3 pts)
Recurring misses: the $8,000/yr funding cap for the phrasing "how much can we
get per year", online-voting info on the elections page, SB 145-19 in the
Feb 24 press release, full team rosters (programming worst at 50%). Two levers:
enable/verify the rerank path (`RERANK` already exists in server.js) and pull
sibling chunks of the same source page when one chunk of a roster/list page is
retrieved, so lists come back whole.

### 4. KB pruning — internal contradictions (low count, high risk)
The 2023 "Funding Resources" blog contradicts the June 2026 Funding page on new
vendors (bot currently resolves it correctly, but it's one retrieval roll away
from answering with 2023 policy). Add the stale funding blog to the exclude
list or down-weight pre-2025 blog content for policy questions.

### 5. Ops footnote
6 sessions hit the 20 req/min per-IP rate limit under concurrency 4 (runner now
backs off; real per-user traffic won't trip this). The USG site itself 502'd
during testing — worth an ingest-time alert so a WP outage doesn't silently
freeze the corpus.

## Bottom line

Items 1 + 2 are a prompt edit and a small ingest/deterministic-answer change,
and together they plausibly clear the 89% bar; item 3 pushes past it with
margin. Re-running this eval after each fix is one command:

    node evals/live-eval.js --wave w5 --concurrency 3 --variant 1

Artifacts: `evals/live-eval.js` (runner) · `evals/live-questions.json` (bank) ·
`evals/live-eval-log.jsonl` (all 526 rows: question, answer, sources, verdict,
comment, traceId).
