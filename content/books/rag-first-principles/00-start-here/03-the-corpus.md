# The Corpus and the Golden Set

## `handbook/`: the corpus

Fourteen markdown documents making up the internal handbook of **Lumora Robotics**, a fictional
company that builds warehouse robots (Atlas), fleet software (Beacon) and analytics (Compass).
About 5,200 words. Written so that:

- Facts are **concrete and checkable** (numbers, names, dates, commands), so retrieval and
  answer quality can be judged mechanically.
- Some facts appear in **two documents** (e.g. the API rate limit in the Beacon doc and the FAQ),
  so multi-document retrieval and deduplication matter.
- One fact **conflicts on purpose**: the FAQ says PTO is 20 days (marked "updated February 2024"),
  the 2026 policy says 24. Chapters 9, 11 and 12 use this.
- Some questions need **two facts combined** (pricing arithmetic, probation + blackout rules).
- Nothing about the company is real. Domains use `.example`.

| File | Topic | Used heavily in |
|---|---|---|
| 01-company-overview | founders, offices, products, scale | ch 3 |
| 02-pto-and-leave-policy | PTO 24 days, carryover, sick, parental, blackout, probation | ch 3, 9, 11 (conflict) |
| 03-travel-and-expense-policy | flights, hotel caps, per diem, Ledger, 30 days | ch 5 |
| 04-atlas-a2-specification | payload, speed, battery, dimensions, IP54, ISO 3691-4, firmware | ch 8 (exact tokens) |
| 05-beacon-fleet-software | architecture, API limits, SLA, RTO/RPO, releases | ch 8, 9 |
| 06-security-policy | MFA, rotation, classification, incidents, AI tools | ch 16 (RBAC) |
| 07-onboarding-guide | Atlas Academy, probation, learning budget | |
| 08-runbook-fleet-outage | severity, on-call, rollback command, certificates | ch 8 |
| 09-support-sla | tiers table, credits | ch 5 (tables) |
| 10-pricing-and-plans | prices, discounts, worked example | ch 9, 15 (multi-hop) |
| 11-release-notes-beacon-4.2 | features, breaking changes, firmware ≥ 3.8 | |
| 12-postmortem-2026-03-rotterdam-halt | timeline, root cause, action items | |
| 13-remote-work-policy | hybrid days, stipend, work-from-anywhere | |
| 14-faq | short answers, one outdated | ch 9, 11 |

## `golden/qa.yaml`: the evaluation set

47 questions. Each has `id`, `question`, `answer` (reference), `sources` (doc ids that contain
the answer), `answerable`, `keywords` (strings a correct answer should contain), `tags`.

- `q01`–`q42`: answerable. Tags: `numeric`, `table`, `factual`, `multi_doc`, `multi_hop`,
  `conflict`, `reasoning`, `code`.
- `u01`–`u05`: **unanswerable** from the corpus. The correct behaviour is to say so. `u03` has a
  false presupposition (there is no "Atlas A3"); `u05` is out of scope.

Retrieval metrics (Chapter 10) treat a retrieved chunk as relevant when its `doc_id` is in
`sources`. That is document-level relevance: coarse but cheap and unambiguous. Chapter 10
discusses chunk-level labels.

Add your own questions here; keep the ids stable so eval results stay comparable over time.

## `gutenberg/`, `generated/` (git-ignored)

Created by Chapter 7: public-domain books used as distractors, and LLM-synthesized handbook
documents. Delete freely; the scripts recreate them.
