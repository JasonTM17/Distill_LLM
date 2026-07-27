---
title: "Repo polish + v0.6 honest closeout"
description: "Audit and complete repo docs, GitHub metadata, community files; honestly close out the v0.6 expand-and-retrain experiment."
status: in-progress
priority: P2
effort: 2h
branch: master
tags: [docs, repo-polish, distillation, closeout]
created: 2026-07-27
---

## Status

In progress. Audit done; executing documentation closeout + repo polish.

## Context

User asked to make the repo thorough ("kĩ") and complete: repo docs, GitHub
About, everything — and felt the distill was "not solid." Audit found v0.6 ran
but was never honestly closed out: eval report untracked, docs still on v0.5,
roadmap's v0.6 section describes a different (unrun) plan, CHANGELOG has no
v0.6, and v0.6 regressed on the headline metric (5.85 vs 5.23) without that
being documented. v0.5 is the better overall model and stays canonical/served.

## Phases

1. **Honest v0.6 closeout** — commit `evaluation-v0.6.md`; update the v0.6 plan
   status + outcome; cut CHANGELOG 0.5.0 and add 0.6.0; sync roadmap, README,
   PDR to reality (v0.5 canonical, v0.6 = documented experiment).
2. **Repo polish** — ignore + untrack `tsconfig.tsbuildinfo`; add CONTRIBUTING,
   SECURITY, CODE_OF_CONDUCT, issue/PR templates; verify + refine GitHub About.
3. **Verify + ship** — ruff, pytest (core + api), commit, push.
4. **Distill decision** — present root cause of the v0.6 regression and the
   v0.7 recovery plan; confirm GPU/9Router readiness before any retrain.

## Acceptance criteria

- v0.6 eval report committed; v0.6 plan reflects actual phase state + honest result
- CHANGELOG has 0.5.0 and 0.6.0; roadmap/README/PDR match reality
- v0.5 clearly marked canonical/served; v0.6 marked experiment, not shipped
- Community health files present; tsbuildinfo untracked
- ruff clean; core + api tests pass; working tree clean after commit/push
- Root cause + v0.7 plan presented; retrain confirmed with user before running

## Risks

- Over-stating v0.6 as a success (mitigated: honest "not shipped, regressed" framing)
- Editing docs in a way that contradicts the eval numbers (mitigated: all figures
  copied verbatim from the two evaluation reports)
