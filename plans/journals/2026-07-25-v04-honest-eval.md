# v0.4 — Honest Evaluation on Held-Out Set

**Date:** 2026-07-25 22:00
**Severity:** Medium (deliverable gap, not blocking)
**Component:** Distillation pipeline, evaluation, docs
**Status:** Resolved (with documented caveats)

## What Happened

The project completed v0.3 with reported metrics that looked great:
- Loss 1.43 → 1.14
- Token accuracy 72.6%
- PPL 4.70 (Excellent)

But on closer inspection, **PPL 4.70 was measured on training data**, not held-out. The "12/12 = 100%" on `evaluate_extended.py` was a heuristic `len(response) > 20` — not a real quality check. We had no honesty in the metrics.

## The Brutal Truth

The v0.3 metrics were misleading by accident, not by design. The model was probably overfitting to the training data (loss 1.14 is suspiciously low for 1.5B after 39 steps). The "100% pass rate" was a lie — a model that outputs 50 characters of mostly-nonsensical text would also pass that heuristic.

This is the kind of evaluation theater that breaks trust with users. When you see a 100% pass rate, you should be suspicious, not proud.

## Technical Details

After Phase 1-6 of the v0.4 plan:

1. **Stratified train/test split** (357/38, seed=42) — first time we had real held-out data
2. **Retrained** for 135 full steps (vs v0.3's 39 steps) — proper 3 epochs on 357 samples
3. **PPL 6.93 on held-out** — honest number, Excellent grade (<10)
4. **ROUGE-L 0.1337** vs teacher — real quality comparison
5. **Per-category PPL**:
   - math 3.61 (best)
   - coding 4.66
   - science 5.67
   - reasoning 5.52
   - business 6.86
   - ml_ai 8.29
   - vietnamese 9.68
   - creative 14.39 (weakest)

## Windows Pagefile Saga

Lost ~2 hours to a Windows-specific error: `OSError: 1455 — paging file is too small`. The root cause:

- C: drive had only 1.9 GB free
- pagefile.sys couldn't grow to accommodate 3 GB model mmap
- transformers' `safe_open` uses `mmap` backend which requires OS page cache

The fix in `train_student.py`:
1. Monkey-patch `safetensors.safe_open` to pre-touch the file (read in 64 MB chunks)
2. This forces OS file system cache to load the model first
3. Then `safe_open` mmap succeeds because pages are already in cache

This is now documented in `docs/project-roadmap.md` known limitations.

## API Rate Limit Saga

Lost ~3 hours to `cx/gpt-5.5-xhigh` 429 rate limits:

- 134/530 prompts failed (philosophy + health = 0 success)
- Sliding window rate limit (~30 min) made retry strategies useless
- Each probe reset the timer
- Background `gen_robust.py` ran for ~2 hours with no new successes

Decision: train on 395 samples (8/10 categories) rather than block indefinitely. v0.5 (full 530) deferred.

## What We Tried

- Waiting 30+ min after each 429 — sometimes worked, mostly didn't
- Background processes that retry patiently — wasted electricity
- Switching to fallback teacher (`cmc/Qwen/Qwen3.6-Max-Preview`) — user wanted consistency with cx/gpt-5.5-xhigh, reverted
- Reverting to v0.3 prompts to retry — same rate limit issue

## Root Cause Analysis

**Why did v0.3 metrics look so good?** Incomplete training (39 steps) + in-sample evaluation. The model barely learned because it didn't see enough data per parameter update. The "low loss" was because the optimizer converged on the small subset it saw.

**Why did Windows fail?** C: drive filled up with apps/data over time. Pagefile on C: hit max, can't grow, can't mmap. The user has spent taste for "avoiding C: drive installs" — needs to keep cleaning C: drive.

**Why did API fail?** 9Router appears to have a daily quota on the `cx/gpt-5.x` model family. The "reset after 30 min" message is misleading.

## Lessons Learned

1. **Always evaluate on held-out data.** A pipeline without a test split is dangerous.
2. **Heuristic evaluation is not evaluation.** `len(response) > 20` is a smoke test, not a quality check.
3. **Document rate limits and resource constraints upfront.** Don't waste time discovering them mid-run.
4. **Windows pagefile is real.** Even with 32 GB RAM, a 3 GB mmap can fail if C: drive is full.
5. **Background processes don't help when the API is fundamentally rate-limited.** Honest time budget matters.

## Next Steps

1. **Today (urgent):** Tell user the v0.4 results and the data gaps
2. **Daily quota reset:** Re-run `gen_batch.py` to fill in philosophy + health
3. **v0.5:** Retrain on full 530, target PPL < 6.0 held-out
4. **v0.6:** Add code execution check for coding category (most reliable signal)
5. **Cleanup:** Delete v0.3 archive after v0.5 confirmed better

## Owner

User decides when to retry full 530. The current 395-sample model is functional and well-evaluated.