<!-- Thanks for the PR! Keep the title in Conventional Commits form (feat:, fix:, docs:, refactor:, test:, chore:). -->

## Summary

<!-- What this changes and why. One or two sentences. -->

## Type

- [ ] feat
- [ ] fix
- [ ] docs
- [ ] refactor
- [ ] test
- [ ] chore

## Checklist

- [ ] `ruff check src/ tests/ services/api/` is clean
- [ ] `python -m pytest tests/ -q` passes
- [ ] `cd services/api && python -m pytest tests/ -q` passes (if API touched)
- [ ] `cd services/web && pnpm test && pnpm build` passes (if web touched)
- [ ] If `docs/openapi.yaml` changed: ran `pnpm run generate-client` and committed
      the regenerated `services/web/src/api/schema.d.ts`
- [ ] No secrets / API keys / private data committed
- [ ] No model weights / GGUF / processed datasets committed (they are gitignored)
- [ ] Docs updated where user-facing behavior, commands, or metrics changed
- [ ] Any new perplexity figure reports its truncation cap

## Evaluation (if this retrains or changes metrics)

- Dataset version:
- Split sizes (train/val/test):
- Best validation loss:
- Held-out PPL @ 512 / 1024 / 2048:
- LLM-as-judge: run / not run (why)
