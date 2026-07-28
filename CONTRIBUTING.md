# Contributing to distill-gpt55

Thanks for considering a contribution. This repo has two distinct parts — an
**offline training pipeline** (`src/distill`) and an **online serving stack**
(`services/api`, `services/web`). Most contributions touch one of them.

## Setup

The package supports Python 3.12+ and CI runs on Python 3.12. The documented
Windows GPU training profile was developed with Python 3.14 and a compatible
PyTorch nightly on an RTX 3060 6GB; choose the PyTorch build that matches your
OS, Python, and CUDA versions. Serving only needs Docker.

```bash
pip install -e .[train,dev] trl
set PYTHONPATH=src
```

`trl` is currently required directly by `src/distill/train.py` but is not yet
declared in the `train` extra.

Web client (generated from the committed OpenAPI contract):

```bash
cd services/web
pnpm install
pnpm run generate-client   # regenerates src/api/schema.d.ts from docs/openapi.yaml
```

## Before you open a PR

1. **Lint + tests must pass:**

   ```bash
   ruff check src/ tests/ services/api/
   python -m pytest tests/ -q
   cd services/api && python -m pytest tests/ -q
   cd services/web && pnpm test && pnpm build
   ```

2. **If you changed the API contract** (`docs/openapi.yaml`), regenerate the
   client and commit the diff so CI's `git diff --exit-code src/api/schema.d.ts`
   stays green:

   ```bash
   cd services/web
   pnpm run generate-client
   pnpm build
   ```

3. **Never commit secrets.** `.env` is gitignored — copy `.env.example` to `.env`
   locally and fill in your 9Router key there. Do not paste real API keys, tokens,
   or teacher outputs containing private data.

4. **Model artifacts are not in git.** `checkpoints/`, `*.safetensors`, `*.gguf`,
   and `data/raw/` + `data/processed/` are gitignored (too heavy). Share artifacts
   out-of-band or via GitHub Releases.

## Commit conventions

- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`,
  `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Keep commits focused; no AI references in commit messages.
- Reference evaluation versions explicitly (e.g. `feat(distill): ...` with the
  `--label vX` in the report) so results are traceable.

## Evaluation discipline

Perplexity is only comparable at an identical truncation cap — always report the
cap alongside the number (see `plans/reports/evaluation-v0.5.md` for the
protocol). Never compare a cap-512 figure against a cap-2048 figure.

When you retrain, record: dataset version, split sizes, chat template, LoRA
config, best validation loss, and held-out PPL at 512/1024/2048.

## Reporting issues

Open an issue with the bug or feature template. For security matters, see
[`SECURITY.md`](SECURITY.md) — do not file public issues for vulnerabilities.
