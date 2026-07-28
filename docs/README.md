# Documentation index

The user-facing entry points are bilingual. Unsuffixed filenames remain the
stable canonical paths used by existing links; `.en.md` files are explicit
English entry points.

> **Ngôn ngữ / Languages:** xem trạng thái từng tài liệu bên dưới / see the
> per-document coverage below.

## Documents

| Document | Canonical path | English path |
|---|---|---|
| Project overview & PDR | [project-overview-pdr.md](project-overview-pdr.md) | [project-overview-pdr.en.md](project-overview-pdr.en.md) |
| System architecture | [system-architecture.md](system-architecture.md) | [system-architecture.en.md](system-architecture.en.md) |
| Deployment guide | [deployment-guide.md](deployment-guide.md) | [deployment-guide.en.md](deployment-guide.en.md) |
| Code standards | [code-standards.md](code-standards.md) | [code-standards.en.md](code-standards.en.md) |
| Codebase summary | [codebase-summary.md](codebase-summary.md) | [codebase-summary.en.md](codebase-summary.en.md) |
| Design guidelines | [design-guidelines.md](design-guidelines.md) | [design-guidelines.en.md](design-guidelines.en.md) |
| Project roadmap | [project-roadmap.md](project-roadmap.md) | [project-roadmap.en.md](project-roadmap.en.md) |
| API contract (OpenAPI) | [openapi.yaml](openapi.yaml) | — (machine-readable) |

## Language coverage

- `README`, project overview, deployment guide, and roadmap have dedicated
  Vietnamese and English content.
- Architecture, codebase, standards, and design references are currently
  English-first technical documents. Their `.en.md` mirrors are retained for
  stable language-specific links.
- `openapi.yaml` is machine-readable and language-neutral.

## Convention

- Unsuffixed files keep their original names so links from `README.md`,
  `CHANGELOG.md`, code comments, and plans stay valid.
- When a dedicated English mirror exists, keep its commands, figures, and
  factual claims synchronized with the canonical path in the same commit.
- Community files (`CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`) and
  GitHub issue/PR templates are intentionally English-only: they follow the
  global GitHub convention so contributors and security reporters are not forced
  to read Vietnamese.

## Root README

| Language | File |
|---|---|
| Tiếng Việt (canonical) | [../README.md](../README.md) |
| English | [../README.en.md](../README.en.md) |
