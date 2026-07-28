# Hướng dẫn triển khai

> **Ngôn ngữ:** **Tiếng Việt** · [English](deployment-guide.en.md)

## Điều kiện

- Docker Desktop với Compose v2.
- Model `checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf`.

Model không được lưu trong Git và không nằm trong container image. Nếu chưa có
artifact, cần chạy pipeline train → merge → `python -m distill.export_gguf`.
Serving dùng llama.cpp trên CPU; không cần GPU hay 9Router.

## Chạy bằng Docker Compose

### Dùng image đã phát hành

`docker-compose.yml` trỏ tới Docker Hub:

```bash
docker compose pull
docker compose up --no-build
```

### Build từ source

```bash
docker compose up --build
```

Web mở tại http://localhost:3000; API mở tại http://localhost:8000. Thư mục
`./checkpoints/gguf` được mount read-only vào `/models`; model không bị đóng gói
vào image.

## Trạng thái khởi động

| Endpoint/trạng thái | Ý nghĩa | Kết quả |
|---|---|---|
| `GET /healthz` | Process FastAPI còn sống | `200 {"status":"ok"}` |
| `/readyz`: `loading` | Runtime đang nạp GGUF | `503` |
| `/readyz`: `ready` | Có thể nhận inference | `200` |
| `/readyz`: `error` | Nạp model thất bại | `503` kèm `detail` |

Healthcheck trong API image dùng `/healthz`, nên container chạy độc lập có thể
được đánh dấu healthy khi model vẫn đang nạp. Compose override healthcheck bằng
`/readyz`; web chỉ khởi động sau khi API sẵn sàng.

```bash
docker compose logs api
curl http://localhost:8000/readyz
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}]}'
```

## Registry và tag

Mỗi push lên `master` publish hai image với tag `latest` và SHA đầy đủ của
commit:

| Registry | API | Web |
|---|---|---|
| GHCR (luôn publish) | `ghcr.io/jasontm17/distill-gpt55-api` | `ghcr.io/jasontm17/distill-gpt55-web` |
| Docker Hub mirror | `nguyenson1710/distill-gpt55-api` | `nguyenson1710/distill-gpt55-web` |

GHCR dùng `GITHUB_TOKEN`. Docker Hub chỉ chạy khi repository có
`DOCKERHUB_USERNAME` và `DOCKERHUB_TOKEN`.

Pull trực tiếp từ GHCR:

```bash
docker pull ghcr.io/jasontm17/distill-gpt55-api:latest
docker pull ghcr.io/jasontm17/distill-gpt55-web:latest
```

Compose mặc định dùng tên Docker Hub. Muốn chạy GHCR với Compose, đổi hai trường
`image:` sang tên GHCR hoặc dùng một Compose override riêng.

## CI

- `.github/workflows/ci.yml`: chạy trên mọi pull request và push vào `master`;
  gồm Ruff, pytest core/API, kiểm tra generated client, Vitest và web build.
- `.github/workflows/docker-publish.yml`: build/publish API và web khi push vào
  `master`.

## Chạy không dùng Docker

### API — PowerShell

```powershell
cd services/api
pip install -r requirements-dev.txt
$env:MODEL_PATH='../../checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf'
uvicorn app.main:app --port 8000
```

### API — Bash

```bash
cd services/api
pip install -r requirements-dev.txt
MODEL_PATH=../../checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf \
  uvicorn app.main:app --port 8000
```

### Web

```bash
cd services/web
pnpm install
pnpm dev
```

`VITE_API_BASE_URL` là biến build-time, mặc định `http://localhost:8000`. Nếu
web và API khác origin, đặt biến này trước khi build/dev và thêm origin của web
vào `CORS_ALLOW_ORIGINS`.

OpenAI SDK có hỗ trợ custom base URL có thể gọi `chat.completions`. API chỉ
triển khai một phần contract OpenAI; không có `/v1/models`, embeddings hoặc
authentication, và API key truyền vào không được kiểm tra.

## Dữ liệu chat

History chỉ nằm trong `localStorage` của browser: tối đa 30 conversation và 100
message không rỗng, không lỗi mỗi conversation. Output dừng giữa chừng vẫn được
lưu. Container restart, volume backup và image rollback không khôi phục history.

## Rollback

Đổi tag image từ `latest` sang SHA đã biết, sau đó chạy:

```bash
docker compose pull
docker compose up --no-build
```

Không dùng `--build` khi rollback image vì build local sẽ tạo image mới. Rollback
model là đổi `MODEL_PATH` hoặc file mount sang GGUF đã xác minh trước đó.

## Tham chiếu

- [API runbook](../services/api/README.md)
- [Frontend guide](../services/web/README.md)
- [Kiến trúc hệ thống](system-architecture.md)
- [Security policy](../SECURITY.md)
