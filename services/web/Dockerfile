# Build stage: compile the Vite bundle
FROM node:26-alpine AS builder

RUN corepack enable
WORKDIR /build
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN pnpm build

# Runtime stage: static files behind nginx, non-root
FROM nginxinc/nginx-unprivileged:1.31-alpine

ARG REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/JasonTM17/Distill_LLM" \
      org.opencontainers.image.description="Streaming web chat interface for the distill-gpt55 model" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${REVISION}"

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /build/dist /usr/share/nginx/html

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -q --spider http://localhost:3000/ || exit 1
