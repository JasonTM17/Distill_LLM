/**
 * Typed API client. Request/response shapes come from the generated
 * `schema.d.ts` (openapi-typescript over docs/openapi.yaml) — do not
 * hand-write contract types here.
 */

import type { components } from './schema'
import { SseBuffer, decodeChatEvent } from './sse'

export type ChatMessage = components['schemas']['ChatMessage']
export type ChatCompletionRequest = components['schemas']['ChatCompletionRequest']
export type ChatCompletionResponse = components['schemas']['ChatCompletionResponse']

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'

export interface StreamCallbacks {
  onToken: (token: string) => void
  onDone: (finishReason: string | null) => void
  signal?: AbortSignal
}

export async function fetchReadiness(): Promise<'ready' | 'loading' | 'down'> {
  try {
    const response = await fetch(`${API_BASE_URL}/readyz`)
    return response.ok ? 'ready' : 'loading'
  } catch {
    return 'down'
  }
}

export interface GenerationOptions {
  temperature?: number
  maxTokens?: number
}

export async function streamChatCompletion(
  messages: ChatMessage[],
  options: GenerationOptions,
  callbacks: StreamCallbacks,
): Promise<void> {
  const body: ChatCompletionRequest = {
    messages,
    stream: true,
    temperature: options.temperature ?? null,
    max_tokens: options.maxTokens ?? null,
  }
  const response = await fetch(`${API_BASE_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: callbacks.signal,
  })
  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => '')
    throw new Error(`API error ${response.status}: ${detail.slice(0, 300)}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const sse = new SseBuffer()
  let finishReason: string | null = null

  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    for (const event of sse.push(decoder.decode(value, { stream: true }))) {
      const delta = decodeChatEvent(event.data)
      if (delta.done) {
        callbacks.onDone(finishReason)
        return
      }
      if (delta.finishReason) finishReason = delta.finishReason
      if (delta.content) callbacks.onToken(delta.content)
    }
  }
  callbacks.onDone(finishReason)
}
