/**
 * Minimal SSE parser for the chat completions stream.
 *
 * Handles chunk boundaries landing mid-event: bytes are buffered until a
 * blank-line event terminator is seen, so a `data: {...}` split across two
 * network chunks parses correctly.
 */

export interface SseEvent {
  data: string
}

/** Incrementally consumes text chunks and yields complete SSE events. */
export class SseBuffer {
  private buffer = ''

  push(chunk: string): SseEvent[] {
    this.buffer += chunk
    const events: SseEvent[] = []
    // Events are separated by a blank line (\n\n); keep the trailing partial.
    let separatorIndex: number
    while ((separatorIndex = this.buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = this.buffer.slice(0, separatorIndex)
      this.buffer = this.buffer.slice(separatorIndex + 2)
      const dataLines = rawEvent
        .split('\n')
        .filter((line) => line.startsWith('data: '))
        .map((line) => line.slice('data: '.length))
      if (dataLines.length > 0) {
        events.push({ data: dataLines.join('\n') })
      }
    }
    return events
  }
}

export interface StreamDelta {
  content?: string
  finishReason?: string
  done: boolean
}

/** Decode one SSE event payload from the chat endpoint. */
export function decodeChatEvent(data: string): StreamDelta {
  if (data === '[DONE]') return { done: true }
  const parsed = JSON.parse(data)
  if (parsed.error) throw new Error(parsed.error.message ?? 'stream error')
  const choice = parsed.choices?.[0] ?? {}
  return {
    content: choice.delta?.content,
    finishReason: choice.finish_reason ?? undefined,
    done: false,
  }
}
