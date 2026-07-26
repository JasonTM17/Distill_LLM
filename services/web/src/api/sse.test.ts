import { describe, expect, it } from 'vitest'
import { SseBuffer, decodeChatEvent } from './sse'

const chunk = (content: string) =>
  `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content }, finish_reason: null }] })}\n\n`

describe('SseBuffer', () => {
  it('parses complete events', () => {
    const buffer = new SseBuffer()
    const events = buffer.push(chunk('Hello') + chunk(' world'))
    expect(events).toHaveLength(2)
  })

  it('handles an event split across two network chunks', () => {
    const buffer = new SseBuffer()
    const full = chunk('Xin chào')
    const first = buffer.push(full.slice(0, 12))
    expect(first).toHaveLength(0)
    const second = buffer.push(full.slice(12))
    expect(second).toHaveLength(1)
    expect(decodeChatEvent(second[0].data).content).toBe('Xin chào')
  })

  it('keeps trailing partial data buffered', () => {
    const buffer = new SseBuffer()
    const events = buffer.push(chunk('a') + 'data: {"partial')
    expect(events).toHaveLength(1)
    const rest = buffer.push('": true}\n\n')
    expect(rest).toHaveLength(1)
  })

  it('ignores non-data lines', () => {
    const buffer = new SseBuffer()
    const events = buffer.push(': comment\nevent: ping\n\n')
    expect(events).toHaveLength(0)
  })
})

describe('decodeChatEvent', () => {
  it('detects the DONE sentinel', () => {
    expect(decodeChatEvent('[DONE]').done).toBe(true)
  })

  it('extracts content and finish reason', () => {
    const delta = decodeChatEvent(
      JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }),
    )
    expect(delta.finishReason).toBe('stop')
    expect(delta.done).toBe(false)
  })

  it('throws on stream errors', () => {
    expect(() =>
      decodeChatEvent(JSON.stringify({ error: { message: 'model_not_ready' } })),
    ).toThrow('model_not_ready')
  })
})
