import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { streamChatCompletion } = vi.hoisted(() => ({ streamChatCompletion: vi.fn() }))

vi.mock('../api/client', () => ({ streamChatCompletion }))

import { useChat } from './use-chat'

describe('useChat', () => {
  beforeEach(() => {
    localStorage.clear()
    streamChatCompletion.mockReset()
    streamChatCompletion.mockImplementation(async (_history, _options, callbacks) => {
      callbacks.onToken('answer')
      callbacks.onDone('stop')
    })
  })

  it('creates a selected conversation and streams into it on the first send', async () => {
    const { result } = renderHook(() => useChat())

    await act(() => result.current.send('Explain SSE', { temperature: 0.7 }))

    expect(result.current.conversations).toHaveLength(1)
    expect(result.current.selectedConversationId).toBe(result.current.conversations[0].id)
    expect(result.current.conversations[0]).toMatchObject({
      title: 'Explain SSE',
      messages: [
        { role: 'user', content: 'Explain SSE' },
        { role: 'assistant', content: 'answer' },
      ],
    })
    expect(streamChatCompletion).toHaveBeenCalledWith(
      expect.arrayContaining([{ role: 'user', content: 'Explain SSE' }]),
      { temperature: 0.7 },
      expect.any(Object),
    )
  })

  it('selects and deletes saved conversations', () => {
    const { result } = renderHook(() => useChat())

    act(() => {
      result.current.newConversation()
      result.current.newConversation()
    })
    const [newest, oldest] = result.current.conversations
    expect(result.current.selectedConversationId).toBe(newest.id)

    act(() => result.current.selectConversation(oldest.id))
    expect(result.current.selectedConversationId).toBe(oldest.id)

    act(() => result.current.deleteConversation(oldest.id))
    expect(result.current.conversations).toEqual([newest])
    expect(result.current.selectedConversationId).toBe(newest.id)
  })
})