import { describe, expect, it } from 'vitest'
import {
  type ChatHistoryStorage,
  createConversation,
  loadConversations,
  prepareConversationsForStorage,
  saveConversations,
  titleFromMessages,
  type Conversation,
} from './chat-history'

const storage = () => {
  const values = new Map<string, string>()
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
  } satisfies ChatHistoryStorage
}

describe('chat history', () => {
  it('creates an empty, untitled conversation', () => {
    const conversation = createConversation(123)
    expect(conversation).toMatchObject({
      title: 'New conversation',
      createdAt: 123,
      updatedAt: 123,
      messages: [],
    })
    expect(conversation.id).toMatch(/^123-/)
  })

  it('derives a compact title from the first user message', () => {
    expect(titleFromMessages([{ role: 'assistant', content: 'Hi' }, { role: 'user', content: '  Explain\n  SSE please ' }]))
      .toBe('Explain SSE please')
    expect(titleFromMessages([])).toBe('New conversation')
  })

  it('loads valid conversations and safely rejects malformed storage', () => {
    const validStorage = storage()
    const conversation: Conversation = { ...createConversation(123), title: 'Saved' }
    saveConversations([conversation], validStorage)
    expect(loadConversations(validStorage)).toEqual([conversation])

    const invalidStorage = {
      getItem: () => '{bad json',
      setItem: () => undefined,
    } satisfies ChatHistoryStorage
    expect(loadConversations(invalidStorage)).toEqual([])
  })

  it('does not persist incomplete or failed messages', () => {
    const conversation: Conversation = {
      ...createConversation(123),
      messages: [
        { role: 'user', content: 'question' },
        { role: 'assistant', content: '' },
        { role: 'assistant', content: 'failed', error: 'offline' },
        { role: 'assistant', content: 'answer' },
      ],
    }
    expect(prepareConversationsForStorage([conversation])[0].messages).toEqual([
      { role: 'user', content: 'question' },
      { role: 'assistant', content: 'answer' },
    ])
  })

  it('keeps the 30 most recently updated conversations', () => {
    const conversations = Array.from({ length: 31 }, (_, index) => ({
      ...createConversation(index),
      id: String(index),
      updatedAt: index,
    }))
    const saved = prepareConversationsForStorage(conversations)
    expect(saved).toHaveLength(30)
    expect(saved[0].id).toBe('30')
    expect(saved.at(-1)?.id).toBe('1')
  })
})