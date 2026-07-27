/** Local-first chat history domain model and browser storage boundary. */

export interface UiMessage {
  role: 'user' | 'assistant'
  content: string
  error?: string
}

export interface Conversation {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  messages: UiMessage[]
}

const STORAGE_KEY = 'distill-gpt55.chat-history.v1'
const MAX_CONVERSATIONS = 30
const MAX_MESSAGES_PER_CONVERSATION = 100
const UNTITLED_CONVERSATION = 'New conversation'

export interface ChatHistoryStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
}

function isUiMessage(value: unknown): value is UiMessage {
  if (!value || typeof value !== 'object') return false
  const message = value as Record<string, unknown>
  return (
    (message.role === 'user' || message.role === 'assistant') &&
    typeof message.content === 'string' &&
    (message.error === undefined || typeof message.error === 'string')
  )
}

function isConversation(value: unknown): value is Conversation {
  if (!value || typeof value !== 'object') return false
  const conversation = value as Record<string, unknown>
  return (
    typeof conversation.id === 'string' &&
    typeof conversation.title === 'string' &&
    typeof conversation.createdAt === 'number' &&
    typeof conversation.updatedAt === 'number' &&
    Array.isArray(conversation.messages) &&
    conversation.messages.every(isUiMessage)
  )
}

export function createConversation(now = Date.now()): Conversation {
  return {
    id: `${now}-${Math.random().toString(36).slice(2, 10)}`,
    title: UNTITLED_CONVERSATION,
    createdAt: now,
    updatedAt: now,
    messages: [],
  }
}

export function titleFromMessages(messages: readonly UiMessage[]): string {
  const firstUserMessage = messages.find((message) => message.role === 'user' && message.content.trim())
  if (!firstUserMessage) return UNTITLED_CONVERSATION

  const title = firstUserMessage.content.replace(/\s+/g, ' ').trim()
  return title.length > 48 ? `${title.slice(0, 47).trimEnd()}…` : title
}

export function prepareConversationsForStorage(conversations: readonly Conversation[]): Conversation[] {
  return [...conversations]
    .sort((first, second) => second.updatedAt - first.updatedAt)
    .slice(0, MAX_CONVERSATIONS)
    .map((conversation) => ({
      ...conversation,
      messages: conversation.messages
        .filter((message) => !message.error && message.content)
        .slice(-MAX_MESSAGES_PER_CONVERSATION),
    }))
}

export function loadConversations(
  storage: ChatHistoryStorage | null = window.localStorage,
): Conversation[] {
  try {
    const rawValue = storage?.getItem(STORAGE_KEY)
    if (!rawValue) return []

    const parsed: unknown = JSON.parse(rawValue)
    if (!Array.isArray(parsed) || !parsed.every(isConversation)) return []
    return prepareConversationsForStorage(parsed)
  } catch {
    return []
  }
}

export function saveConversations(
  conversations: readonly Conversation[],
  storage: ChatHistoryStorage | null = window.localStorage,
): void {
  try {
    storage?.setItem(STORAGE_KEY, JSON.stringify(prepareConversationsForStorage(conversations)))
  } catch {
    // Storage can be unavailable or full; the in-memory conversation remains usable.
  }
}