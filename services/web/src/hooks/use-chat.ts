/** Chat history state and streaming orchestration for local-first conversations. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { streamChatCompletion, type ChatMessage, type GenerationOptions } from '../api/client'
import {
  createConversation,
  loadConversations,
  saveConversations,
  titleFromMessages,
  type Conversation,
  type UiMessage,
} from '../chat-history'

export type { Conversation, UiMessage } from '../chat-history'

const SYSTEM_PROMPT = 'You are a helpful, knowledgeable assistant. Answer thoroughly and clearly.'

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations)
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const selectedConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === selectedConversationId) ?? null,
    [conversations, selectedConversationId],
  )
  const messages = selectedConversation?.messages ?? []

  useEffect(() => {
    if (!busy) saveConversations(conversations)
  }, [busy, conversations])

  const updateConversation = useCallback((id: string, updater: (current: Conversation) => Conversation) => {
    setConversations((current) => current.map((conversation) => (
      conversation.id === id ? updater(conversation) : conversation
    )))
  }, [])

  const newConversation = useCallback(() => {
    if (busy) return
    const conversation = createConversation()
    setConversations((current) => [conversation, ...current])
    setSelectedConversationId(conversation.id)
  }, [busy])

  const selectConversation = useCallback((id: string) => {
    if (!busy) setSelectedConversationId(id)
  }, [busy])

  const deleteConversation = useCallback((id: string) => {
    if (busy) return
    setConversations((current) => {
      const remaining = current.filter((conversation) => conversation.id !== id)
      if (selectedConversationId === id) setSelectedConversationId(remaining[0]?.id ?? null)
      return remaining
    })
  }, [busy, selectedConversationId])

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const send = useCallback(async (text: string, options: GenerationOptions) => {
    let conversation = selectedConversation
    if (!conversation) {
      conversation = createConversation()
      setConversations((current) => [conversation!, ...current])
      setSelectedConversationId(conversation.id)
    }

    const conversationId = conversation.id
    const userMessage: UiMessage = { role: 'user', content: text }
    const history: ChatMessage[] = [
      { role: 'system', content: SYSTEM_PROMPT },
      ...conversation.messages
        .filter((message) => !message.error && message.content)
        .map((message) => ({ role: message.role, content: message.content })),
      { role: 'user', content: text },
    ]
    const startedAt = Date.now()
    updateConversation(conversationId, (current) => {
      const pendingAssistantMessage: UiMessage = { role: 'assistant', content: '' }
      const messages = [...current.messages, userMessage, pendingAssistantMessage]
      return { ...current, title: titleFromMessages(messages), updatedAt: startedAt, messages }
    })
    setBusy(true)
    const controller = new AbortController()
    abortRef.current = controller

    const updateLastMessage = (updater: (last: UiMessage) => UiMessage) => {
      updateConversation(conversationId, (current) => ({
        ...current,
        updatedAt: Date.now(),
        messages: [...current.messages.slice(0, -1), updater(current.messages[current.messages.length - 1])],
      }))
    }

    try {
      await streamChatCompletion(history, options, {
        signal: controller.signal,
        onToken: (token) => updateLastMessage((last) => ({ ...last, content: last.content + token })),
        onDone: () => setBusy(false),
      })
    } catch (error) {
      const aborted = controller.signal.aborted
      updateLastMessage((last) => ({
        ...last,
        error: aborted ? undefined : error instanceof Error ? error.message : String(error),
      }))
    } finally {
      abortRef.current = null
      setBusy(false)
    }
  }, [selectedConversation, updateConversation])

  return {
    conversations,
    selectedConversationId,
    messages,
    busy,
    send,
    stop,
    newConversation,
    selectConversation,
    deleteConversation,
  }
}