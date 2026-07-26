/** Chat state + streaming orchestration for the single conversation view. */

import { useCallback, useRef, useState } from 'react'
import { streamChatCompletion, type ChatMessage, type GenerationOptions } from '../api/client'

export interface UiMessage {
  role: 'user' | 'assistant'
  content: string
  error?: string
}

const SYSTEM_PROMPT = 'You are a helpful, knowledgeable assistant. Answer thoroughly and clearly.'

export function useChat() {
  const [messages, setMessages] = useState<UiMessage[]>([])
  const [busy, setBusy] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const send = useCallback(
    async (text: string, options: GenerationOptions) => {
      const userMessage: UiMessage = { role: 'user', content: text }
      setMessages((current) => [...current, userMessage, { role: 'assistant', content: '' }])
      setBusy(true)
      const controller = new AbortController()
      abortRef.current = controller

      const history: ChatMessage[] = [
        { role: 'system', content: SYSTEM_PROMPT },
        ...messages
          .filter((message) => !message.error && message.content)
          .map((message) => ({ role: message.role, content: message.content })),
        { role: 'user', content: text },
      ]

      const appendToLast = (updater: (last: UiMessage) => UiMessage) =>
        setMessages((current) => [...current.slice(0, -1), updater(current[current.length - 1])])

      try {
        await streamChatCompletion(history, options, {
          signal: controller.signal,
          onToken: (token) =>
            appendToLast((last) => ({ ...last, content: last.content + token })),
          onDone: () => setBusy(false),
        })
      } catch (error) {
        const aborted = controller.signal.aborted
        appendToLast((last) => ({
          ...last,
          error: aborted ? undefined : error instanceof Error ? error.message : String(error),
          content: last.content || (aborted ? last.content : ''),
        }))
        setBusy(false)
      } finally {
        abortRef.current = null
        setBusy(false)
      }
    },
    [messages],
  )

  const clear = useCallback(() => setMessages([]), [])

  return { messages, busy, send, stop, clear }
}
