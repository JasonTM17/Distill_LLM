/** Single-view chat app for the distilled model. */

import { useEffect, useRef, useState } from 'react'
import { fetchReadiness } from './api/client'
import { ChatComposer } from './components/chat-composer'
import { ChatHistorySidebar } from './components/chat-history-sidebar'
import { MessageBubble } from './components/message-bubble'
import { useChat } from './hooks/use-chat'

type Readiness = 'ready' | 'loading' | 'down' | 'checking'

export default function App() {
  const {
    conversations, selectedConversationId, messages, busy, send, stop,
    newConversation, selectConversation, deleteConversation,
  } = useChat()
  const [readiness, setReadiness] = useState<Readiness>('checking')
  const scrollAnchor = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      const status = await fetchReadiness()
      if (!cancelled) setReadiness(status)
    }
    poll()
    const timer = setInterval(poll, 5000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    scrollAnchor.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="app-layout">
      <ChatHistorySidebar
        conversations={conversations}
        selectedConversationId={selectedConversationId}
        disabled={busy}
        onNewConversation={newConversation}
        onSelectConversation={selectConversation}
        onDeleteConversation={deleteConversation}
      />
      <div className="shell">
      <header>
        <h1>
          distill-gpt55 <span className="subtitle">Qwen2.5-1.5B · distilled from GPT-5.5</span>
        </h1>
        <div className="header-actions">
          <span className={`status-badge status-${readiness}`}>
            <span className="dot" />
            {readiness === 'ready' ? 'model ready'
              : readiness === 'loading' ? 'model loading'
              : readiness === 'down' ? 'API offline' : 'checking…'}
          </span>
          {messages.length > 0 ? (
            <button className="ghost" onClick={newConversation} disabled={busy}>
              New chat
            </button>
          ) : null}
        </div>
      </header>

      <main aria-live="polite" aria-relevant="additions text">
        {messages.length === 0 ? (
          <div className="empty-state">
            <p className="empty-title">Chat with a 1.5B model distilled on this machine</p>
            <p className="empty-hint">
              LoRA fine-tuned in bf16 on GPT-5.5-xhigh outputs · strongest at math and coding
            </p>
          </div>
        ) : (
          messages.map((message, index) => (
            <MessageBubble
              key={index}
              message={message}
              streaming={busy && index === messages.length - 1}
            />
          ))
        )}
        <div ref={scrollAnchor} />
      </main>

      <ChatComposer busy={busy} disabled={readiness !== 'ready'} onSend={send} onStop={stop} />
      </div>
    </div>
  )
}
