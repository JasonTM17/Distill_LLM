/** One chat message: user text verbatim, assistant text as sanitized markdown. */

import DOMPurify from 'dompurify'
import { marked } from 'marked'
import type { UiMessage } from '../hooks/use-chat'

marked.setOptions({ breaks: true })

export function MessageBubble({ message, streaming }: { message: UiMessage; streaming: boolean }) {
  const isUser = message.role === 'user'
  return (
    <div className={`bubble-row ${isUser ? 'from-user' : 'from-assistant'}`}>
      <div className="bubble">
        {isUser ? (
          <p>{message.content}</p>
        ) : message.content ? (
          <div
            className="markdown"
            dangerouslySetInnerHTML={{
              __html: DOMPurify.sanitize(marked.parse(message.content, { async: false })),
            }}
          />
        ) : streaming ? (
          <span className="typing-dots" aria-label="assistant is typing">
            <span /><span /><span />
          </span>
        ) : null}
        {message.error ? <p className="bubble-error">⚠ {message.error}</p> : null}
      </div>
    </div>
  )
}
