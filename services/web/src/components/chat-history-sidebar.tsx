/** Conversation navigation for the browser-local chat history. */

import type { Conversation } from '../hooks/use-chat'

interface ChatHistorySidebarProps {
  conversations: readonly Conversation[]
  selectedConversationId: string | null
  disabled: boolean
  onNewConversation: () => void
  onSelectConversation: (id: string) => void
  onDeleteConversation: (id: string) => void
}

export function ChatHistorySidebar({
  conversations,
  selectedConversationId,
  disabled,
  onNewConversation,
  onSelectConversation,
  onDeleteConversation,
}: ChatHistorySidebarProps) {
  return (
    <aside className="history-sidebar" aria-label="Chat history">
      <button className="new-chat-button" type="button" onClick={onNewConversation} disabled={disabled}>
        + New chat
      </button>
      <nav className="conversation-list" aria-label="Saved conversations">
        {conversations.map((conversation) => (
          <div className="conversation-item" key={conversation.id}>
            <button
              className={`conversation-title ${conversation.id === selectedConversationId ? 'selected' : ''}`}
              type="button"
              aria-current={conversation.id === selectedConversationId ? 'page' : undefined}
              onClick={() => onSelectConversation(conversation.id)}
              disabled={disabled}
            >
              {conversation.title}
            </button>
            <button
              className="delete-conversation"
              type="button"
              aria-label={`Delete ${conversation.title}`}
              onClick={() => onDeleteConversation(conversation.id)}
              disabled={disabled}
            >
              ×
            </button>
          </div>
        ))}
      </nav>
    </aside>
  )
}