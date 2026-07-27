import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ChatHistorySidebar } from './chat-history-sidebar'

describe('ChatHistorySidebar', () => {
  it('selects, creates, and deletes conversations through its callbacks', () => {
    const onNewConversation = vi.fn()
    const onSelectConversation = vi.fn()
    const onDeleteConversation = vi.fn()
    render(
      <ChatHistorySidebar
        conversations={[{ id: 'one', title: 'Explain SSE', createdAt: 1, updatedAt: 1, messages: [] }]}
        selectedConversationId="one"
        disabled={false}
        onNewConversation={onNewConversation}
        onSelectConversation={onSelectConversation}
        onDeleteConversation={onDeleteConversation}
      />,
    )

    expect(screen.getByRole('button', { name: 'Explain SSE' }).getAttribute('aria-current')).toBe('page')
    fireEvent.click(screen.getByRole('button', { name: '+ New chat' }))
    fireEvent.click(screen.getByRole('button', { name: 'Explain SSE' }))
    fireEvent.click(screen.getByRole('button', { name: 'Delete Explain SSE' }))
    expect(onNewConversation).toHaveBeenCalledOnce()
    expect(onSelectConversation).toHaveBeenCalledWith('one')
    expect(onDeleteConversation).toHaveBeenCalledWith('one')
  })
})