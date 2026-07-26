import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { MessageBubble } from './message-bubble'

describe('MessageBubble', () => {
  it('renders user text verbatim (no markdown)', () => {
    render(
      <MessageBubble
        message={{ role: 'user', content: '**not bold** <b>plain</b>' }}
        streaming={false}
      />,
    )
    expect(screen.getByText('**not bold** <b>plain</b>')).toBeTruthy()
  })

  it('renders assistant markdown as HTML', () => {
    const { container } = render(
      <MessageBubble
        message={{ role: 'assistant', content: 'Here is `code` and **bold**' }}
        streaming={false}
      />,
    )
    expect(container.querySelector('code')?.textContent).toBe('code')
    expect(container.querySelector('strong')?.textContent).toBe('bold')
  })

  it('sanitizes script tags out of assistant output', () => {
    const { container } = render(
      <MessageBubble
        message={{ role: 'assistant', content: 'hi <script>alert(1)</script>' }}
        streaming={false}
      />,
    )
    expect(container.querySelector('script')).toBeNull()
  })

  it('shows typing indicator while streaming an empty message', () => {
    render(<MessageBubble message={{ role: 'assistant', content: '' }} streaming={true} />)
    expect(screen.getByLabelText('assistant is typing')).toBeTruthy()
  })

  it('shows the error line when present', () => {
    render(
      <MessageBubble
        message={{ role: 'assistant', content: '', error: 'API error 503' }}
        streaming={false}
      />,
    )
    expect(screen.getByText(/API error 503/)).toBeTruthy()
  })
})
