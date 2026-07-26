/** Message input with send/stop controls and generation settings. */

import { useState, type FormEvent } from 'react'
import type { GenerationOptions } from '../api/client'

interface ChatComposerProps {
  busy: boolean
  disabled: boolean
  onSend: (text: string, options: GenerationOptions) => void
  onStop: () => void
}

export function ChatComposer({ busy, disabled, onSend, onStop }: ChatComposerProps) {
  const [text, setText] = useState('')
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(512)
  const [showSettings, setShowSettings] = useState(false)

  const submit = (event: FormEvent) => {
    event.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || busy || disabled) return
    onSend(trimmed, { temperature, maxTokens })
    setText('')
  }

  return (
    <form className="composer" onSubmit={submit}>
      {showSettings ? (
        <div className="settings-row">
          <label>
            temperature {temperature.toFixed(1)}
            <input
              type="range" min={0} max={1.5} step={0.1} value={temperature}
              onChange={(event) => setTemperature(Number(event.target.value))}
            />
          </label>
          <label>
            max tokens {maxTokens}
            <input
              type="range" min={64} max={2048} step={64} value={maxTokens}
              onChange={(event) => setMaxTokens(Number(event.target.value))}
            />
          </label>
        </div>
      ) : null}
      <div className="composer-row">
        <button
          type="button" className="ghost" title="Generation settings"
          onClick={() => setShowSettings((value) => !value)}
        >
          ⚙
        </button>
        <textarea
          value={text}
          placeholder={disabled ? 'API is not ready…' : 'Ask the distilled model anything…'}
          disabled={disabled}
          rows={1}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) submit(event)
          }}
        />
        {busy ? (
          <button type="button" className="stop" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="submit" disabled={disabled || !text.trim()}>
            Send
          </button>
        )}
      </div>
    </form>
  )
}
