# Design Guidelines

## Scope, honestly

The UI surface of this project is **one view**: a streaming chat page with
browser-local conversation history in `services/web`. There is no component
library, theming layer, or route hierarchy.

This document records what actually governs that one view, so changes stay
consistent with it. It does not invent guidelines the code does not follow. If the
UI grows past a second view, this file should grow with it; until then, a short
accurate page beats an aspirational one.

Everything below is drawn from `services/web/index.html`,
`services/web/src/styles.css`, `App.tsx`, `components/`, and
`hooks/use-chat.ts`.

## Visual language

Dark, single-column, terminal-adjacent. Deliberately plain: the model output is the
content, and the chrome stays out of its way.

### Tokens

The table below is the core palette in `:root`. A few legacy raw colours remain
in `styles.css`; new CSS should use an existing token or introduce a named token
instead of adding another raw value.

| Token | Value | Used for |
|---|---|---|
| `--bg` | `#0d1117` | Page background |
| `--panel` | `#161b22` | Bubbles, textarea, raised surfaces |
| `--border` | `#21262d` | Every divider and outline |
| `--text` | `#e6edf3` | Primary text |
| `--muted` | `#8b949e` | Secondary text, placeholders, idle dots |
| `--accent` | `#4493f8` | Primary buttons, focus ring, range thumb |
| `--accent-soft` | `#1c2d45` | User message bubble background |
| `--danger` | `#f85149` | Errors, stop button, offline dot |
| `--ok` | `#3fb950` | Ready state dot |
| `--warn` | `#d29922` | Loading state dot |

Conventions that are consistent across the sheet and worth preserving:

- **Radii**: `7px` for compact conversation controls, `10px` for primary controls
  (buttons, textarea), `12px` for message bubbles, `8px` for code blocks, and
  `999px` for the status pill.
- **Type scale**: sizes are in `rem` and stay small — `1.05rem` for the H1, `1.1rem`
  for the empty-state title, `0.8`–`0.88rem` for secondary text. Body line-height `1.55`.
- **Spacing**: multiples of 2px, mostly 4/6/8/10/12/14/16/20/24.
- **Fonts**: system UI stack for prose; `'Cascadia Code', Consolas, monospace` for code.
- **Layout**: `.app-layout` holds a 240px history sidebar and a centred `.shell` flex
  column. Header and composer are fixed bands; `main` is the only scrolling region.

## Component structure

Small components, each with one job. Keep it that way — this is small enough that indirection
costs more than it saves.

| File | Responsibility |
|---|---|
| `App.tsx` | The only view. Composes header, message list, composer; polls readiness; auto-scrolls. |
| `hooks/use-chat.ts` | Conversation state, local persistence, and stream orchestration. |
| `chat-history.ts` | Validates, bounds, and persists browser-local conversation data. |
| `components/chat-history-sidebar.tsx` | Presentational create/select/delete history navigation. |
| `components/chat-composer.tsx` | Input, send/stop, and the collapsible generation settings. |
| `components/message-bubble.tsx` | Renders one message. Owns the user-vs-assistant rendering split. |
| `api/client.ts` + `api/sse.ts` | Transport. No UI concerns. |

The state rule: `use-chat` owns conversations, selected messages, `busy`, and the `AbortController`.
Components receive values and callbacks as props and stay presentational. New UI state
that is genuinely local (the settings toggle, the draft text) stays in the component.

## Interaction and states

The interesting part of this UI is that it is streaming and remote-dependent, so
almost every state is a real state the user will hit — not a hypothetical.

### Readiness

`App` polls `GET /readyz` every 5 seconds and renders one of four badge states. The
model loads in a background thread on the api side, so "loading" is common on a cold
start, not an edge case.

| State | Dot colour | Label | Composer |
|---|---|---|---|
| `checking` | `--muted` | `checking…` | disabled |
| `ready` | `--ok` | `model ready` | enabled |
| `loading` | `--warn`, pulsing | `model loading` | disabled |
| `down` | `--danger` | `API offline` | disabled |

The composer is disabled unless the state is exactly `ready`, and its placeholder
changes to `API is not ready…`. Any new action that hits the model must respect the
same gate rather than failing at request time.

### Streaming

- Tokens append to the last assistant message as they arrive; the bubble grows in place.
- While the assistant message is still empty, the bubble shows `.typing-dots` — three
  staggered pulsing dots — instead of an empty box.
- The send button is replaced by a red **Stop** button while `busy`. Stop aborts via
  `AbortController`; an aborted stream keeps whatever text already arrived and records
  no error.
- The message list auto-scrolls to a bottom anchor on every change.

### Errors

Stream and request failures attach an `error` string to the message and render as a
`⚠`-prefixed line in `--danger` inside the bubble. The failed assistant message
is excluded from the next request, but the originating user message remains.
There is no toast layer or global error boundary — errors belong to the message
that caused them.

### Local conversation history

The sidebar creates, selects, and deletes browser-local conversations. The first
user prompt becomes a compact title. While generation is active, history navigation
is disabled so streamed tokens cannot be written into a different conversation.

Persistence is intentionally local-only: no authentication, sync, export, or
server-side history exists. At most 30 conversations and 100 non-empty,
non-error messages per conversation are stored. A stopped partial assistant
response has no error and is retained; empty or failed assistant messages are
excluded.

### Empty state

Before the first message, `main` shows a centred title and a one-line hint about what
the model is. There is no example-prompt grid. If one is added, it belongs in this
same `.empty-state` block.

### Generation settings

Collapsed by default behind a `⚙` ghost button. Temperature (0–1.5, step 0.1) and max
tokens (64–2048, step 64) are range inputs with their current value in the label.
Defaults are 0.7 and 512, matching the API defaults.

## Content rendering rules

This is the one part of the UI with a security consequence, so it is a rule rather
than a preference:

- **User text renders verbatim** in a plain `<p>`. Never parse user input as markdown.
- **Assistant text renders as markdown**, parsed with `marked` (`breaks: true`) and
  **always** sanitized with `DOMPurify` before it reaches `dangerouslySetInnerHTML`.

Model output is untrusted input. There is a test asserting that script tags are
stripped from assistant output; keep it passing. Any new surface that renders model
output takes the same path.

Code blocks inside assistant markdown get a darker background, a border, and
horizontal scroll rather than wrapping.

## Accessibility — current state

Stated plainly, including the gaps, so nobody assumes coverage that is not there.

Present:

- `<html lang="en">`, viewport meta, descriptive `<title>`.
- Semantic `header` / `main` / `form` landmarks.
- The typing indicator carries `aria-label="assistant is typing"`.
- History navigation uses labelled `aside` and `nav` landmarks; its selected item
  exposes `aria-current` and delete buttons have explicit names.
- The message list is a polite live region for streamed output.
- The settings toggle has an accessible name and `aria-expanded`.
- Controls have visible `:focus-visible` rings.
- Disabled controls use the real `disabled` attribute, not just styling.
- Enter submits, Shift+Enter inserts a newline.
- Motion-sensitive users get reduced animation and non-smooth scrolling.

Status colour is reinforced by a text label; the dot alone is not the status signal.

## Responsiveness

At 720px and below, the layout becomes vertical: history becomes a horizontally
scrollable strip, the content shell accounts for it, headers stack safely, settings
wrap, and message bubbles expand to 92%. Desktop retains the 240px sidebar.

## When adding UI

1. Reuse the tokens; do not introduce a colour outside `:root`.
2. Follow the existing radius and type scale rather than picking new values.
3. Keep chat state in `use-chat`; keep components presentational.
4. Give every remote-dependent control a disabled state tied to readiness.
5. Render any model output through the sanitize path.
6. If you add a second view, add routing and revisit this document — the
   single-view assumption is load-bearing in `App.tsx`.

## Deliberately absent

No component library, no CSS framework, no design tokens package, no dark/light
toggle, no icon set (two Unicode glyphs, `⚙` and `⚠`, do the job), no animation
library, no state manager. Each of these would be scaffolding for a UI that does not
exist yet. Add one when a second view makes it pay for itself, not before.
