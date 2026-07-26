# Design Guidelines

## Scope, honestly

The UI surface of this project is **one view**: a single streaming chat page in
`services/web`. There is no design system here, no component library, no theming
layer, and no route hierarchy — because there is nothing to apply them to.

This document records what actually governs that one view, so changes stay
consistent with it. It does not invent guidelines the code does not follow. If the
UI grows past a second view, this file should grow with it; until then, a short
accurate page beats an aspirational one.

Everything below is drawn from `services/web/src/styles.css`, `App.tsx`,
`components/`, and `hooks/use-chat.ts`.

## Visual language

Dark, single-column, terminal-adjacent. Deliberately plain: the model output is the
content, and the chrome stays out of its way.

### Tokens

All colour lives in CSS custom properties on `:root` in `styles.css`. There are no
other colour sources — no inline styles, no per-component palettes. Use these names,
never a raw hex value, in new CSS.

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

- **Radii**: `10px` for interactive controls (buttons, textarea), `12px` for message
  bubbles, `8px` for code blocks, `999px` for the status pill.
- **Type scale**: sizes are in `rem` and stay small — `1.05rem` for the H1, `1.1rem`
  for the empty-state title, `0.8`–`0.88rem` for secondary text. Body line-height `1.55`.
- **Spacing**: multiples of 2px, mostly 4/6/8/10/12/14/16/20/24.
- **Fonts**: system UI stack for prose; `'Cascadia Code', Consolas, monospace` for code.
- **Layout**: `.shell` is a flex column at `height: 100vh`, capped at `max-width: 860px`
  and centred. Header and composer are fixed bands; `main` is the only scrolling region.

## Component structure

Five files, each with one job. Keep it that way — this is small enough that indirection
costs more than it saves.

| File | Responsibility |
|---|---|
| `App.tsx` | The only view. Composes header, message list, composer; polls readiness; auto-scrolls. |
| `hooks/use-chat.ts` | All conversation state and stream orchestration. Components hold no chat state. |
| `components/chat-composer.tsx` | Input, send/stop, and the collapsible generation settings. |
| `components/message-bubble.tsx` | Renders one message. Owns the user-vs-assistant rendering split. |
| `api/client.ts` + `api/sse.ts` | Transport. No UI concerns. |

The state rule: `use-chat` owns `messages`, `busy`, and the `AbortController`.
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
`⚠`-prefixed line in `--danger` inside the bubble. Errored messages are excluded from
the history sent on the next turn, so one failure does not poison the conversation.
There is no toast layer and no global error boundary — errors belong to the message
that caused them.

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
- The settings toggle has a `title`.
- Focus ring on the textarea (`outline: 1px solid var(--accent)`).
- Disabled controls use the real `disabled` attribute, not just styling.
- Enter submits, Shift+Enter inserts a newline.

Known gaps — fix these if a11y work is picked up:

- No live region on the message list, so screen readers are not told about streamed
  tokens arriving.
- The `⚙` toggle has a `title` but no `aria-label` or `aria-expanded`.
- Buttons rely on the default focus ring; there is no `:focus-visible` styling.
- No `prefers-reduced-motion` handling for the pulsing dot and typing animations.
- Status colour is reinforced by a text label, which is good; the dot alone is not
  distinguishable.

## Responsiveness

There are no media queries. The layout is a centred flex column with a `max-width`
and percentage-width bubbles (78%), which degrades acceptably on narrow screens but
has not been designed or tested for mobile. Treat mobile as unverified rather than
supported.

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
