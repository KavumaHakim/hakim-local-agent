/**
 * The transcript: messages, the tools each turn ran, and the empty state.
 */

import { useEffect, useRef, useState } from 'react'
import { Markdown } from '../lib/markdown'
import type { Message, ToolCall } from '../lib/types'
import {
  AlertIcon,
  BrainIcon,
  CheckIcon,
  ChevronIcon,
  PencilIcon,
  RetryIcon,
  SparkIcon,
  ToolIcon,
} from './Icons'
import { CopyButton } from './CopyButton'

/**
 * The tool calls a turn made, each expandable to what was sent and what came
 * back.
 *
 * The collapsed row used to be all there was, and a one-line summary is not
 * enough to check the agent's work — which is the only reason to look. Expanded
 * shows the arguments it passed and the whole result payload, which is exactly
 * what the model itself saw.
 */
export function ToolPills({ tools }: { tools: ToolCall[] }) {
  if (!tools.length) return null
  return (
    <div className="mb-2 space-y-1">
      {tools.map((tool, index) => (
        <ToolCallRow key={`${tool.name}-${index}`} tool={tool} />
      ))}
    </div>
  )
}

function ToolCallRow({ tool }: { tool: ToolCall }) {
  // Nothing to expand for a turn loaded before these were recorded.
  const detailed = Boolean(tool.arguments || tool.output)

  return (
    <details
      className={`group overflow-hidden rounded-lg border ${
        tool.ok ? 'border-line bg-surface' : 'border-danger/40 bg-danger/5'
      }`}
    >
      <summary
        className={`flex cursor-pointer list-none items-center gap-2 px-2.5 py-1.5 text-xs ${
          detailed ? '' : 'cursor-default'
        }`}
      >
        {tool.ok ? (
          <CheckIcon className="size-3 shrink-0 text-ok" />
        ) : (
          <AlertIcon className="size-3 shrink-0 text-danger" />
        )}
        <span className="shrink-0 font-mono text-fg">{tool.name}</span>
        <span className="min-w-0 flex-1 truncate text-faint">{tool.summary}</span>
        {detailed && (
          <ChevronIcon className="size-3 shrink-0 text-faint transition group-open:rotate-90" />
        )}
      </summary>

      {detailed && (
        <div className="space-y-2 border-t border-line px-2.5 py-2">
          {tool.arguments && (
            <CodePane title="Sent" text={tool.arguments} />
          )}
          {tool.output && <CodePane title="Returned" text={tool.output} />}
          {tool.clipped && (
            <p className="text-[11px] text-faint">
              Shortened for display. The model received the whole thing.
            </p>
          )}
        </div>
      )}
    </details>
  )
}

function CodePane({ title, text }: { title: string; text: string }) {
  return (
    <div>
      <div className="mb-1 flex items-center gap-2">
        <span className="text-[11px] font-medium tracking-wide text-faint uppercase">
          {title}
        </span>
        <CopyButton text={text} className="ml-auto" />
      </div>
      <pre className="max-h-64 overflow-auto rounded-md bg-sunken p-2 text-[11px] leading-relaxed">
        <code className="font-mono whitespace-pre">{text}</code>
      </pre>
    </div>
  )
}

/**
 * The model's thinking, folded away by default.
 *
 * Collapsed because it is usually long and rambling, and the answer is what
 * you came for. Rendered as plain text rather than markdown: a thinking trace
 * is half-formed by nature and formatting it lends it a confidence it has not
 * earned.
 */
export function ReasoningPanel({
  text,
  live = false,
}: {
  text: string
  live?: boolean
}) {
  if (!text.trim()) return null
  return (
    <details className="group mb-2 rounded-xl border border-line bg-sunken/60">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs text-muted transition hover:text-fg">
        <BrainIcon
          className={`size-3.5 shrink-0 text-accent ${live ? 'animate-breathe' : ''}`}
        />
        <span>{live ? 'Thinking…' : 'Thinking'}</span>
        <span className="text-faint">
          {text.length.toLocaleString()} characters
        </span>
        <span className="ml-auto text-faint transition group-open:rotate-90">›</span>
      </summary>
      <div className="max-h-72 overflow-y-auto border-t border-line px-3 py-2">
        <div className="mb-1 flex">
          <CopyButton text={text} className="ml-auto" />
        </div>
        <p className="font-mono text-[12px] leading-relaxed whitespace-pre-wrap text-muted">
          {text}
        </p>
        {!live && (
          <p className="mt-2 border-t border-line pt-2 text-[11px] text-faint">
            Not saved — the model is never shown its own thinking again, and
            this disappears when the page reloads.
          </p>
        )}
      </div>
    </details>
  )
}

export function MessageView({
  message,
  onRetry,
  onEdit,
  editingBlocked,
}: {
  message: Message
  /** Only on the last assistant message: re-ask the preceding question. */
  onRetry?: () => void
  /** Rewind to this question, replace it, and ask again. */
  onEdit?: (text: string) => void
  /** Why editing is unavailable right now, if it is. */
  editingBlocked?: string
}) {
  if (message.role === 'user') {
    return (
      <UserMessage
        message={message}
        onEdit={onEdit}
        editingBlocked={editingBlocked}
      />
    )
  }

  return (
    <div className="group animate-rise">
      {message.reasoning && <ReasoningPanel text={message.reasoning} />}
      <ToolPills tools={message.tools} />
      <div className="text-[15px] text-fg">
        <Markdown text={message.content} />
      </div>
      <div className="mt-2 flex items-center gap-3 text-[11.5px] text-faint">
        {message.model_key && (
          <span className="font-mono">{message.model_key}</span>
        )}
        {message.elapsed != null && <span>{formatDuration(message.elapsed)}</span>}
        <div className="ml-auto flex gap-1 opacity-0 transition group-hover:opacity-100 focus-within:opacity-100">
          <CopyButton text={message.content} label="" />
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              title="Ask again"
              className="grid size-[22px] place-items-center rounded-sm text-faint transition hover:bg-tint hover:text-fg"
            >
              <RetryIcon className="size-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * A question, and the ability to change it.
 *
 * Editing rewinds: this question and everything that answered it are deleted,
 * and the new text is asked in their place. That is destructive, and worth
 * being plain about in the UI - the alternative, appending a correction, would
 * leave the model reading a conversation where the same thing is asked twice.
 */
function UserMessage({
  message,
  onEdit,
  editingBlocked,
}: {
  message: Message
  onEdit?: (text: string) => void
  editingBlocked?: string
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(message.content)
  const box = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (!editing) return
    const element = box.current
    if (!element) return
    element.focus()
    element.setSelectionRange(element.value.length, element.value.length)
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 260)}px`
  }, [editing, draft])

  function cancel() {
    setEditing(false)
    setDraft(message.content)
  }

  function save() {
    const trimmed = draft.trim()
    if (!trimmed || trimmed === message.content) {
      cancel()
      return
    }
    setEditing(false)
    onEdit?.(trimmed)
  }

  if (editing) {
    return (
      <div className="animate-rise">
        <div className="ml-auto w-[82%] rounded-[14px] border border-accent-line bg-raised p-2.5">
          <textarea
            ref={box}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Escape') cancel()
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                save()
              }
            }}
            className="max-h-[260px] w-full resize-none bg-transparent text-[14.5px] leading-relaxed outline-none"
          />
          <div className="mt-2 flex items-center gap-2">
            <p className="min-w-0 flex-1 text-[11px] leading-snug text-faint">
              This replaces the question. The answer it got, and everything
              after, are deleted.
            </p>
            <button
              type="button"
              onClick={cancel}
              className="h-7 shrink-0 rounded-md px-2 text-[12px] text-faint transition hover:text-fg"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={save}
              disabled={!draft.trim()}
              className="h-7 shrink-0 rounded-md border border-accent px-2.5 text-[12px] text-accent transition hover:bg-accent-tint disabled:opacity-30"
            >
              Ask again
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="group flex animate-rise justify-end gap-1">
      <div className="flex self-end opacity-0 transition group-hover:opacity-100 focus-within:opacity-100">
        {onEdit && (
          <button
            type="button"
            onClick={() => !editingBlocked && setEditing(true)}
            disabled={Boolean(editingBlocked)}
            title={editingBlocked || 'Edit and ask again'}
            className="grid size-[22px] place-items-center rounded-sm text-faint transition hover:bg-tint hover:text-fg disabled:cursor-not-allowed disabled:opacity-40"
          >
            <PencilIcon className="size-3.5" />
          </button>
        )}
        <CopyButton text={message.content} label="" />
      </div>
      {/* A raised surface, not an accent flood. The system uses the accent
          as a line and a glow; a solid violet bubble was the one large
          saturated fill in the old build. */}
      <div className="max-w-[82%] rounded-[14px_14px_4px_14px] bg-raised px-3.5 py-2.5 shadow-[var(--shadow-sm)]">
        <p className="text-[14.5px] leading-relaxed break-words whitespace-pre-wrap">
          {message.content}
        </p>
      </div>
    </div>
  )
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  return `${minutes}m ${rest}s`
}

const SUGGESTIONS = [
  'Calculate sqrt(144) + 25^2',
  'List the files in the workspace root',
  'Read requirements.txt and tell me what it depends on',
  'What is 17 * 43 - 209?',
]

export function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="mx-auto flex max-w-2xl flex-col items-center px-6 py-16 text-center">
      <div className="mb-5 flex size-12 items-center justify-center rounded-2xl border border-line bg-surface text-accent">
        <SparkIcon className="size-6" />
      </div>
      <h2 className="text-2xl font-semibold tracking-tight">
        What can I help with?
      </h2>
      <p className="mt-2 max-w-md text-sm text-muted">
        Ask a question, or give the agent a task it can use its tools for.
        Everything runs on this machine.
      </p>

      <div className="mt-8 grid w-full gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => onPick(suggestion)}
            className="group rounded-xl border border-line bg-surface px-4 py-3 text-left text-sm text-muted transition hover:border-accent/50 hover:bg-raised hover:text-fg"
          >
            <ToolIcon className="mb-1.5 size-3.5 text-faint transition group-hover:text-accent" />
            <span className="block">{suggestion}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
