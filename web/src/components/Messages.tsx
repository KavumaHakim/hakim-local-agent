/**
 * The transcript: messages, the tools each turn ran, and the empty state.
 */

import { Markdown } from '../lib/markdown'
import type { Message, ToolCall } from '../lib/types'
import { AlertIcon, CheckIcon, SparkIcon, ToolIcon } from './Icons'

export function ToolPills({ tools }: { tools: ToolCall[] }) {
  if (!tools.length) return null
  return (
    <div className="mb-2 flex flex-wrap gap-1.5">
      {tools.map((tool, index) => (
        <span
          key={`${tool.name}-${index}`}
          title={tool.summary}
          className={`inline-flex max-w-full items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs ${
            tool.ok
              ? 'border-line bg-surface text-muted'
              : 'border-danger/40 bg-danger/10 text-danger'
          }`}
        >
          {tool.ok ? (
            <CheckIcon className="size-3 shrink-0 text-ok" />
          ) : (
            <AlertIcon className="size-3 shrink-0" />
          )}
          <span className="font-mono">{tool.name}</span>
          {tool.summary && (
            <span className="truncate text-faint">{tool.summary}</span>
          )}
        </span>
      ))}
    </div>
  )
}

export function MessageView({ message }: { message: Message }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end animate-rise">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-white shadow-sm">
          <p className="whitespace-pre-wrap break-words leading-relaxed">
            {message.content}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="animate-rise">
      <ToolPills tools={message.tools} />
      <div className="text-[15px] text-fg">
        <Markdown text={message.content} />
      </div>
      {(message.elapsed || message.model_key) && (
        <div className="mt-1.5 flex items-center gap-2 text-xs text-faint">
          {message.model_key && <span className="font-mono">{message.model_key}</span>}
          {message.elapsed != null && <span>{formatDuration(message.elapsed)}</span>}
        </div>
      )}
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
