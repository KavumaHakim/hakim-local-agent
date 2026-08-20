/**
 * ⌘K / Ctrl+K palette.
 *
 * Covers the commands, the models and the saved conversations in one list, so
 * the things you reach for by reflex do not need the sidebar open.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { COMMANDS, type CommandId } from '../lib/commands'
import type { Conversation, Model } from '../lib/types'
import { ChatIcon, ChipIcon, CommandIcon } from './Icons'

export interface PaletteAction {
  key: string
  group: 'Commands' | 'Models' | 'Conversations'
  label: string
  hint: string
  run: () => void
}

interface Props {
  open: boolean
  onClose: () => void
  models: Model[]
  conversations: Conversation[]
  onCommand: (id: CommandId, argument?: string) => void
  onSelectModel: (key: string) => void
  onOpenConversation: (id: number) => void
}

export function CommandPalette({
  open,
  onClose,
  models,
  conversations,
  onCommand,
  onSelectModel,
  onOpenConversation,
}: Props) {
  const [query, setQuery] = useState('')
  const [index, setIndex] = useState(0)
  const input = useRef<HTMLInputElement>(null)

  const actions = useMemo<PaletteAction[]>(() => {
    const entries: PaletteAction[] = COMMANDS.filter(
      // Needs an argument, so it is not a single-click action here; the
      // models section below covers the same ground properly.
      (command) => !command.argument,
    ).map((command) => ({
      key: `command:${command.id}`,
      group: 'Commands',
      label: command.title,
      hint: command.slash,
      run: () => onCommand(command.id),
    }))

    for (const model of models) {
      entries.push({
        key: `model:${model.key}`,
        group: 'Models',
        label: model.label,
        hint: model.available
          ? model.state === 'ready'
            ? 'loaded'
            : model.key
          : 'file missing',
        run: () => onSelectModel(model.key),
      })
    }

    for (const conversation of conversations) {
      entries.push({
        key: `conversation:${conversation.id}`,
        group: 'Conversations',
        label: conversation.title,
        hint: `${conversation.message_count} messages`,
        run: () => onOpenConversation(conversation.id),
      })
    }

    return entries
  }, [models, conversations, onCommand, onSelectModel, onOpenConversation])

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return actions
    return actions.filter(
      (action) =>
        action.label.toLowerCase().includes(needle) ||
        action.hint.toLowerCase().includes(needle),
    )
  }, [actions, query])

  useEffect(() => {
    if (open) {
      setQuery('')
      setIndex(0)
      // Focus after the element exists, not during the render that creates it.
      const timer = window.setTimeout(() => input.current?.focus(), 0)
      return () => window.clearTimeout(timer)
    }
  }, [open])

  useEffect(() => setIndex(0), [query])

  if (!open) return null

  function choose(action: PaletteAction | undefined) {
    if (!action) return
    action.run()
    onClose()
  }

  function onKeyDown(event: React.KeyboardEvent) {
    if (event.key === 'Escape') {
      event.preventDefault()
      onClose()
    } else if (event.key === 'ArrowDown') {
      event.preventDefault()
      setIndex((current) => (current + 1) % Math.max(filtered.length, 1))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setIndex(
        (current) =>
          (current - 1 + Math.max(filtered.length, 1)) %
          Math.max(filtered.length, 1),
      )
    } else if (event.key === 'Enter') {
      event.preventDefault()
      choose(filtered[index])
    }
  }

  let lastGroup = ''

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/50 p-4 pt-[12vh] backdrop-blur-sm"
      onMouseDown={onClose}
    >
      <div
        className="w-full max-w-xl overflow-hidden rounded-2xl border border-line bg-raised shadow-2xl"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b border-line px-4">
          <CommandIcon className="size-4 text-faint" />
          <input
            ref={input}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Commands, models, conversations…"
            className="flex-1 bg-transparent py-3.5 text-sm outline-none placeholder:text-faint"
          />
          <kbd className="rounded border border-line px-1.5 py-0.5 text-[10px] text-faint">
            esc
          </kbd>
        </div>

        <div className="max-h-[50vh] overflow-y-auto p-1.5">
          {filtered.length === 0 && (
            <p className="px-3 py-6 text-center text-sm text-faint">
              Nothing matches “{query}”.
            </p>
          )}

          {filtered.map((action, position) => {
            const header = action.group !== lastGroup ? action.group : null
            lastGroup = action.group
            return (
              <div key={action.key}>
                {header && (
                  <p className="px-3 pt-3 pb-1 text-[11px] font-medium tracking-wide text-faint uppercase">
                    {header}
                  </p>
                )}
                <button
                  type="button"
                  onMouseEnter={() => setIndex(position)}
                  onClick={() => choose(action)}
                  className={`flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm transition ${
                    position === index ? 'bg-accent-dim text-fg' : 'text-muted'
                  }`}
                >
                  {action.group === 'Models' ? (
                    <ChipIcon className="size-3.5 shrink-0 text-faint" />
                  ) : action.group === 'Conversations' ? (
                    <ChatIcon className="size-3.5 shrink-0 text-faint" />
                  ) : (
                    <CommandIcon className="size-3.5 shrink-0 text-faint" />
                  )}
                  <span className="truncate">{action.label}</span>
                  <span className="ml-auto shrink-0 font-mono text-xs text-faint">
                    {action.hint}
                  </span>
                </button>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
