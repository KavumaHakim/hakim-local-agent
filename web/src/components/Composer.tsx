/**
 * The message box, with inline slash-command completion.
 *
 * Streamlit could not do this: `st.chat_input` takes no children, so the old
 * build injected a dropdown through a same-origin iframe reaching into the
 * page's DOM. Here it is just a component.
 */

import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { matchCommands, type CommandSpec } from '../lib/commands'
import { CommandIcon, SendIcon } from './Icons'

interface Props {
  value: string
  onChange: (value: string) => void
  onSubmit: (value: string) => void
  disabled: boolean
  placeholder: string
}

export function Composer({
  value,
  onChange,
  onSubmit,
  disabled,
  placeholder,
}: Props) {
  const box = useRef<HTMLTextAreaElement>(null)
  const [highlighted, setHighlighted] = useState(0)

  const suggestions = matchCommands(value)
  const showing = suggestions.length > 0 && !value.includes('\n')

  // Grow with the text, up to a limit, then scroll inside.
  useLayoutEffect(() => {
    const element = box.current
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 240)}px`
  }, [value])

  useEffect(() => setHighlighted(0), [value])

  function accept(command: CommandSpec) {
    onChange(command.argument ? `${command.slash} ` : command.slash)
    box.current?.focus()
  }

  function submit() {
    if (disabled) return
    const text = value.trim()
    if (!text) return
    onSubmit(text)
  }

  function onKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (showing) {
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        setHighlighted((current) => (current + 1) % suggestions.length)
        return
      }
      if (event.key === 'ArrowUp') {
        event.preventDefault()
        setHighlighted(
          (current) => (current - 1 + suggestions.length) % suggestions.length,
        )
        return
      }
      if (event.key === 'Tab' || (event.key === 'Enter' && !event.shiftKey)) {
        event.preventDefault()
        accept(suggestions[highlighted])
        return
      }
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <div className="relative">
      {showing && (
        <div className="absolute bottom-full left-0 z-20 mb-2 w-full overflow-hidden rounded-xl border border-line bg-raised shadow-xl">
          {suggestions.map((command, index) => (
            <button
              key={command.id}
              type="button"
              onMouseEnter={() => setHighlighted(index)}
              onClick={() => accept(command)}
              className={`flex w-full items-baseline gap-3 px-3 py-2 text-left text-sm transition ${
                index === highlighted ? 'bg-accent-dim text-fg' : 'text-muted'
              }`}
            >
              <span className="font-mono text-accent">{command.slash}</span>
              {command.argument && (
                <span className="font-mono text-xs text-faint">
                  {command.argument}
                </span>
              )}
              <span className="ml-auto truncate text-xs text-faint">
                {command.title}
              </span>
            </button>
          ))}
        </div>
      )}

      <div className="flex items-end gap-2 rounded-2xl border border-line bg-surface p-2 transition focus-within:border-accent/60">
        <textarea
          ref={box}
          rows={1}
          value={value}
          disabled={disabled}
          placeholder={placeholder}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={onKeyDown}
          className="max-h-60 min-h-[2.25rem] flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] leading-relaxed outline-none placeholder:text-faint disabled:opacity-50"
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled || !value.trim()}
          title="Send  (Enter)"
          className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-accent text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-30"
        >
          <SendIcon className="size-4" />
        </button>
      </div>

      <div className="mt-1.5 flex items-center justify-between px-1 text-[11px] text-faint">
        <span>
          <span className="font-mono">/</span> for commands · Enter to send ·
          Shift+Enter for a new line
        </span>
        <span className="hidden items-center gap-1 sm:flex">
          <CommandIcon className="size-3" />K
        </span>
      </div>
    </div>
  )
}
