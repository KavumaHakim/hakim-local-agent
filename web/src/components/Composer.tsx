/**
 * The message box, with inline slash-command completion.
 *
 * Streamlit could not do this: `st.chat_input` takes no children, so the old
 * build injected a dropdown through a same-origin iframe reaching into the
 * page's DOM. Here it is just a component.
 */

import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { matchCommands, type CommandSpec } from '../lib/commands'
import { CommandIcon, ImageIcon, PaperclipIcon, SendIcon } from './Icons'
import type { UploadResult } from '../lib/types'

interface Props {
  value: string
  onChange: (value: string) => void
  onSubmit: (value: string) => void
  disabled: boolean
  placeholder: string
  attachments: UploadResult[]
  onAttach: (files: FileList | File[]) => void
  onRemoveAttachment: (path: string) => void
  uploading: boolean
  uploadError: string | null
}

export function Composer({
  value,
  onChange,
  onSubmit,
  disabled,
  placeholder,
  attachments,
  onAttach,
  onRemoveAttachment,
  uploading,
  uploadError,
}: Props) {
  const box = useRef<HTMLTextAreaElement>(null)
  const picker = useRef<HTMLInputElement>(null)
  const [highlighted, setHighlighted] = useState(0)
  const [dragging, setDragging] = useState(false)

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
    if (disabled || uploading) return
    const text = value.trim()
    // An attachment on its own is a turn: dropping an image and pressing send
    // is a reasonable way to ask "what does this say".
    if (!text && attachments.length === 0) return
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

      {(attachments.length > 0 || uploading || uploadError) && (
        <div className="mb-2 space-y-1.5">
          {attachments.map((file) => (
            <div
              key={file.path}
              className="flex items-center gap-2 rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs"
            >
              <ImageIcon className="size-3.5 shrink-0 text-accent" />
              <span className="min-w-0 flex-1 truncate">{file.name}</span>
              <span className="shrink-0 text-faint">
                {Math.max(1, Math.round(file.size / 1024)).toLocaleString()} KB
              </span>
              {/* The upload succeeded; whether anything can read it is a
                  separate question, and one worth answering before send. */}
              {!file.ocr_ready && (
                <span
                  title={file.hint}
                  className="shrink-0 rounded border border-warn/40 px-1 text-[10px] text-warn"
                >
                  OCR not ready
                </span>
              )}
              <button
                type="button"
                onClick={() => onRemoveAttachment(file.path)}
                title="Remove"
                className="shrink-0 text-faint transition hover:text-danger"
              >
                ✕
              </button>
            </div>
          ))}

          {attachments.some((file) => !file.ocr_ready) && (
            <p className="px-1 text-[11px] leading-relaxed text-warn">
              {attachments.find((file) => !file.ocr_ready)?.hint}
            </p>
          )}

          {uploading && (
            <p className="px-1 text-[11px] text-faint">Uploading…</p>
          )}
          {uploadError && (
            <p className="px-1 text-[11px] text-danger">{uploadError}</p>
          )}
        </div>
      )}

      <div
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setDragging(false)
          if (event.dataTransfer.files.length) onAttach(event.dataTransfer.files)
        }}
        className={`flex items-end gap-2 rounded-2xl border bg-surface p-2 transition ${
          dragging
            ? 'border-accent bg-accent-dim/30'
            : 'border-line focus-within:border-accent/60'
        }`}
      >
        <input
          ref={picker}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(event) => {
            if (event.target.files?.length) onAttach(event.target.files)
            // Cleared so choosing the same file twice still fires a change.
            event.target.value = ''
          }}
        />
        <button
          type="button"
          onClick={() => picker.current?.click()}
          disabled={disabled || uploading}
          title="Attach an image to read with OCR"
          className="flex size-9 shrink-0 items-center justify-center rounded-xl border border-line text-faint transition hover:border-accent/50 hover:text-accent disabled:cursor-not-allowed disabled:opacity-30"
        >
          <PaperclipIcon className="size-4" />
        </button>
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
          disabled={disabled || uploading || (!value.trim() && attachments.length === 0)}
          title="Send  (Enter)"
          className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-accent text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-30"
        >
          <SendIcon className="size-4" />
        </button>
      </div>

      <div className="mt-1.5 flex items-center justify-between px-1 text-[11px] text-faint">
        <span>
          <span className="font-mono">/</span> for commands · Enter to send ·
          drop an image to read it
        </span>
        <span className="hidden items-center gap-1 sm:flex">
          <CommandIcon className="size-3" />K
        </span>
      </div>
    </div>
  )
}
