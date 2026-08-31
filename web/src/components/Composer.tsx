/**
 * The message box.
 *
 * The redesign moves the model picker, the tool count and the thinking toggle
 * *into* the composer, as a control row under the textarea. They belong to the
 * turn you are about to send, not to a settings panel — and putting them here
 * is what let the sidebar collapse to one subject at a time.
 *
 * Attachments sit inside the box, above the text, so an image and the question
 * about it read as one message.
 */

import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { matchCommands, type CommandSpec } from '../lib/commands'
import type { Model, UploadResult, WorkspaceInfo } from '../lib/types'
import { basename } from '../lib/paths'
import {
  ChevronIcon,
  CloudIcon,
  FolderIcon,
  ImageIcon,
  PaperclipIcon,
  SendIcon,
  SparkIcon,
  ToolIcon,
} from './Icons'

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

  model: Model | null
  onOpenModels: () => void
  toolCount: number
  onOpenTools: () => void
  thinking: boolean
  onToggleThinking: () => void
  workspace: WorkspaceInfo | null
  onOpenWorkspace: () => void
  /** Turns already waiting. Shown so a queue cannot build up unnoticed. */
  queued: number
  /** True when nothing more may be queued until one of them finishes. */
  atCapacity: boolean
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
  model,
  onOpenModels,
  toolCount,
  onOpenTools,
  thinking,
  onToggleThinking,
  workspace,
  onOpenWorkspace,
  queued,
  atCapacity,
}: Props) {
  const box = useRef<HTMLTextAreaElement>(null)
  const picker = useRef<HTMLInputElement>(null)
  const [highlighted, setHighlighted] = useState(0)
  const [dragging, setDragging] = useState(false)

  const suggestions = matchCommands(value)
  const showing = suggestions.length > 0 && !value.includes('\n')
  // Drafting stays possible at capacity - it is sending that has to wait, and
  // taking the textarea away would lose what someone was in the middle of
  // typing.
  const sendable =
    (Boolean(value.trim()) || attachments.length > 0) && !atCapacity

  useLayoutEffect(() => {
    const element = box.current
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 180)}px`
  }, [value])

  useEffect(() => setHighlighted(0), [value])

  function accept(command: CommandSpec) {
    onChange(command.argument ? `${command.slash} ` : command.slash)
    box.current?.focus()
  }

  function submit() {
    if (disabled || uploading || !sendable) return
    onSubmit(value.trim())
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
        <div className="absolute bottom-full left-0 z-20 mb-2 w-full overflow-hidden rounded-lg bg-surface shadow-[var(--shadow-lg)]">
          {suggestions.map((command, index) => (
            <button
              key={command.id}
              type="button"
              onMouseEnter={() => setHighlighted(index)}
              onClick={() => accept(command)}
              className={`flex w-full items-baseline gap-2.5 px-[9px] py-[7px] text-left transition ${
                index === highlighted ? 'bg-accent-tint' : ''
              }`}
            >
              <span className="font-mono text-[13px] text-accent">
                {command.slash}
              </span>
              {command.argument && (
                <span className="font-mono text-[11px] text-faint">
                  {command.argument}
                </span>
              )}
              <span className="ml-auto truncate text-[12.5px] text-muted">
                {command.title}
              </span>
            </button>
          ))}
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
        className={`flex flex-col rounded-lg border bg-surface transition ${
          dragging
            ? 'border-accent bg-accent-tint/30'
            : 'border-line focus-within:border-accent-line'
        }`}
      >
        {attachments.map((file) => (
          <div
            key={file.path}
            className="mx-2 mt-2 flex items-center gap-2 rounded-md bg-tint px-[9px] py-[5px] text-xs"
          >
            <ImageIcon className="size-[13px] shrink-0 text-accent" />
            <span className="min-w-0 flex-1 truncate">{file.name}</span>
            <span className="shrink-0 text-faint">
              {Math.max(1, Math.round(file.size / 1024)).toLocaleString()} KB
            </span>
            {!file.ocr_ready && (
              <span
                title={file.hint}
                className="shrink-0 rounded-sm border border-warn px-[5px] text-[10px] text-warn"
              >
                OCR off
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

        {(uploading || uploadError) && (
          <p
            className={`mx-2 mt-2 text-[11px] ${uploadError ? 'text-danger' : 'text-faint'}`}
          >
            {uploadError ?? 'Uploading…'}
          </p>
        )}

        <textarea
          ref={box}
          rows={2}
          value={value}
          disabled={disabled}
          placeholder={placeholder}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={onKeyDown}
          className="mx-1 mt-1 max-h-[180px] min-h-[52px] resize-none bg-transparent px-3 py-2.5 text-[15px] leading-[1.55] outline-none placeholder:text-faint disabled:opacity-50"
        />

        <div className="flex items-center gap-1.5 px-2 pt-1.5 pb-2">
          <Pill onClick={onOpenModels} title="Choose the model">
            <span
              className={`size-[5px] shrink-0 rounded-full ${
                model?.state === 'ready' || model?.remote
                  ? 'bg-ok'
                  : 'bg-neutral-600'
              }`}
              style={
                model?.state === 'ready' || model?.remote
                  ? undefined
                  : { background: 'var(--neutral-600)' }
              }
            />
            {model?.remote && <CloudIcon className="size-3 text-warn" />}
            <span className="max-w-[120px] truncate">
              {model?.label ?? 'No model'}
            </span>
            <ChevronIcon className="size-3 rotate-90 opacity-40" />
          </Pill>

          <Pill onClick={onOpenTools} title="Tools offered to the model">
            <ToolIcon className="size-[13px] opacity-60" />
            {toolCount} tools
            <ChevronIcon className="size-3 rotate-90 opacity-40" />
          </Pill>

          {/* The folder this turn's file tools may reach. It belongs here for
              the same reason the model picker does: it is a property of the
              message you are about to send, not of a settings page. */}
          <Pill
            onClick={onOpenWorkspace}
            title={
              workspace
                ? `Working in ${workspace.path}${
                    workspace.writable
                      ? ' — files there can be changed'
                      : ' — read-only'
                  }. Click to change.`
                : 'Choose the folder the agent may work in'
            }
          >
            <FolderIcon
              className={`size-[13px] ${
                workspace?.writable ? 'text-warn' : 'opacity-60'
              }`}
            />
            <span className="max-w-[110px] truncate">
              {workspace ? basename(workspace.path) : 'Workspace'}
            </span>
            <ChevronIcon className="size-3 rotate-90 opacity-40" />
          </Pill>

          <Pill
            onClick={onToggleThinking}
            title="Reason before answering. Slower."
            active={thinking}
          >
            <SparkIcon className="size-[13px]" />
            Thinking
          </Pill>

          <input
            ref={picker}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={(event) => {
              if (event.target.files?.length) onAttach(event.target.files)
              event.target.value = ''
            }}
          />
          <button
            type="button"
            onClick={() => picker.current?.click()}
            disabled={disabled || uploading}
            title="Attach an image to read with OCR"
            className="grid size-7 shrink-0 place-items-center rounded-md border border-line text-fg opacity-65 transition hover:border-accent-line hover:opacity-100 disabled:cursor-not-allowed disabled:opacity-30"
          >
            <PaperclipIcon className="size-3.5" />
          </button>

          {/* Outlined, not filled: the system uses the accent as a line, and
              a solid accent button would be the flood it avoids. */}
          <button
            type="button"
            onClick={submit}
            disabled={disabled || uploading || !sendable}
            title="Send  (Enter)"
            className="ml-auto grid size-8 shrink-0 place-items-center rounded-[9px] border border-accent text-accent transition hover:bg-accent-tint disabled:cursor-not-allowed disabled:opacity-30"
          >
            <SendIcon className="size-[15px]" />
          </button>
        </div>
      </div>

      <p className="mt-[7px] px-0.5 text-[11px] text-faint">
        {atCapacity ? (
          <span className="text-warn">
            That is as many as this machine will usefully hold. One has to
            finish before the next can be asked.
          </span>
        ) : queued > 0 ? (
          <>
            <span className="text-accent-soft">
              {queued} {queued === 1 ? 'question is' : 'questions are'} waiting
            </span>{' '}
            · they run one at a time, in the order you asked
          </>
        ) : (
          <>
            Enter to send · Shift+Enter for a newline · drop an image to read it
            {model && !model.remote && ' · this turn stays on your machine'}
          </>
        )}
      </p>
    </div>
  )
}

function Pill({
  onClick,
  title,
  active = false,
  children,
}: {
  onClick: () => void
  title: string
  active?: boolean
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className={`flex h-7 shrink-0 items-center gap-1.5 rounded-md border px-[9px] text-xs transition ${
        active
          ? 'border-accent-line bg-accent-tint text-accent-soft'
          : 'border-line text-fg hover:border-accent-line'
      }`}
    >
      {children}
    </button>
  )
}
