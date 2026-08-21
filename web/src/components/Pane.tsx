/**
 * The 262px pane beside the rail.
 *
 * Shows one subject at a time, chosen by the rail. That is the whole point of
 * the split: the old sidebar stacked model, settings, history and tools into
 * one column, so every section was cramped and the panel always scrolled.
 */

import { useState } from 'react'
import type {
  Conversation,
  ModelsResponse,
  ToolsResponse,
  ToolSwitch,
} from '../lib/types'
import type { PaneId } from './Rail'
import {
  AlertIcon,
  ChatIcon,
  CollapseIcon,
  CloudIcon,
  FolderIcon,
  SearchIcon,
  TrashIcon,
} from './Icons'

const TITLES: Record<PaneId, string> = {
  history: 'History',
  tools: 'Tools',
  workspace: 'Workspace',
  settings: 'Settings',
}

interface Props {
  pane: PaneId
  onClose: () => void

  conversations: Conversation[]
  activeConversationId: number | null
  onOpenConversation: (id: number) => void
  onDeleteConversation: (id: number) => void

  models: ModelsResponse | null
  tools: ToolsResponse | null
  toolPending: string | null
  toolError: string | null
  onToggleTool: (id: string, enabled: boolean) => void

  autoRoute: boolean
  onAutoRoute: (value: boolean) => void
  thinking: boolean
  onThinking: (value: boolean) => void
}

export function Pane(props: Props) {
  return (
    <aside className="flex w-[262px] shrink-0 flex-col border-r border-line bg-surface">
      <div className="flex items-center gap-2 px-3.5 pt-3.5 pb-2.5">
        <h2 className="flex-1 text-[13px] tracking-[0.02em]">
          {TITLES[props.pane]}
        </h2>
        <button
          type="button"
          onClick={props.onClose}
          title="Collapse"
          className="grid size-6 place-items-center rounded-md text-fg opacity-50 transition hover:bg-tint hover:opacity-100"
        >
          <CollapseIcon className="size-[15px]" />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3.5 pb-4">
        {props.pane === 'history' && <HistoryPane {...props} />}
        {props.pane === 'tools' && <ToolsPane {...props} />}
        {props.pane === 'workspace' && <WorkspacePane {...props} />}
        {props.pane === 'settings' && <SettingsPane {...props} />}
      </div>
    </aside>
  )
}

function HistoryPane({
  conversations,
  activeConversationId,
  onOpenConversation,
  onDeleteConversation,
}: Props) {
  const [query, setQuery] = useState('')
  const needle = query.trim().toLowerCase()
  const shown = needle
    ? conversations.filter((c) => c.title.toLowerCase().includes(needle))
    : conversations

  return (
    <>
      <div className="mb-2.5 flex h-8 items-center gap-[7px] rounded-md border border-transparent bg-sunken px-[9px] focus-within:border-accent-line">
        <SearchIcon className="size-3.5 shrink-0 text-faint" />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search"
          className="min-w-0 flex-1 bg-transparent text-[12.5px] outline-none placeholder:text-faint"
        />
      </div>

      {shown.length === 0 && (
        <p className="px-1 text-[11.5px] text-faint">
          {conversations.length === 0
            ? 'Nothing saved yet.'
            : `Nothing matches “${query}”.`}
        </p>
      )}

      <div className="space-y-px">
        {shown.map((conversation) => {
          const active = conversation.id === activeConversationId
          return (
            <div
              key={conversation.id}
              className={`group flex items-center gap-1 rounded-md px-2 py-1.5 transition ${
                active ? 'bg-accent-tint' : 'hover:bg-tint'
              }`}
            >
              <button
                type="button"
                onClick={() => onOpenConversation(conversation.id)}
                title={`${conversation.message_count} messages`}
                className="min-w-0 flex-1 truncate text-left text-[12.5px] text-fg opacity-80 group-hover:opacity-100"
              >
                {conversation.title}
              </button>
              <button
                type="button"
                title="Delete"
                onClick={() => onDeleteConversation(conversation.id)}
                className="shrink-0 rounded p-1 text-faint opacity-0 transition group-hover:opacity-100 hover:text-danger"
              >
                <TrashIcon className="size-3.5" />
              </button>
            </div>
          )
        })}
      </div>
    </>
  )
}

function ToolsPane({ tools, toolPending, toolError, onToggleTool }: Props) {
  if (!tools) return <p className="text-[11.5px] text-faint">Loading…</p>

  const parents = tools.switches.filter((entry) => !entry.depends_on)

  return (
    <>
      <p className="mb-2 text-[11.5px] leading-relaxed text-faint">
        {tools.tools.length} offered to the model. Switches last until the API
        restarts; the environment variables make one permanent.
      </p>

      <div className="space-y-0.5">
        {parents.map((entry) => (
          <SwitchRow
            key={entry.id}
            entry={entry}
            sharpEnds={tools.switches.filter((o) => o.depends_on === entry.id)}
            pending={toolPending}
            onToggle={onToggleTool}
          />
        ))}
      </div>

      {toolError && <p className="mt-2 text-[11px] text-danger">{toolError}</p>}
    </>
  )
}

function SwitchRow({
  entry,
  sharpEnds,
  pending,
  onToggle,
}: {
  entry: ToolSwitch
  sharpEnds: ToolSwitch[]
  pending: string | null
  onToggle: (id: string, enabled: boolean) => void
}) {
  const [showRisk, setShowRisk] = useState(false)

  return (
    <div>
      <div className="flex items-center gap-2.5 py-1">
        <Toggle
          on={entry.enabled}
          busy={pending === entry.id}
          disabled={pending !== null}
          label={entry.label}
          onClick={() => onToggle(entry.id, !entry.enabled)}
        />
        <span className="min-w-0 flex-1 truncate text-[12.5px]">
          {entry.label}
          {entry.id === 'ocr' && (
            <span className="ml-1.5 text-[10px] text-faint">
              {pending === 'ocr'
                ? 'starting…'
                : entry.enabled
                  ? 'server up'
                  : 'starts a server'}
            </span>
          )}
        </span>
        {entry.from_env && (
          <span
            title="On because the environment says so — survives a restart."
            className="shrink-0 rounded border border-line px-1 text-[10px] text-faint"
          >
            env
          </span>
        )}
        {entry.risk && (
          <button
            type="button"
            onClick={() => setShowRisk((value) => !value)}
            title="What this allows"
            className="shrink-0 text-faint transition hover:text-warn"
          >
            <AlertIcon className="size-3" />
          </button>
        )}
      </div>

      {showRisk && entry.risk && (
        <p className="mt-1 mb-1.5 ml-9 border-l-2 border-warn/50 pl-2 text-[11px] leading-relaxed text-faint">
          {entry.risk}
        </p>
      )}

      {entry.enabled && sharpEnds.length > 0 && (
        <div className="ml-9 space-y-0.5">
          {sharpEnds.map((child) => (
            <div key={child.id} className="flex items-center gap-2 py-0.5">
              <Toggle
                on={child.enabled}
                busy={pending === child.id}
                disabled={pending !== null}
                label={child.label}
                onClick={() => onToggle(child.id, !child.enabled)}
                sharp
              />
              <span className="min-w-0 flex-1 truncate text-[11px] text-faint">
                {child.label.replace(/^.*— /, '')}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Toggle({
  on,
  busy,
  disabled,
  label,
  onClick,
  sharp = false,
}: {
  on: boolean
  busy: boolean
  disabled: boolean
  label: string
  onClick: () => void
  sharp?: boolean
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className={`relative h-[15px] w-[26px] shrink-0 rounded-full transition disabled:opacity-45 ${
        on ? (sharp ? 'bg-warn' : 'bg-accent') : 'bg-tint-2'
      }`}
    >
      <span
        className={`absolute top-0.5 size-[11px] rounded-full transition-all ${
          on ? 'left-[13px] bg-bg' : 'left-0.5 bg-fg opacity-60'
        } ${busy ? 'animate-breathe' : ''}`}
      />
    </button>
  )
}

function WorkspacePane({ tools, models }: Props) {
  return (
    <>
      <div className="mb-3 flex items-start gap-1.5 text-[11.5px] text-faint">
        <FolderIcon className="mt-px size-3 shrink-0" />
        <span className="min-w-0 font-mono break-all">
          {tools?.workspace ?? '—'}
        </span>
      </div>

      <p className="mb-1.5 text-[10px] tracking-[0.1em] text-faint uppercase">
        Offered to the model
      </p>
      <div className="space-y-1">
        {(tools?.tools ?? []).map((tool) => (
          <div key={tool.name} className="flex gap-2 text-[11.5px]">
            <span className="w-20 shrink-0 text-faint">{tool.category}</span>
            <span className="min-w-0 flex-1 font-mono">{tool.name}</span>
          </div>
        ))}
      </div>

      {models?.available_ram_mb != null && (
        <p className="mt-3 text-[11px] text-faint">
          {models.available_ram_mb.toLocaleString()} MB RAM free
        </p>
      )}
    </>
  )
}

function SettingsPane({
  models,
  autoRoute,
  onAutoRoute,
  thinking,
  onThinking,
}: Props) {
  return (
    <div className="space-y-4">
      <Setting
        label="Auto-route by task"
        hint="Simple prompts to the fast model, involved ones to the strong one. Never routes back down, and asks before sending a turn off this machine."
        on={autoRoute}
        onToggle={onAutoRoute}
      />
      <Setting
        label="Extended thinking"
        hint="The model reasons before answering. Much slower on CPU; the trace is shown but never replayed to the model."
        on={thinking}
        onToggle={onThinking}
      />

      {models && (
        <div className="border-t border-line pt-3 text-[11px] text-faint">
          <p className="mb-1">
            Router: <span className="font-mono">{models.router_fast}</span> →{' '}
            <span className="font-mono">{models.router_strong}</span>
          </p>
          <p className="mb-1">
            One model resident at a time; unloads after{' '}
            {models.idle_timeout_seconds}s idle.
          </p>
          <p className="flex items-center gap-1.5">
            {models.online ? (
              <>
                <CloudIcon className="size-3 shrink-0" />
                Hosted models reachable
              </>
            ) : (
              <>
                <ChatIcon className="size-3 shrink-0" />
                Offline — local models only
              </>
            )}
          </p>
        </div>
      )}
    </div>
  )
}

function Setting({
  label,
  hint,
  on,
  onToggle,
}: {
  label: string
  hint: string
  on: boolean
  onToggle: (value: boolean) => void
}) {
  return (
    <div>
      <div className="flex items-center gap-2.5">
        <Toggle
          on={on}
          busy={false}
          disabled={false}
          label={label}
          onClick={() => onToggle(!on)}
        />
        <span className="text-[12.5px]">{label}</span>
      </div>
      <p className="mt-1 ml-9 text-[11px] leading-relaxed text-faint">{hint}</p>
    </div>
  )
}
