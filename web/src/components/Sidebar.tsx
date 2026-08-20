/**
 * Model, settings, history and tools.
 *
 * Loading a model is a button, never a side effect of selecting one: it costs
 * minutes and evicts whatever else is resident, so it happens when asked for.
 * Selecting a model only decides what the next message will use.
 */

import { useState } from 'react'
import type {
  Conversation,
  ModelsResponse,
  ToolsResponse,
  ToolSwitch,
} from '../lib/types'
import {
  AlertIcon,
  ChipIcon,
  FolderIcon,
  PlusIcon,
  SparkIcon,
  ToolIcon,
  TrashIcon,
} from './Icons'

interface Props {
  models: ModelsResponse | null
  modelsBusyKey: string | null
  modelsError: string | null
  selectedKey: string | null
  onSelectModel: (key: string) => void
  onLoadModel: (key: string) => void
  onUnloadModel: (key: string) => void

  conversations: Conversation[]
  activeConversationId: number | null
  onOpenConversation: (id: number) => void
  onDeleteConversation: (id: number) => void
  onNewConversation: () => void

  tools: ToolsResponse | null
  toolPending: string | null
  toolError: string | null
  onToggleTool: (id: string, enabled: boolean) => void

  autoRoute: boolean
  onAutoRoute: (value: boolean) => void
  thinking: boolean
  onThinking: (value: boolean) => void
}

export function Sidebar(props: Props) {
  const { models, tools } = props
  const selected =
    models?.models.find((model) => model.key === props.selectedKey) ?? null
  const ready = selected?.state === 'ready'

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-line bg-surface">
      <header className="flex items-center gap-2.5 px-4 py-4">
        <div className="flex size-8 items-center justify-center rounded-lg bg-accent text-white">
          <SparkIcon className="size-4" />
        </div>
        <div className="min-w-0">
          <h1 className="truncate text-sm font-semibold">Hakim AI System</h1>
          <p className="truncate text-[11px] text-faint">
            Local models · llama.cpp
          </p>
        </div>
      </header>

      <div className="px-3">
        <button
          type="button"
          onClick={props.onNewConversation}
          className="flex w-full items-center justify-center gap-2 rounded-xl border border-line bg-raised px-3 py-2 text-sm font-medium transition hover:border-accent/50 hover:text-accent"
        >
          <PlusIcon className="size-4" />
          New conversation
        </button>
      </div>

      <div className="mt-4 flex-1 space-y-5 overflow-y-auto px-3 pb-4">
        <Section title="Model">
          <select
            value={props.selectedKey ?? ''}
            onChange={(event) => props.onSelectModel(event.target.value)}
            className="w-full rounded-lg border border-line bg-sunken px-2.5 py-2 text-sm outline-none focus:border-accent/60"
          >
            {models?.models.map((model) => (
              <option key={model.key} value={model.key}>
                {model.label}
                {!model.available ? '  (file missing)' : ''}
                {model.state === 'ready' ? '  ●' : ''}
              </option>
            ))}
          </select>

          {selected && (
            <p className="mt-2 text-[11px] leading-relaxed text-faint">
              {selected.description}
            </p>
          )}

          {selected && !selected.available && (
            <p className="mt-2 rounded-lg border border-danger/30 bg-danger/5 px-2 py-1.5 text-[11px] text-danger">
              Weights not on disk.
            </p>
          )}

          <div className="mt-2.5 flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] ${
                ready
                  ? 'border-ok/40 bg-ok/10 text-ok'
                  : 'border-line bg-sunken text-faint'
              }`}
            >
              <span
                className={`size-1.5 rounded-full ${ready ? 'bg-ok' : 'bg-faint'}`}
              />
              {ready ? 'loaded' : selected?.state ?? 'stopped'}
            </span>
            {selected?.adopted && (
              <span className="text-[11px] text-faint">started elsewhere</span>
            )}
          </div>

          {selected && (
            <button
              type="button"
              disabled={
                props.modelsBusyKey !== null ||
                (!ready && !selected.available)
              }
              onClick={() =>
                ready
                  ? props.onUnloadModel(selected.key)
                  : props.onLoadModel(selected.key)
              }
              className="mt-2 w-full rounded-lg border border-line px-3 py-1.5 text-xs transition hover:border-accent/50 hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
            >
              {props.modelsBusyKey === selected.key
                ? 'Working…'
                : ready
                  ? 'Unload model'
                  : `Load ${selected.label}`}
            </button>
          )}

          {selected?.warning && (
            <p className="mt-2 rounded-lg border border-warn/30 bg-warn/5 px-2 py-1.5 text-[11px] text-warn">
              {selected.warning}
            </p>
          )}
          {props.modelsError && (
            <p className="mt-2 text-[11px] text-danger">{props.modelsError}</p>
          )}
          {models?.available_ram_mb != null && (
            <p className="mt-2 text-[11px] text-faint">
              {models.available_ram_mb.toLocaleString()} MB RAM free
            </p>
          )}
        </Section>

        <Section title="Settings">
          <Toggle
            label="Auto-route by task"
            hint="Simple prompts to the fast model, involved ones to the strong one. Never routes back down."
            checked={props.autoRoute}
            onChange={props.onAutoRoute}
          />
          <Toggle
            label="Extended thinking"
            hint="Qwen3 reasons before answering. Much slower on CPU, and the reasoning is never shown."
            checked={props.thinking}
            onChange={props.onThinking}
          />
        </Section>

        <Section title={`History (${props.conversations.length})`}>
          {props.conversations.length === 0 && (
            <p className="text-[11px] text-faint">Nothing saved yet.</p>
          )}
          <div className="space-y-0.5">
            {props.conversations.map((conversation) => {
              const active = conversation.id === props.activeConversationId
              return (
                <div
                  key={conversation.id}
                  className={`group flex items-center gap-1 rounded-lg px-2 py-1.5 transition ${
                    active ? 'bg-accent-dim' : 'hover:bg-raised'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => props.onOpenConversation(conversation.id)}
                    title={`${conversation.message_count} messages`}
                    className="min-w-0 flex-1 truncate text-left text-[13px] text-muted group-hover:text-fg"
                  >
                    {active && <span className="mr-1 text-accent">●</span>}
                    {conversation.title}
                  </button>
                  <button
                    type="button"
                    title="Delete"
                    onClick={() => props.onDeleteConversation(conversation.id)}
                    className="shrink-0 rounded p-1 text-faint opacity-0 transition group-hover:opacity-100 hover:text-danger"
                  >
                    <TrashIcon className="size-3.5" />
                  </button>
                </div>
              )
            })}
          </div>
        </Section>

        {tools && (
          <Section title={`Tools (${tools.tools.length})`}>
            <div className="space-y-1">
              {groupByCategory(tools).map(([category, names]) => (
                <div key={category} className="flex gap-2 text-[11px]">
                  <span className="w-20 shrink-0 text-faint">{category}</span>
                  <span className="min-w-0 flex-1 font-mono text-muted">
                    {names.join(' · ')}
                  </span>
                </div>
              ))}
            </div>

            <div className="mt-3 space-y-1 border-t border-line-soft pt-3">
              {tools.switches
                .filter((entry) => !entry.depends_on)
                .map((entry) => (
                  <ToolSwitchRow
                    key={entry.id}
                    entry={entry}
                    sharpEnds={tools.switches.filter(
                      (other) => other.depends_on === entry.id,
                    )}
                    pending={props.toolPending}
                    onToggle={props.onToggleTool}
                  />
                ))}
            </div>

            {props.toolError && (
              <p className="mt-2 text-[11px] text-danger">{props.toolError}</p>
            )}

            <p className="mt-3 text-[11px] leading-relaxed text-faint">
              Switches last until the API restarts; the environment variables
              are what make one permanent.
            </p>

            <div className="mt-3 flex items-start gap-1.5 text-[11px] text-faint">
              <FolderIcon className="mt-px size-3 shrink-0" />
              <span className="min-w-0 break-all font-mono">
                {tools.workspace}
              </span>
            </div>
          </Section>
        )}
      </div>

      <footer className="border-t border-line px-4 py-2.5 text-[11px] text-faint">
        Nothing leaves this machine.
      </footer>
    </aside>
  )
}

/**
 * One tool switch, with its sharp end nested underneath.
 *
 * The risk text is shown against the switch rather than hidden behind a
 * tooltip. These flags were environment variables precisely so that turning
 * one on was a considered act; making it a click is the trade, and the least
 * this can do is put the reason where the switch is.
 */
function ToolSwitchRow({
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
  const [open, setOpen] = useState(false)

  return (
    <div>
      <div className="flex items-center gap-2 py-0.5">
        <button
          type="button"
          role="switch"
          aria-checked={entry.enabled}
          aria-label={entry.label}
          disabled={pending !== null}
          onClick={() => onToggle(entry.id, !entry.enabled)}
          className={`relative h-4 w-7 shrink-0 rounded-full transition disabled:opacity-40 ${
            entry.enabled ? 'bg-accent' : 'bg-line'
          }`}
        >
          <span
            className={`absolute top-0.5 size-3 rounded-full bg-white transition-all ${
              entry.enabled ? 'left-3.5' : 'left-0.5'
            }`}
          />
        </button>
        <span className="min-w-0 flex-1 truncate text-[12px] text-muted">
          {entry.label}
        </span>
        {entry.from_env && (
          <span
            title="On because the environment says so, so it survives a restart."
            className="shrink-0 rounded border border-line px-1 text-[10px] text-faint"
          >
            env
          </span>
        )}
        {entry.risk && (
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            title="What this allows"
            className="shrink-0 text-faint transition hover:text-warn"
          >
            <AlertIcon className="size-3" />
          </button>
        )}
      </div>

      {open && entry.risk && (
        <p className="mt-1 mb-1.5 ml-9 border-l-2 border-warn/40 pl-2 text-[11px] leading-relaxed text-faint">
          {entry.risk}
        </p>
      )}

      {sharpEnds.length > 0 && entry.enabled && (
        <div className="ml-9 space-y-0.5">
          {sharpEnds.map((child) => (
            <div key={child.id} className="flex items-center gap-2 py-0.5">
              <button
                type="button"
                role="switch"
                aria-checked={child.enabled}
                aria-label={child.label}
                disabled={pending !== null}
                onClick={() => onToggle(child.id, !child.enabled)}
                className={`relative h-3.5 w-6 shrink-0 rounded-full transition disabled:opacity-40 ${
                  child.enabled ? 'bg-warn' : 'bg-line'
                }`}
              >
                <span
                  className={`absolute top-0.5 size-2.5 rounded-full bg-white transition-all ${
                    child.enabled ? 'left-3' : 'left-0.5'
                  }`}
                />
              </button>
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

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="mb-2 flex items-center gap-1.5 text-[11px] font-medium tracking-wide text-faint uppercase">
        {title.startsWith('Model') ? (
          <ChipIcon className="size-3" />
        ) : title.startsWith('Tools') ? (
          <ToolIcon className="size-3" />
        ) : null}
        {title}
      </h2>
      {children}
    </section>
  )
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string
  hint: string
  checked: boolean
  onChange: (value: boolean) => void
}) {
  const [showHint, setShowHint] = useState(false)
  return (
    <div className="mb-2 last:mb-0">
      <label className="flex cursor-pointer items-center gap-2.5">
        <button
          type="button"
          role="switch"
          aria-checked={checked}
          onClick={() => onChange(!checked)}
          className={`relative h-5 w-9 shrink-0 rounded-full transition ${
            checked ? 'bg-accent' : 'bg-line'
          }`}
        >
          <span
            className={`absolute top-0.5 size-4 rounded-full bg-white transition-all ${
              checked ? 'left-[1.125rem]' : 'left-0.5'
            }`}
          />
        </button>
        <span
          className="flex-1 text-[13px] text-muted"
          onMouseEnter={() => setShowHint(true)}
          onMouseLeave={() => setShowHint(false)}
        >
          {label}
        </span>
      </label>
      {showHint && (
        <p className="mt-1 pl-[3.125rem] text-[11px] leading-relaxed text-faint">
          {hint}
        </p>
      )}
    </div>
  )
}

function groupByCategory(tools: ToolsResponse): [string, string[]][] {
  const groups = new Map<string, string[]>()
  for (const tool of tools.tools) {
    groups.set(tool.category, [...(groups.get(tool.category) ?? []), tool.name])
  }
  return [...groups.entries()]
}
