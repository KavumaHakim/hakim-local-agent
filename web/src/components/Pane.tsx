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
  Model,
  ModelOverride,
  ModelsResponse,
  OcrBackend,
  ToolsResponse,
  ToolSwitch,
  WorkspaceInfo,
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
  modelBusyKey: string | null
  onSetPrimary: (key: string) => void
  onRescanModels: () => void
  onSetModelHidden: (key: string, hidden: boolean) => void
  onOverrideModel: (key: string, values: ModelOverride) => void
  onClearModelOverride: (key: string) => void

  tools: ToolsResponse | null
  toolPending: string | null
  toolError: string | null
  onToggleTool: (id: string, enabled: boolean) => void
  onSetOcrBackend: (backend: OcrBackend) => void

  workspace: WorkspaceInfo | null
  onOpenWorkspacePicker: () => void

  onOpenModelBrowser: () => void

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

function ToolsPane({
  tools,
  toolPending,
  toolError,
  onToggleTool,
  onSetOcrBackend,
}: Props) {
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

      <OcrBackendChooser
        tools={tools}
        pending={toolPending}
        onChoose={onSetOcrBackend}
      />

      {toolError && <p className="mt-2 text-[11px] text-danger">{toolError}</p>}
    </>
  )
}

/**
 * Which reader `ocr_image` uses.
 *
 * A chooser rather than a switch, because the two are different trades rather
 * than more and less of one thing: Tesseract is ~50 MB and under a second but
 * transcribes only; the model understands tables and handwriting and costs
 * ~1.4 GB and ~30 s a page.
 */
function OcrBackendChooser({
  tools,
  pending,
  onChoose,
}: {
  tools: ToolsResponse
  pending: string | null
  onChoose: (backend: OcrBackend) => void
}) {
  const busy = pending === 'ocr-backend'
  const options: { id: OcrBackend; label: string; hint: string }[] = [
    {
      id: 'tesseract',
      label: 'Tesseract',
      hint: '~50 MB, under a second. Transcribes text; no tables or layout.',
    },
    {
      id: 'model',
      label: 'GLM-OCR model',
      hint: '~1.4 GB and ~30 s a page. Understands tables and handwriting.',
    },
  ]

  return (
    <div className="mt-3 border-t border-line pt-3">
      <p className="mb-1.5 text-[12px]">OCR reader</p>
      <div className="space-y-0.5">
        {options.map((option) => {
          const active = tools.ocr_backend === option.id
          return (
            <button
              key={option.id}
              type="button"
              onClick={() => !active && onChoose(option.id)}
              disabled={busy || active}
              className={`w-full rounded border px-2 py-1.5 text-left transition disabled:cursor-default ${
                active ? 'border-accent bg-raised' : 'border-line hover:border-faint'
              }`}
            >
              <span className="text-[12px]">{option.label}</span>
              <span className="mt-0.5 block text-[10.5px] leading-snug text-faint">
                {option.hint}
              </span>
            </button>
          )
        })}
      </div>
      {!tools.ocr_ready && tools.ocr_hint && (
        <p className="mt-1.5 text-[10.5px] leading-snug text-faint">
          {tools.ocr_hint}
        </p>
      )}
    </div>
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

function WorkspacePane({ tools, models, workspace, onOpenWorkspacePicker }: Props) {
  return (
    <>
      <div className="mb-3 rounded-md border border-line bg-sunken px-2.5 py-2">
        <div className="flex items-start gap-1.5">
          <FolderIcon className="mt-0.5 size-3 shrink-0 text-accent" />
          <span
            title={workspace?.path ?? tools?.workspace}
            className="min-w-0 flex-1 font-mono text-[11.5px] break-all text-muted"
          >
            {workspace?.path ?? tools?.workspace ?? '—'}
          </span>
        </div>

        <div className="mt-2 flex items-center gap-1.5">
          <button
            type="button"
            onClick={onOpenWorkspacePicker}
            className="h-6 rounded-md border border-line px-2 text-[11px] transition hover:border-accent-line"
          >
            Change folder
          </button>
          {workspace && !workspace.from_env && (
            <span
              title={`AGENT_WORKSPACE says ${workspace.default}, which a restart returns to.`}
              className="rounded border border-line px-1 text-[10px] text-faint"
            >
              this session
            </span>
          )}
          {workspace?.is_project && (
            <span
              title="The default. Rarely the folder you actually want the agent working in."
              className="rounded border border-line px-1 text-[10px] text-faint"
            >
              project
            </span>
          )}
        </div>

        {workspace && workspace.active_tools.length > 0 && (
          <p className="mt-2 text-[10.5px] leading-relaxed text-faint">
            <span className={workspace.writable ? 'text-warn' : undefined}>
              {workspace.active_tools.join(', ')}
            </span>{' '}
            {workspace.active_tools.length === 1 ? 'acts' : 'act'} on this
            folder.
          </p>
        )}
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
  modelBusyKey,
  onSetPrimary,
  onRescanModels,
  onOpenModelBrowser,
  onSetModelHidden,
  onOverrideModel,
  onClearModelOverride,
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

      <div className="rounded-md border border-line bg-sunken px-2.5 py-2">
        <p className="text-[11.5px]">Need another model?</p>
        <p className="mt-0.5 text-[10.5px] leading-snug text-faint">
          Search Hugging Face and see what each one would need in RAM before
          downloading it.
        </p>
        <button
          type="button"
          onClick={onOpenModelBrowser}
          className="mt-2 h-6 rounded-md border border-line px-2 text-[11px] transition hover:border-accent-line"
        >
          Find a model
        </button>
      </div>

      {models && (
        <ModelSettings
          models={models}
          busyKey={modelBusyKey}
          onSetPrimary={onSetPrimary}
          onRescan={onRescanModels}
          onSetHidden={onSetModelHidden}
          onOverride={onOverrideModel}
          onClearOverride={onClearModelOverride}
        />
      )}

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

/**
 * Choosing the primary model, and picking up files copied into the folder.
 *
 * The primary is deliberately separate from "load this now": choosing costs
 * nothing and loading costs minutes on this hardware, so they are two
 * different decisions and two different buttons.
 */
function ModelSettings({
  models,
  busyKey,
  onSetPrimary,
  onRescan,
  onSetHidden,
  onOverride,
  onClearOverride,
}: {
  models: ModelsResponse
  busyKey: string | null
  onSetPrimary: (key: string) => void
  onRescan: () => void
  onSetHidden: (key: string, hidden: boolean) => void
  onOverride: (key: string, values: ModelOverride) => void
  onClearOverride: (key: string) => void
}) {
  // Only chat models can be the primary. A vision backend runs alongside one
  // and cannot drive the agent loop, so offering it here would be a trap.
  const [editing, setEditing] = useState<string | null>(null)
  const chat = models.models.filter((model) => model.role === 'chat')
  const choices = chat.filter((model) => !model.hidden)
  const hidden = chat.filter((model) => model.hidden)
  const rescanning = busyKey === '__rescan__'

  return (
    <div className="border-t border-line pt-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[12.5px]">Primary model</span>
        <button
          type="button"
          onClick={onRescan}
          disabled={rescanning}
          className="rounded border border-line px-1.5 py-0.5 text-[10.5px] text-faint hover:text-ink disabled:opacity-50"
          title={`Re-read ${models.models_dir}`}
        >
          {rescanning ? 'Scanning…' : 'Rescan folder'}
        </button>
      </div>

      {models.setup_required && (
        <p className="mb-2 rounded border border-line bg-raised px-2 py-1.5 text-[11px] text-faint">
          Several models are available and none has been chosen. Pick the one
          everything should default to.
        </p>
      )}

      <div className="space-y-1">
        {choices.map((model) => {
          const primary = model.key === models.default_key
          return (
            <div
              key={model.key}
              className={`rounded border transition ${
                primary
                  ? 'border-accent bg-raised'
                  : 'border-line hover:border-faint'
              }`}
            >
            <div className="flex items-start gap-1 px-2 py-1.5">
            <button
              type="button"
              onClick={() => !primary && onSetPrimary(model.key)}
              disabled={busyKey === model.key || !model.usable}
              className="min-w-0 flex-1 text-left disabled:opacity-50"
            >
              <span className="flex items-center gap-1.5">
                <span className="truncate text-[12px]">{model.label}</span>
                {primary && (
                  <span className="shrink-0 text-[10px] text-accent">
                    primary
                  </span>
                )}
                {model.discovered && (
                  <span
                    className="shrink-0 text-[10px] text-faint"
                    title="Found in the models folder rather than declared in models.json"
                  >
                    found on disk
                  </span>
                )}
                {model.customised && (
                  <span className="shrink-0 text-[10px] text-faint">
                    retuned
                  </span>
                )}
              </span>
              <span className="mt-0.5 block text-[10.5px] text-faint">
                {model.remote
                  ? model.provider
                  : `${model.file_mb.toLocaleString()} MB · ${model.context.toLocaleString()} ctx · needs ${model.min_free_mb.toLocaleString()} MB free`}
              </span>
              {model.notes.map((note) => (
                <span
                  key={note}
                  className="mt-0.5 block text-[10px] leading-snug text-faint"
                >
                  {note}
                </span>
              ))}
            </button>
            {!model.remote && (
              <button
                type="button"
                onClick={() =>
                  setEditing(editing === model.key ? null : model.key)
                }
                disabled={busyKey === model.key}
                className="shrink-0 rounded px-1 text-[10.5px] text-faint hover:text-ink disabled:opacity-50"
                title="Context, threads and RAM for this model"
              >
                {editing === model.key ? 'close' : 'tune'}
              </button>
            )}
            {!primary && (
              <button
                type="button"
                onClick={() => onSetHidden(model.key, true)}
                disabled={busyKey === model.key}
                className="shrink-0 rounded px-1 text-[10.5px] text-faint hover:text-ink disabled:opacity-50"
                title="Stop offering this model. The file is not deleted."
              >
                hide
              </button>
            )}
            </div>
            {editing === model.key && (
              <ModelTuner
                model={model}
                busy={busyKey === model.key}
                onApply={(values) => {
                  onOverride(model.key, values)
                  setEditing(null)
                }}
                onReset={() => {
                  onClearOverride(model.key)
                  setEditing(null)
                }}
              />
            )}
            </div>
          )
        })}
      </div>

      {hidden.length > 0 && (
        <div className="mt-2 space-y-1">
          <p className="text-[10.5px] text-faint">Hidden</p>
          {hidden.map((model) => (
            <div
              key={model.key}
              className="flex items-center gap-1 text-[11px] text-faint"
            >
              <span className="min-w-0 flex-1 truncate">{model.label}</span>
              <button
                type="button"
                onClick={() => onSetHidden(model.key, false)}
                disabled={busyKey === model.key}
                className="shrink-0 rounded px-1 hover:text-ink disabled:opacity-50"
              >
                show
              </button>
            </div>
          ))}
        </div>
      )}

      <p className="mt-2 text-[10.5px] leading-snug text-faint">
        Copy a <span className="font-mono">.gguf</span> into{' '}
        <span className="font-mono break-all">{models.models_dir}</span> and
        press Rescan. Context and RAM are read from the file&apos;s own header.
      </p>
    </div>
  )
}

/**
 * Context, threads and the RAM threshold for one model.
 *
 * Context gets live feedback because it is the field that can quietly make a
 * model unstartable: the KV cache grows linearly with it, and on an 8 GB
 * machine the difference between 4,096 and 32,768 is gigabytes. The cost per
 * token is derived from what the server already reported for the current
 * setting, so the estimate is the model's real arithmetic rather than a guess.
 *
 * Blank means "leave it alone" rather than "set it to zero" - only fields that
 * were actually typed into are sent.
 */
function ModelTuner({
  model,
  busy,
  onApply,
  onReset,
}: {
  model: Model
  busy: boolean
  onApply: (values: ModelOverride) => void
  onReset: () => void
}) {
  const [label, setLabel] = useState(model.label)
  const [context, setContext] = useState(String(model.context))
  const [threads, setThreads] = useState(String(model.threads))
  const [minFree, setMinFree] = useState(String(model.min_free_mb))

  // Bytes of KV cache per token, from the figures already measured for the
  // context this model is running at now.
  const perToken =
    model.context > 0 && model.kv_cache_mb > 0
      ? (model.kv_cache_mb * 1024 * 1024) / model.context
      : 0
  const wanted = Number(context)
  const projected =
    perToken > 0 && Number.isFinite(wanted) && wanted > 0
      ? Math.round((perToken * wanted) / (1024 * 1024))
      : 0
  const overTrained =
    model.training_context > 0 && wanted > model.training_context

  function apply() {
    const values: ModelOverride = {}
    if (label.trim() && label.trim() !== model.label) values.label = label.trim()
    if (context && Number(context) !== model.context)
      values.context = Number(context)
    if (threads && Number(threads) !== model.threads)
      values.threads = Number(threads)
    if (minFree && Number(minFree) !== model.min_free_mb)
      values.min_free_mb = Number(minFree)
    onApply(values)
  }

  return (
    <div className="border-t border-line px-2 py-2">
      <Field label="Name">
        <input
          value={label}
          onChange={(event) => setLabel(event.target.value)}
          className="w-full rounded border border-line bg-base px-1.5 py-0.5 text-[11px]"
        />
      </Field>

      <div className="mt-1.5 grid grid-cols-3 gap-1.5">
        <Field label="Context">
          <input
            type="number"
            value={context}
            min={512}
            step={512}
            onChange={(event) => setContext(event.target.value)}
            className="w-full rounded border border-line bg-base px-1.5 py-0.5 text-[11px]"
          />
        </Field>
        <Field label="Threads">
          <input
            type="number"
            value={threads}
            min={1}
            max={64}
            onChange={(event) => setThreads(event.target.value)}
            className="w-full rounded border border-line bg-base px-1.5 py-0.5 text-[11px]"
          />
        </Field>
        <Field label="Needs MB">
          <input
            type="number"
            value={minFree}
            min={0}
            step={50}
            onChange={(event) => setMinFree(event.target.value)}
            className="w-full rounded border border-line bg-base px-1.5 py-0.5 text-[11px]"
          />
        </Field>
      </div>

      <div className="mt-1.5 space-y-0.5 text-[10px] leading-snug text-faint">
        {projected > 0 && (
          <p>
            KV cache at {wanted.toLocaleString()} tokens:{' '}
            <span className={projected > 1024 ? 'text-danger' : ''}>
              ~{projected.toLocaleString()} MB
            </span>
            {model.kv_cache_mb > 0 && (
              <> (now ~{model.kv_cache_mb.toLocaleString()} MB)</>
            )}
          </p>
        )}
        {overTrained && (
          <p className="text-danger">
            Above the {model.training_context.toLocaleString()} tokens this
            model was trained for.
          </p>
        )}
        <p>Applies the next time this model starts.</p>
      </div>

      <div className="mt-2 flex items-center gap-1.5">
        <button
          type="button"
          onClick={apply}
          disabled={busy}
          className="rounded border border-accent px-2 py-0.5 text-[11px] disabled:opacity-50"
        >
          {busy ? 'Saving…' : 'Apply'}
        </button>
        {model.customised && (
          <button
            type="button"
            onClick={onReset}
            disabled={busy}
            className="rounded border border-line px-2 py-0.5 text-[11px] text-faint hover:text-ink disabled:opacity-50"
            title="Back to the registry or discovered values"
          >
            Reset
          </button>
        )}
      </div>
    </div>
  )
}

function Field({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <label className="block">
      <span className="mb-0.5 block text-[10px] text-faint">{label}</span>
      {children}
    </label>
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
