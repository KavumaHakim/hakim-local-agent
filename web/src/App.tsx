/**
 * The application shell.
 *
 * Layout is the rail, one contextual pane, and the conversation. The rail
 * chooses what the pane shows; clicking the active rail button collapses it.
 * Everything that belongs to the *next turn* — model, tools, thinking — lives
 * in the composer instead, because that is the thing it acts on.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { CommandPalette } from './components/CommandPalette'
import { Composer } from './components/Composer'
import { EmptyState, MessageView } from './components/Messages'
import { Pane } from './components/Pane'
import { Rail, type PaneId } from './components/Rail'
import { RemoteConsent } from './components/RemoteConsent'
import { TurnStatus } from './components/TurnStatus'
import { WorkspacePicker } from './components/WorkspacePicker'
import { CommandIcon, ExpandIcon } from './components/Icons'
import { useChat } from './hooks/useChat'
import {
  useConversations,
  useHotkey,
  useModels,
  useTools,
  useWorkspace,
} from './hooks/useResources'
import { COMMANDS, parseCommand, type CommandId } from './lib/commands'
import { api } from './lib/api'
import type { UploadResult } from './lib/types'

type Theme = 'dark' | 'light'

export default function App() {
  const [draft, setDraft] = useState('')
  const [modelKey, setModelKey] = useState<string | null>(null)
  const [autoRoute, setAutoRoute] = useState(false)
  const [thinking, setThinking] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [note, setNote] = useState<string | null>(null)
  const [attachments, setAttachments] = useState<UploadResult[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const [pane, setPane] = useState<PaneId>('history')
  const [paneOpen, setPaneOpen] = useState(true)
  const [pickingWorkspace, setPickingWorkspace] = useState(false)
  const [theme, setTheme] = useState<Theme>(
    () =>
      (document.documentElement.dataset.theme as Theme | undefined) ?? 'dark',
  )

  const models = useModels()
  const tools = useTools()
  const conversations = useConversations()
  const workspace = useWorkspace()

  /**
   * Move the workspace, then re-read the tool roster.
   *
   * The roster carries the path too, and the tool descriptions are built
   * against the config the registry saw — so leaving it alone would have the
   * Tools panel quietly disagreeing with the Workspace one about where the
   * agent is working.
   */
  const chooseWorkspace = useCallback(
    async (path: string) => {
      const moved = await workspace.choose(path)
      if (moved) void tools.refresh()
      return moved
    },
    [workspace, tools],
  )

  /**
   * And the same in the other direction.
   *
   * Which tools act on the workspace is part of what the workspace panel says
   * about the folder, so flipping a switch has to re-read it - otherwise the
   * picker would go on claiming the agent can only read there while the write
   * tool it just gained sat one panel away.
   */
  const toggleTool = useCallback(
    async (id: string, enabled: boolean) => {
      await tools.toggle(id, enabled)
      void workspace.refresh()
    },
    [tools, workspace],
  )

  const chat = useChat({
    modelKey,
    enableThinking: thinking,
    autoRoute,
    onConversationChanged: () => void conversations.refresh(),
  })

  useEffect(() => {
    if (modelKey === null && models.models) {
      setModelKey(models.models.active_key ?? models.models.default_key)
    }
  }, [models.models, modelKey])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  useHotkey('k', () => setPaletteOpen((open) => !open), { meta: true })

  const bottom = useRef<HTMLDivElement>(null)
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [chat.messages.length, chat.turn.phase, chat.turn.text])

  const newConversation = useCallback(() => {
    void chat.openConversation(null)
    setNote(null)
  }, [chat])

  /** Rail click: open that pane, or collapse it if it is already showing. */
  const selectPane = useCallback(
    (next: PaneId) => {
      if (paneOpen && pane === next) setPaneOpen(false)
      else {
        setPane(next)
        setPaneOpen(true)
      }
    },
    [pane, paneOpen],
  )

  const runCommand = useCallback(
    (id: CommandId, argument = '') => {
      switch (id) {
        case 'new':
        case 'clear':
          newConversation()
          break
        case 'model':
          if (argument) setModelKey(argument)
          else setNote('Give a model key, e.g. /model tiny')
          break
        case 'load':
          void models.load(argument || modelKey || '')
          break
        case 'unload':
          if (modelKey) void models.unload(modelKey)
          break
        case 'auto':
          setAutoRoute((value) => {
            setNote(`Automatic routing ${!value ? 'on' : 'off'}.`)
            return !value
          })
          break
        case 'thinking':
          setThinking((value) => !value)
          break
        case 'models':
          setPane('settings')
          setPaneOpen(true)
          break
        case 'tools':
          setPane('tools')
          setPaneOpen(true)
          break
        case 'workspace':
          // With a path it moves straight there, because someone who pasted
          // one already knows where they are going; without, it opens the
          // picker, because nobody types a Windows path from memory.
          if (argument) {
            void chooseWorkspace(argument).then((moved) => {
              if (moved) setNote(null)
              else setPickingWorkspace(true)
            })
          } else {
            setPickingWorkspace(true)
          }
          break
        case 'help':
          setNote(
            COMMANDS.map(
              (command) =>
                `${command.slash}${command.argument ? ` ${command.argument}` : ''} — ${command.title}`,
            ).join('\n'),
          )
          break
      }
    },
    [models, modelKey, newConversation, chooseWorkspace],
  )

  const submit = useCallback(
    (text: string) => {
      const command = parseCommand(text)
      if (command) {
        runCommand(command.spec.id, command.argument)
        setDraft('')
        return
      }
      setNote(null)
      setDraft('')
      const paths = attachments.map((file) => file.path)
      setAttachments([])
      void chat.send(text, undefined, false, paths)
    },
    [chat, runCommand, attachments],
  )

  const attach = useCallback(async (files: FileList | File[]) => {
    setUploading(true)
    setUploadError(null)
    try {
      for (const file of Array.from(files)) {
        const stored = await api.upload(file)
        setAttachments((current) => [...current, stored])
      }
    } catch (failure) {
      setUploadError(failure instanceof Error ? failure.message : String(failure))
    } finally {
      setUploading(false)
    }
  }, [])

  /** Re-ask the last question, which is the one before the last answer. */
  const retryLast = useCallback(() => {
    const lastUser = [...chat.messages]
      .reverse()
      .find((message) => message.role === 'user')
    if (lastUser) void chat.send(lastUser.content)
  }, [chat])

  const selectedModel =
    models.models?.models.find((model) => model.key === modelKey) ?? null
  const empty = chat.messages.length === 0 && chat.turn.phase === 'idle'
  const title =
    conversations.conversations.find(
      (conversation) => conversation.id === chat.conversationId,
    )?.title ?? 'New conversation'
  const lastIndex = chat.messages.length - 1

  return (
    <div className="flex h-full">
      <Rail
        active={pane}
        open={paneOpen}
        onSelect={selectPane}
        onNewConversation={newConversation}
        theme={theme}
        onToggleTheme={() =>
          setTheme((current) => (current === 'dark' ? 'light' : 'dark'))
        }
        toolsWarning={Boolean(
          tools.data?.switches.some((entry) => entry.enabled && entry.risk),
        )}
      />

      {paneOpen && (
        <Pane
          pane={pane}
          onClose={() => setPaneOpen(false)}
          conversations={conversations.conversations}
          activeConversationId={chat.conversationId}
          onOpenConversation={(id) => void chat.openConversation(id)}
          onDeleteConversation={async (id) => {
            await conversations.remove(id)
            if (id === chat.conversationId) newConversation()
          }}
          models={models.models}
          modelBusyKey={models.busyKey}
          onSetPrimary={(key) => void models.setPrimary(key)}
          onRescanModels={() => void models.rescan()}
          onSetModelHidden={(key, hidden) =>
            void models.setHidden(key, hidden)
          }
          onOverrideModel={(key, values) => void models.override(key, values)}
          onClearModelOverride={(key) => void models.clearOverride(key)}
          tools={tools.data}
          toolPending={tools.pending}
          onSetOcrBackend={(backend) => void tools.setOcrBackend(backend)}
          toolError={tools.error}
          onToggleTool={(id, enabled) => void toggleTool(id, enabled)}
          workspace={workspace.workspace}
          onOpenWorkspacePicker={() => setPickingWorkspace(true)}
          autoRoute={autoRoute}
          onAutoRoute={setAutoRoute}
          thinking={thinking}
          onThinking={setThinking}
        />
      )}

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-2.5 border-b border-line px-4">
          {!paneOpen && (
            <button
              type="button"
              onClick={() => setPaneOpen(true)}
              title="Open panel"
              className="grid size-7 shrink-0 place-items-center rounded-md text-fg opacity-60 transition hover:bg-tint hover:opacity-100"
            >
              <ExpandIcon className="size-4" />
            </button>
          )}

          <h1 className="truncate text-[13px]">{title}</h1>

          {chat.busy && (
            <span className="flex shrink-0 items-center gap-1.5 rounded-full bg-accent-tint px-2 py-0.5 text-[11px] text-accent-soft">
              <span className="size-[5px] animate-breathe rounded-full bg-accent" />
              {chat.turn.phase === 'queued'
                ? 'queued'
                : chat.turn.phase === 'loading'
                  ? 'loading'
                  : 'streaming'}
            </span>
          )}

          <button
            type="button"
            onClick={() => setPaletteOpen(true)}
            className="ml-auto flex h-[26px] shrink-0 items-center gap-1 rounded-md border border-line px-2 text-[11px] text-fg opacity-60 transition hover:border-accent-line hover:opacity-100"
          >
            <CommandIcon className="size-3" />K
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {empty ? (
            <EmptyState onPick={(text) => setDraft(text)} />
          ) : (
            <div className="mx-auto flex max-w-[760px] flex-col gap-[26px] px-6 pt-7 pb-2">
              {chat.messages.map((message, index) => (
                <MessageView
                  key={message.id}
                  message={message}
                  onRetry={
                    index === lastIndex && message.role === 'assistant'
                      ? retryLast
                      : undefined
                  }
                />
              ))}
              <TurnStatus
                turn={chat.turn}
                onEscalate={() => {
                  const strong = models.models?.router_strong
                  if (!strong) return
                  setModelKey(strong)
                  const last = [...chat.messages]
                    .reverse()
                    .find((message) => message.role === 'user')
                  chat.dismissError()
                  if (last) void chat.send(last.content, strong)
                }}
                onDismiss={chat.dismissError}
                onStop={() => void chat.stop()}
                stopping={chat.stopping}
              />
              <div ref={bottom} />
            </div>
          )}
        </div>

        <div className="px-6 pb-4">
          <div className="mx-auto max-w-[760px]">
            {note && (
              <div className="mb-2 flex items-start gap-2 rounded-md border border-line bg-surface px-3 py-2">
                <pre className="min-w-0 flex-1 overflow-x-auto font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-muted">
                  {note}
                </pre>
                <button
                  type="button"
                  onClick={() => setNote(null)}
                  className="shrink-0 text-faint transition hover:text-fg"
                >
                  ✕
                </button>
              </div>
            )}
            <Composer
              value={draft}
              onChange={setDraft}
              onSubmit={submit}
              disabled={chat.busy}
              placeholder={
                chat.busy ? 'Waiting for the current turn…' : 'Message Hakim…'
              }
              attachments={attachments}
              onAttach={attach}
              onRemoveAttachment={(path) =>
                setAttachments((current) =>
                  current.filter((file) => file.path !== path),
                )
              }
              uploading={uploading}
              uploadError={uploadError}
              model={selectedModel}
              onOpenModels={() => setPaletteOpen(true)}
              toolCount={tools.data?.tools.length ?? 0}
              onOpenTools={() => selectPane('tools')}
              thinking={thinking}
              onToggleThinking={() => setThinking((value) => !value)}
              workspace={workspace.workspace}
              onOpenWorkspace={() => setPickingWorkspace(true)}
            />
          </div>
        </div>
      </main>

      {pickingWorkspace && workspace.workspace && (
        <WorkspacePicker
          workspace={workspace.workspace}
          pending={workspace.pending}
          error={workspace.error}
          onChoose={chooseWorkspace}
          onReset={() => {
            void workspace.reset().then(() => tools.refresh())
          }}
          onClose={() => {
            workspace.clearError()
            setPickingWorkspace(false)
          }}
          onClearError={workspace.clearError}
        />
      )}

      {chat.consent && (
        <RemoteConsent
          request={chat.consent.request}
          localLabel={
            models.models?.models.find(
              (model) => model.key === models.models?.default_key,
            )?.label ?? 'the local model'
          }
          onApprove={chat.approveRemote}
          onDecline={() =>
            chat.declineRemote(models.models?.default_key ?? 'mistral')
          }
          onDismiss={chat.dismissConsent}
        />
      )}

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        models={models.models?.models ?? []}
        conversations={conversations.conversations}
        onCommand={runCommand}
        onSelectModel={setModelKey}
        onOpenConversation={(id) => void chat.openConversation(id)}
      />
    </div>
  )
}
