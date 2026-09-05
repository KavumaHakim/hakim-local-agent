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
import { Resources } from './components/Resources'
import { RemoteConsent } from './components/RemoteConsent'
import { TurnStatus } from './components/TurnStatus'
import { ModelBrowser } from './components/ModelBrowser'
import { WorkspacePicker } from './components/WorkspacePicker'
import { CommandIcon, ExpandIcon } from './components/Icons'
import { useChat } from './hooks/useChat'
import {
  useConversations,
  useHotkey,
  useModelHub,
  useModels,
  useTools,
  useWorkspace,
} from './hooks/useResources'
import { COMMANDS, parseCommand, type CommandId } from './lib/commands'
import { api } from './lib/api'
import type { Attachment } from './lib/types'
import {
  apply as applyAppearance,
  load as loadAppearance,
  save as saveAppearance,
  type Appearance,
} from './lib/appearance'

export default function App() {
  const [draft, setDraft] = useState('')
  const [modelKey, setModelKey] = useState<string | null>(null)
  const [autoRoute, setAutoRoute] = useState(false)
  const [thinking, setThinking] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [note, setNote] = useState<string | null>(null)
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const [pane, setPane] = useState<PaneId>('history')
  const [paneOpen, setPaneOpen] = useState(true)
  const [pickingWorkspace, setPickingWorkspace] = useState(false)
  const [browsingModels, setBrowsingModels] = useState(false)
  // Seeded from storage, which main.tsx has already applied to the document -
  // so the first render agrees with what is on screen rather than correcting
  // it.
  const [appearance, setAppearance] = useState<Appearance>(loadAppearance)

  const models = useModels()
  const tools = useTools()
  const conversations = useConversations()
  const workspace = useWorkspace()

  // A finished download is a new model, and the catalogue is what the picker
  // reads - so rescan once when one lands rather than making someone reload.
  const hub = useModelHub(
    useCallback(() => {
      void models.rescan()
    }, [models]),
  )

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

  // Conversations the model is currently naming. Held here rather than in the
  // list itself because the list is refetched, and a refetch mid-write would
  // drop the fact that something is coming.
  const [naming, setNaming] = useState<Set<number>>(new Set())

  const chat = useChat({
    modelKey,
    enableThinking: thinking,
    autoRoute,
    onConversationChanged: () => void conversations.refresh(),
    onTitle: useCallback(
      (id: number, title: string | null) => {
        setNaming((current) => {
          const next = new Set(current)
          if (title === null) next.add(id)
          else next.delete(id)
          return next
        })
        // A real name means the stored row changed, so the list has to be
        // re-read. A null means it is still being written, or that nothing
        // usable came back - neither changed anything worth refetching.
        if (title !== null) void conversations.refresh()
      },
      [conversations],
    ),
  })

  useEffect(() => {
    if (modelKey === null && models.models) {
      setModelKey(models.models.active_key ?? models.models.default_key)
    }
  }, [models.models, modelKey])

  useEffect(() => {
    applyAppearance(appearance)
    saveAppearance(appearance)
  }, [appearance])

  const changeAppearance = useCallback((next: Partial<Appearance>) => {
    setAppearance((current) => ({ ...current, ...next }))
  }, [])

  // Asked once. Neither whisper nor a Piper voice appears or disappears while
  // the page is open, and the answer only decides whether a button is drawn.
  // They are independent installs, so they are two booleans and not one.
  const [dictation, setDictation] = useState(false)
  const [canSpeak, setCanSpeak] = useState(false)
  useEffect(() => {
    let cancelled = false
    void api
      .speechStatus()
      .then((status) => {
        if (cancelled) return
        setDictation(status.available)
        setCanSpeak(status.voice_available)
      })
      .catch(() => {
        // An older backend has no /api/speech at all, and that is a fine
        // reason to have no microphone rather than an error to report.
      })
    return () => {
      cancelled = true
    }
  }, [])

  useHotkey('k', () => setPaletteOpen((open) => !open), { meta: true })

  const bottom = useRef<HTMLDivElement>(null)
  // Follows the streamed text of whichever turn is running, and the arrival
  // of any new one - so queueing a question scrolls to it rather than
  // leaving it below the fold.
  const streamed = chat.turns.map((turn) => turn.text).join('').length
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [chat.messages.length, chat.turns.length, streamed])

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

  /**
   * Drop one attachment, releasing its preview.
   *
   * An object URL keeps the whole file alive until it is revoked, so every
   * path that removes an attachment has to come through here or through
   * `clearAttachments`. On an 8 GB machine a leaked image is not a rounding
   * error.
   */
  const forget = useCallback((path: string) => {
    setAttachments((current) => {
      const going = current.find((file) => file.path === path)
      if (going?.preview) URL.revokeObjectURL(going.preview)
      return current.filter((file) => file.path !== path)
    })
  }, [])

  const clearAttachments = useCallback(() => {
    setAttachments((current) => {
      for (const file of current) {
        if (file.preview) URL.revokeObjectURL(file.preview)
      }
      return []
    })
  }, [])

  /**
   * Copy this conversation up to a message, and open the copy.
   *
   * Opening it is the point: forking and staying put would leave no sign that
   * anything happened, and the copy is where the next question belongs.
   */
  const forkFrom = useCallback(
    async (messageId: number) => {
      if (chat.conversationId === null || messageId < 0) return
      try {
        const copy = await api.forkConversation(chat.conversationId, messageId)
        await conversations.refresh()
        await chat.openConversation(copy.id)
        setNote(null)
      } catch (failure) {
        setNote(failure instanceof Error ? failure.message : String(failure))
      }
    },
    [chat, conversations],
  )

  /** Remove one message, leaving the rest of the conversation alone. */
  const deleteMessage = useCallback(
    async (messageId: number) => {
      if (chat.conversationId === null || messageId < 0) return
      try {
        await api.deleteMessage(chat.conversationId, messageId)
        await chat.openConversation(chat.conversationId)
        setNote(null)
      } catch (failure) {
        // The useful refusal is the 409: a turn is still running.
        setNote(failure instanceof Error ? failure.message : String(failure))
      }
    },
    [chat],
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
      clearAttachments()
      void chat.send(text, undefined, false, paths)
    },
    [chat, runCommand, attachments, clearAttachments],
  )

  const attach = useCallback(async (files: FileList | File[]) => {
    setUploading(true)
    setUploadError(null)
    try {
      for (const file of Array.from(files)) {
        const stored = await api.upload(file)
        setAttachments((current) => [
          ...current,
          {
            ...stored,
            // Over the File the browser already holds, so the preview costs
            // no round trip and no route that serves workspace files back
            // out. Revoked in `forget` and `clearAttachments`.
            preview: file.type.startsWith('image/')
              ? URL.createObjectURL(file)
              : undefined,
          },
        ])
      }
    } catch (failure) {
      setUploadError(failure instanceof Error ? failure.message : String(failure))
    } finally {
      setUploading(false)
    }
  }, [])

  /**
   * Replace an already-asked question and ask it again.
   *
   * The refusal worth surfacing is the 409: the server will not rewind a
   * conversation while a turn is in flight, and silently doing nothing would
   * look like the edit had been lost.
   */
  const editQuestion = useCallback(
    async (messageId: number, text: string) => {
      try {
        await chat.editAndResend(messageId, text)
        setNote(null)
      } catch (failure) {
        setNote(failure instanceof Error ? failure.message : String(failure))
      }
    },
    [chat],
  )

  /**
   * Ask the last question again, replacing the answer it got.
   *
   * The question and everything that answered it are deleted first, and the
   * same text is then asked afresh. This used to append instead, which left
   * the model reading a transcript where the same thing was asked twice -
   * and a question asked twice is a different question.
   *
   * Nothing else is held fixed on purpose: the answer is regenerated with
   * whatever model and settings are selected *now*, so switching model and
   * pressing this is the obvious way to compare two answers to one question.
   */
  const regenerateLast = useCallback(async () => {
    const lastUser = [...chat.messages]
      .reverse()
      .find((message) => message.role === 'user')
    if (!lastUser) return
    try {
      await chat.editAndResend(lastUser.id, lastUser.content)
      setNote(null)
    } catch (failure) {
      // The useful refusal is the server's "a turn is still running".
      setNote(failure instanceof Error ? failure.message : String(failure))
    }
  }, [chat])

  const selectedModel =
    models.models?.models.find((model) => model.key === modelKey) ?? null
  // The one turn that is actually being worked on, if any. The rest are
  // waiting, and the server guarantees there is never more than one.
  const running = chat.turns.find(
    (turn) => turn.phase !== 'queued' && turn.phase !== 'error',
  )
  const runningPhase =
    running?.phase === 'loading'
      ? 'loading'
      : running?.phase === 'generating'
        ? 'streaming'
        : 'queued'
  const empty = chat.messages.length === 0 && chat.turns.length === 0
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
        theme={appearance.theme}
        onToggleTheme={() =>
          changeAppearance({
            theme: appearance.theme === 'dark' ? 'light' : 'dark',
          })
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
          namingConversations={naming}
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
          onSetServerExe={(path) => void models.setServerExe(path)}
          tools={tools.data}
          toolPending={tools.pending}
          onSetOcrBackend={(backend) => void tools.setOcrBackend(backend)}
          toolError={tools.error}
          onToggleTool={(id, enabled) => void toggleTool(id, enabled)}
          workspace={workspace.workspace}
          onOpenWorkspacePicker={() => setPickingWorkspace(true)}
          onOpenModelBrowser={() => setBrowsingModels(true)}
          autoRoute={autoRoute}
          onAutoRoute={setAutoRoute}
          thinking={thinking}
          onThinking={setThinking}
          appearance={appearance}
          onAppearance={changeAppearance}
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
              {runningPhase}
              {chat.waiting > 0 && ` · ${chat.waiting} waiting`}
            </span>
          )}

          <div className="ml-auto flex shrink-0 items-center gap-2">
            <Resources />
            <button
              type="button"
              onClick={() => setPaletteOpen(true)}
              className="flex h-[26px] shrink-0 items-center gap-1 rounded-md border border-line px-2 text-[11px] text-fg opacity-60 transition hover:border-accent-line hover:opacity-100"
            >
              <CommandIcon className="size-3" />K
            </button>
          </div>
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
                    // Not offered mid-turn: regenerating rewinds, and the
                    // server refuses that while anything is in flight. A
                    // button whose only outcome is an error is worse than
                    // no button.
                    index === lastIndex &&
                    message.role === 'assistant' &&
                    !chat.busy
                      ? () => void regenerateLast()
                      : undefined
                  }
                  onEdit={(text) => void editQuestion(message.id, text)}
                  onFork={
                    message.id > 0 ? () => void forkFrom(message.id) : undefined
                  }
                  onDelete={
                    message.id > 0 && !chat.busy
                      ? () => void deleteMessage(message.id)
                      : undefined
                  }
                  editingBlocked={
                    chat.busy
                      ? 'Editing rewinds the conversation, so it waits for the running turn to finish or be stopped.'
                      : message.id < 0
                        ? 'Still being sent.'
                        : undefined
                  }
                  canSpeak={canSpeak}
                />
              ))}
              {/* One per turn in flight, in the order they were asked. At
                  most one is past `queued` - the server runs them one at a
                  time - so this reads as the running turn followed by the
                  questions waiting behind it. */}
              {chat.turns.map((turn) => (
                <TurnStatus
                  key={turn.key}
                  turn={turn}
                  onEscalate={() => {
                    const strong = models.models?.router_strong
                    if (!strong) return
                    setModelKey(strong)
                    const prompt = turn.prompt
                    chat.dismissError(turn.key)
                    if (prompt) void chat.send(prompt, strong)
                  }}
                  onDismiss={() => chat.dismissError(turn.key)}
                  onStop={() => void chat.stop(turn.key)}
                  onApprove={(granted) =>
                    void chat.answerApproval(turn.key, granted)
                  }
                  onSkipReasoning={() => void chat.skipReasoning(turn.key)}
                />
              ))}
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
              disabled={false}
              placeholder={
                chat.busy
                  ? 'Ask the next one — it queues behind this turn…'
                  : 'Message Hakim…'
              }
              queued={chat.waiting}
              atCapacity={chat.atCapacity}
              attachments={attachments}
              onAttach={attach}
              onRemoveAttachment={forget}
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
              dictation={dictation}
            />
          </div>
        </div>
      </main>

      {browsingModels && (
        <ModelBrowser hub={hub} onClose={() => setBrowsingModels(false)} />
      )}

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
