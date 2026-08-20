/**
 * The application shell.
 *
 * Holds the settings the composer sends with each turn (model, thinking,
 * auto-routing) and routes the slash commands to the REST endpoints. The
 * server has no notion of a command or a session; this is where both live.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { CommandPalette } from './components/CommandPalette'
import { Composer } from './components/Composer'
import { EmptyState, MessageView } from './components/Messages'
import { Sidebar } from './components/Sidebar'
import { TurnStatus } from './components/TurnStatus'
import { SidebarIcon } from './components/Icons'
import { useChat } from './hooks/useChat'
import { useConversations, useHotkey, useModels, useTools } from './hooks/useResources'
import { COMMANDS, parseCommand, type CommandId } from './lib/commands'

export default function App() {
  const [draft, setDraft] = useState('')
  const [modelKey, setModelKey] = useState<string | null>(null)
  const [autoRoute, setAutoRoute] = useState(false)
  const [thinking, setThinking] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [note, setNote] = useState<string | null>(null)

  const models = useModels()
  const tools = useTools()
  const conversations = useConversations()

  const chat = useChat({
    modelKey,
    enableThinking: thinking,
    autoRoute,
    // A new conversation has to appear in the sidebar without a manual reload.
    onConversationChanged: () => void conversations.refresh(),
  })

  // Default to whatever the server considers active, or its default.
  useEffect(() => {
    if (modelKey === null && models.models) {
      setModelKey(models.models.active_key ?? models.models.default_key)
    }
  }, [models.models, modelKey])

  useHotkey('k', () => setPaletteOpen((open) => !open), { meta: true })

  const bottom = useRef<HTMLDivElement>(null)
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [chat.messages.length, chat.turn.phase, chat.turn.text])

  const newConversation = useCallback(() => {
    void chat.openConversation(null)
    setNote(null)
  }, [chat])

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
          setThinking((value) => {
            setNote(`Extended thinking ${!value ? 'on' : 'off'}.`)
            return !value
          })
          break
        case 'models':
          setNote(
            models.models?.models
              .map(
                (model) =>
                  `${model.key} — ${model.label}${model.state === 'ready' ? ' (loaded)' : ''}${!model.available ? ' (file missing)' : ''}`,
              )
              .join('\n') ?? 'No models.',
          )
          break
        case 'tools':
          setNote(
            tools.data
              ? `Enabled: ${tools.data.tools.map((tool) => tool.name).join(', ')}\nOff: ${tools.data.disabled.map((item) => item.category).join(', ')}`
              : 'Tools not loaded yet.',
          )
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
    [models, modelKey, tools, newConversation],
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
      void chat.send(text)
    },
    [chat, runCommand],
  )

  const empty = chat.messages.length === 0 && chat.turn.phase === 'idle'

  return (
    <div className="flex h-full">
      {sidebarOpen && (
        <Sidebar
          models={models.models}
          modelsBusyKey={models.busyKey}
          modelsError={models.error}
          selectedKey={modelKey}
          onSelectModel={setModelKey}
          onLoadModel={(key) => void models.load(key)}
          onUnloadModel={(key) => void models.unload(key)}
          conversations={conversations.conversations}
          activeConversationId={chat.conversationId}
          onOpenConversation={(id) => void chat.openConversation(id)}
          onDeleteConversation={async (id) => {
            await conversations.remove(id)
            if (id === chat.conversationId) newConversation()
          }}
          onNewConversation={newConversation}
          tools={tools.data}
          toolPending={tools.pending}
          toolError={tools.error}
          onToggleTool={(id, enabled) => void tools.toggle(id, enabled)}
          autoRoute={autoRoute}
          onAutoRoute={setAutoRoute}
          thinking={thinking}
          onThinking={setThinking}
        />
      )}

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-line px-4 py-2.5">
          <button
            type="button"
            onClick={() => setSidebarOpen((open) => !open)}
            title="Toggle sidebar"
            className="rounded-lg p-1.5 text-faint transition hover:text-fg"
          >
            <SidebarIcon className="size-4" />
          </button>
          <p className="min-w-0 truncate text-xs text-faint">
            {models.models?.models.find((model) => model.key === modelKey)?.label ??
              'No model'}
            {autoRoute && ' · auto-routing'}
            {thinking && ' · thinking'}
          </p>
          <button
            type="button"
            onClick={() => setPaletteOpen(true)}
            className="ml-auto rounded-lg border border-line px-2 py-1 text-[11px] text-faint transition hover:text-fg"
          >
            ⌘K
          </button>
        </header>

        <div className="flex-1 overflow-y-auto">
          {empty ? (
            <EmptyState onPick={(text) => setDraft(text)} />
          ) : (
            <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
              {chat.messages.map((message) => (
                <MessageView key={message.id} message={message} />
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
                onCancel={chat.cancel}
              />
              <div ref={bottom} />
            </div>
          )}
        </div>

        <div className="border-t border-line px-4 py-3">
          <div className="mx-auto max-w-3xl">
            {note && (
              <div className="mb-2 flex items-start gap-2 rounded-xl border border-line bg-surface px-3 py-2">
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
            />
          </div>
        </div>
      </main>

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
