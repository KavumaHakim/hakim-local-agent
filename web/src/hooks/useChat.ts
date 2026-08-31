/**
 * The turn state machine.
 *
 * One turn moves through: queued (maybe), model loading, generating, done.
 * Each stage is surfaced separately because on this machine they have wildly
 * different durations - a model load can be 130 s and generation minutes - and
 * "thinking..." for all of them tells the user nothing about whether anything
 * is wrong.
 *
 * Several turns can be in flight at once, and this tracks all of them. Not
 * because they run at once - the server runs exactly one at a time, on two
 * cores that cannot do better - but because a question thought of while the
 * last one is still generating should not have to be held in the user's head
 * for five minutes. Each turn is its own SSE stream and its own entry here;
 * the phases keep them apart, since only one can ever be past `queued`.
 */

import { useCallback, useRef, useState } from 'react'
import { ApiError, api, type RemoteConfirmation } from '../lib/api'
import { streamChat } from '../lib/stream'
import type { ChatRequest, Message, ToolCall, TurnEvent } from '../lib/types'

export type TurnPhase =
  | 'idle'
  | 'queued'
  | 'loading'
  | 'generating'
  | 'error'

export interface TurnState {
  phase: TurnPhase
  /** Turns that must finish first. 0 means this one is running. */
  position: number
  /** Text streamed so far in the current round. */
  text: string
  /**
   * The model's thinking, accumulated across every round of this turn.
   *
   * Unlike `text` this is not cleared on a tool round: the whole point is
   * seeing how it got there, and the rounds read as one train of thought.
   */
  reasoning: string
  tools: ToolCall[]
  modelKey: string | null
  modelLabel: string | null
  /** Set when the auto-router moved this turn to another model. */
  routeReason: string | null
  ramWarning: string | null
  /** Set when a hosted model was wanted and there was no internet. */
  fallback: { from: string; to: string; reason: string } | null
  /** True while this turn is being answered off this machine. */
  remote: boolean
  provider: string | null
  startedAt: number | null
  error: { kind: string; message: string; canEscalate: boolean } | null
}

/**
 * One turn in flight, as the UI needs it.
 *
 * `key` is ours and exists from the moment the request leaves; `id` is the
 * server's and only arrives with `accepted`. Both are needed: the key routes
 * events and React keys, the id is what the stop endpoint takes.
 */
export interface Turn extends TurnState {
  key: string
  id: string | null
  /** What was asked, so a queued turn can say which one it is. */
  prompt: string
  /** True between asking this turn to stop and it actually stopping. */
  stopping: boolean
}

const IDLE: TurnState = {
  phase: 'idle',
  position: 0,
  text: '',
  reasoning: '',
  tools: [],
  modelKey: null,
  modelLabel: null,
  routeReason: null,
  ramWarning: null,
  fallback: null,
  remote: false,
  provider: null,
  startedAt: null,
  error: null,
}

export interface ChatOptions {
  modelKey: string | null
  enableThinking: boolean
  autoRoute: boolean
  onConversationChanged: (id: number) => void
}

let counter = 0

/**
 * How many turns this client will have in flight at once.
 *
 * Not a guess, and not the server's limit - the server holds eight, which is
 * the right number for several tabs. This is the browser's: every turn in
 * flight is an open SSE connection, and browsers allow six per origin over
 * HTTP/1.1. Reach that and *every* other request queues behind them, so the
 * page stops being able to ask anything - `/health` included - and looks
 * hung. Four leaves room for the rest of the app to keep working.
 *
 * It is also about the right number on its own terms: at under a token a
 * second, three questions waiting is already the best part of an hour.
 */
export const MAX_IN_FLIGHT = 4

export function useChat(options: ChatOptions) {
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [turns, setTurns] = useState<Turn[]>([])
  // A turn the router wanted to send off this machine, waiting on a yes.
  const [consent, setConsent] = useState<{
    request: RemoteConfirmation
    prompt: string
  } | null>(null)
  // One controller per turn in flight, so hanging up on one - or on all of
  // them, when the conversation changes - never touches the others.
  const aborts = useRef(new Map<string, AbortController>())
  // The same set, counted before React has committed the new state: two
  // sends in one tick would both read a stale `turns` and both think there
  // was room.
  const inFlight = useRef(new Set<string>())

  // Read inside the stream callback, which would otherwise close over the
  // options from the render that started the turn.
  const latest = useRef(options)
  latest.current = options

  /** Apply a change to one turn, leaving the rest alone. */
  const patch = useCallback(
    (key: string, change: Partial<Turn> | ((turn: Turn) => Partial<Turn>)) => {
      setTurns((current) =>
        current.map((turn) =>
          turn.key === key
            ? {
                ...turn,
                ...(typeof change === 'function' ? change(turn) : change),
              }
            : turn,
        ),
      )
    },
    [],
  )

  const drop = useCallback((key: string) => {
    aborts.current.delete(key)
    inFlight.current.delete(key)
    setTurns((current) => current.filter((turn) => turn.key !== key))
  }, [])

  const openConversation = useCallback(async (id: number | null) => {
    // Stops watching, and deliberately does not end the turns: one that has
    // cost minutes is worth finishing even if nobody is looking, and its
    // answer is stored either way.
    for (const controller of aborts.current.values()) controller.abort()
    aborts.current.clear()
    inFlight.current.clear()
    setTurns([])
    if (id === null) {
      setConversationId(null)
      setMessages([])
      return
    }
    const detail = await api.conversation(id)
    setConversationId(detail.id)
    setMessages(detail.messages)
  }, [])

  const send = useCallback(
    async (
      prompt: string,
      overrideModel?: string,
      confirmRemote = false,
      attachments: string[] = [],
    ) => {
      const trimmed = prompt.trim()
      // An attachment with no text is a valid turn.
      if (!trimmed && attachments.length === 0) return
      // Guarded here as well as in the composer, because `send` is also
      // reached from the palette, the slash commands and the retry buttons.
      if (inFlight.current.size >= MAX_IN_FLIGHT) return

      // Shown immediately. The real row exists server-side the moment the
      // request is accepted; this is only so the message does not appear to
      // vanish for however long the queue is.
      const optimistic: Message = {
        id: -Date.now() - counter,
        role: 'user',
        content: trimmed || '(image)',
        tools: [],
        elapsed: null,
        model_key: null,
        created_at: '',
      }
      setMessages((current) => [...current, optimistic])

      const body: ChatRequest = {
        prompt: trimmed,
        conversation_id: conversationId,
        model_key: overrideModel ?? latest.current.modelKey,
        enable_thinking: latest.current.enableThinking,
        auto_route: latest.current.autoRoute,
        confirm_remote: confirmRemote,
        attachments,
      }

      const key = `turn-${(counter += 1)}`
      inFlight.current.add(key)
      setTurns((current) => [
        ...current,
        {
          ...IDLE,
          phase: 'queued',
          key,
          id: null,
          prompt: trimmed || '(image)',
          stopping: false,
        },
      ])

      const controller = new AbortController()
      aborts.current.set(key, controller)
      // Kept alongside the state copy so `done` can attach the finished trace
      // to the message without reading state it may not have committed yet.
      let thinking = ''
      // Whether this turn reached an end of its own. Errors are an end that
      // stays on screen, so the tidy-up below must not sweep them away the
      // moment the stream closes behind them.
      let settled = false

      try {
        await streamChat(body, {
          signal: controller.signal,
          onEvent: (event) => apply(event),
        })
        // A stream that ends without `done`, `error` or `stopped` has nothing
        // left to say. Leaving the entry behind would show a turn that is
        // permanently about to start.
        if (!settled) drop(key)
      } catch (error) {
        if (controller.signal.aborted) return

        // The consent gate. Nothing was stored or queued, so the retry is the
        // same request with agreement attached - and the optimistic message
        // has to come back out, because there is no turn yet.
        const confirmation =
          error instanceof ApiError ? error.confirmation : null
        if (confirmation) {
          setMessages((current) =>
            current.filter((message) => message.id !== optimistic.id),
          )
          drop(key)
          setConsent({ request: confirmation, prompt: trimmed })
          return
        }

        // Anything else - a refused backlog, a server that went away - stays
        // on screen as this turn's own error, next to the question it belongs
        // to rather than replacing whatever else is running.
        patch(key, {
          phase: 'error',
          error: {
            kind: 'transport',
            message: error instanceof Error ? error.message : String(error),
            canEscalate: false,
          },
        })
      } finally {
        // The stream is closed either way, so its connection is back whatever
        // became of the turn - an errored entry stays on screen but no longer
        // holds one, and must not go on counting against the limit.
        aborts.current.delete(key)
        inFlight.current.delete(key)
      }

      function apply(event: TurnEvent) {
        switch (event.type) {
          case 'accepted':
            setConversationId(event.conversation_id)
            latest.current.onConversationChanged(event.conversation_id)
            setMessages((current) =>
              current.map((message) =>
                message.id === optimistic.id
                  ? { ...message, id: event.user_message_id }
                  : message,
              ),
            )
            patch(key, { id: event.turn_id, position: event.position })
            break

          case 'queued':
            patch(key, { phase: 'queued', position: event.position })
            break

          case 'route':
            patch(key, {
              routeReason: event.reason,
              modelKey: event.key,
              modelLabel: event.label,
              remote: event.remote,
            })
            break

          case 'fallback':
            patch(key, {
              fallback: { from: event.from, to: event.to, reason: event.reason },
              remote: false,
            })
            break

          case 'model':
            patch(key, (turn) => ({
              phase: event.state === 'loading' ? 'loading' : turn.phase,
              modelKey: event.key,
              modelLabel: event.label,
              remote: event.remote,
              provider: event.provider,
              ramWarning: event.warning || turn.ramWarning,
            }))
            break

          case 'start':
            patch(key, { phase: 'generating', startedAt: Date.now() })
            break

          case 'token':
            patch(key, (turn) => ({ text: turn.text + event.text }))
            break

          case 'reasoning':
            thinking += event.text
            patch(key, (turn) => ({ reasoning: turn.reasoning + event.text }))
            break

          case 'tool':
            // A tool round produces no prose, so the streamed text is cleared
            // rather than glued to whatever the next round writes.
            patch(key, (turn) => ({
              text: '',
              tools: [...turn.tools, event],
            }))
            break

          case 'done': {
            const answer: Message = {
              id: event.message_id,
              role: 'assistant',
              content: event.content,
              tools: event.tools,
              elapsed: event.elapsed,
              model_key: event.model_key,
              created_at: '',
              // Not persisted server-side, so it lasts until a reload.
              reasoning: thinking || undefined,
            }
            setMessages((current) => [...current, answer])
            settled = true
            drop(key)
            break
          }

          case 'stopped': {
            // Whatever had been generated is kept, and the server has already
            // stored it with a note saying it was cut short. Nothing is added
            // to the transcript when nothing was produced - an empty bubble
            // would be a row that says only "this happened".
            if (event.message_id !== null) {
              setMessages((current) => [
                ...current,
                {
                  id: event.message_id as number,
                  role: 'assistant',
                  content: event.content,
                  tools: event.tools,
                  elapsed: event.elapsed,
                  model_key: event.model_key ?? null,
                  created_at: '',
                  reasoning: thinking || undefined,
                },
              ])
            }
            settled = true
            drop(key)
            break
          }

          case 'error':
            settled = true
            patch(key, {
              phase: 'error',
              text: '',
              error: {
                kind: event.kind,
                message: event.message,
                canEscalate: Boolean(event.can_escalate),
              },
            })
            break
        }
      }
    },
    [conversationId, drop, patch],
  )

  const dismissError = useCallback((key: string) => drop(key), [drop])

  /** Agree to send the waiting turn to the hosted provider. */
  const approveRemote = useCallback(() => {
    if (!consent) return
    const { prompt, request } = consent
    setConsent(null)
    void send(prompt, request.model_key, true)
  }, [consent, send])

  /** Decline, and run it on the local model instead. */
  const declineRemote = useCallback(
    (localKey: string) => {
      if (!consent) return
      const { prompt } = consent
      setConsent(null)
      void send(prompt, localKey)
    },
    [consent, send],
  )

  const dismissConsent = useCallback(() => setConsent(null), [])

  /**
   * End one turn for real.
   *
   * The stream is deliberately left open. The server answers the stop request
   * immediately but the turn ends at its next checkpoint, and it is the
   * `stopped` event that carries whatever had been generated - so hanging up
   * here would throw away the very thing the stop was meant to preserve.
   *
   * If the request itself fails there is nothing left to wait for, so it falls
   * back to what this used to do: stop watching.
   */
  const stop = useCallback(
    async (key: string) => {
      const turn = turns.find((entry) => entry.key === key)
      const hangUp = () => {
        aborts.current.get(key)?.abort()
        drop(key)
      }
      if (!turn?.id) {
        hangUp()
        return
      }
      patch(key, { stopping: true })
      try {
        await api.stopTurn(turn.id)
      } catch {
        hangUp()
      }
    },
    [turns, patch, drop],
  )

  return {
    conversationId,
    messages,
    turns,
    consent,
    approveRemote,
    declineRemote,
    dismissConsent,
    /** True while anything is queued or running - errors are not work. */
    busy: turns.some((turn) => turn.phase !== 'error'),
    /** How many are waiting behind whatever is running. */
    waiting: turns.filter((turn) => turn.phase === 'queued').length,
    /** True when no more may be queued from here until one finishes. */
    atCapacity: turns.filter((turn) => turn.phase !== 'error').length
      >= MAX_IN_FLIGHT,
    send,
    openConversation,
    dismissError,
    stop,
  }
}
