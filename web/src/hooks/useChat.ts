/**
 * The turn state machine.
 *
 * One turn moves through: queued (maybe), model loading, generating, done.
 * Each stage is surfaced separately because on this machine they have wildly
 * different durations - a model load can be 130 s and generation minutes - and
 * "thinking..." for all of them tells the user nothing about whether anything
 * is wrong.
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

export function useChat(options: ChatOptions) {
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [turn, setTurn] = useState<TurnState>(IDLE)
  // A turn the router wanted to send off this machine, waiting on a yes.
  const [consent, setConsent] = useState<{
    request: RemoteConfirmation
    prompt: string
  } | null>(null)
  const abort = useRef<AbortController | null>(null)
  // The id the stop endpoint needs, and whether one has been asked for. Both
  // refs rather than state: they are read inside the stream callback, which
  // closes over the render that started the turn.
  const turnId = useRef<string | null>(null)
  const [stopping, setStopping] = useState(false)

  // Read inside the stream callback, which would otherwise close over the
  // options from the render that started the turn.
  const latest = useRef(options)
  latest.current = options

  const openConversation = useCallback(async (id: number | null) => {
    // Stops watching, and deliberately does not end the turn: a turn that has
    // cost minutes is worth finishing even if nobody is looking, and its
    // answer is stored either way.
    abort.current?.abort()
    turnId.current = null
    setTurn(IDLE)
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

      // Shown immediately. The real row exists server-side the moment the
      // request is accepted; this is only so the message does not appear to
      // vanish for however long the queue is.
      const optimistic: Message = {
        id: -Date.now(),
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

      setTurn({ ...IDLE, phase: 'queued' })
      // Cleared before the new id arrives, so a stop pressed in the gap
      // cannot reach the turn that just finished.
      turnId.current = null
      const controller = new AbortController()
      abort.current = controller
      // Kept alongside the state copy so `done` can attach the finished trace
      // to the message without reading state it may not have committed yet.
      let thinking = ''

      try {
        await streamChat(body, {
          signal: controller.signal,
          onEvent: (event) => apply(event),
        })
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
          setTurn(IDLE)
          setConsent({ request: confirmation, prompt: trimmed })
          return
        }

        setTurn((current) => ({
          ...current,
          phase: 'error',
          error: {
            kind: 'transport',
            message: error instanceof Error ? error.message : String(error),
            canEscalate: false,
          },
        }))
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
            turnId.current = event.turn_id
            setTurn((current) => ({
              ...current,
              position: event.position,
              phase: event.position > 0 ? 'queued' : current.phase,
            }))
            break

          case 'queued':
            setTurn((current) => ({
              ...current,
              phase: 'queued',
              position: event.position,
            }))
            break

          case 'route':
            setTurn((current) => ({
              ...current,
              routeReason: event.reason,
              modelKey: event.key,
              modelLabel: event.label,
              remote: event.remote,
            }))
            break

          case 'fallback':
            setTurn((current) => ({
              ...current,
              fallback: { from: event.from, to: event.to, reason: event.reason },
              remote: false,
            }))
            break

          case 'model':
            setTurn((current) => ({
              ...current,
              phase: event.state === 'loading' ? 'loading' : current.phase,
              modelKey: event.key,
              modelLabel: event.label,
              remote: event.remote,
              provider: event.provider,
              ramWarning: event.warning || current.ramWarning,
            }))
            break

          case 'start':
            setTurn((current) => ({
              ...current,
              phase: 'generating',
              startedAt: Date.now(),
            }))
            break

          case 'token':
            setTurn((current) => ({ ...current, text: current.text + event.text }))
            break

          case 'reasoning':
            thinking += event.text
            setTurn((current) => ({
              ...current,
              reasoning: current.reasoning + event.text,
            }))
            break

          case 'tool':
            // A tool round produces no prose, so the streamed text is cleared
            // rather than glued to whatever the next round writes.
            setTurn((current) => ({
              ...current,
              text: '',
              tools: [...current.tools, event],
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
            setTurn(IDLE)
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
            setTurn(IDLE)
            break
          }

          case 'error':
            setTurn((current) => ({
              ...current,
              phase: 'error',
              text: '',
              error: {
                kind: event.kind,
                message: event.message,
                canEscalate: Boolean(event.can_escalate),
              },
            }))
            break
        }
      }
    },
    [conversationId],
  )

  const dismissError = useCallback(() => setTurn(IDLE), [])

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
   * End the turn for real.
   *
   * The stream is deliberately left open. The server answers the stop request
   * immediately but the turn ends at its next checkpoint, and it is the
   * `stopped` event that carries whatever had been generated - so hanging up
   * here would throw away the very thing the stop was meant to preserve.
   *
   * If the request itself fails there is nothing left to wait for, so it falls
   * back to what this used to do: stop watching.
   */
  const stop = useCallback(async () => {
    const id = turnId.current
    if (!id) {
      abort.current?.abort()
      setTurn(IDLE)
      return
    }
    setStopping(true)
    try {
      await api.stopTurn(id)
    } catch {
      abort.current?.abort()
      setTurn(IDLE)
    } finally {
      setStopping(false)
    }
  }, [])

  return {
    conversationId,
    messages,
    turn,
    consent,
    approveRemote,
    declineRemote,
    dismissConsent,
    busy: turn.phase !== 'idle' && turn.phase !== 'error',
    send,
    openConversation,
    dismissError,
    stop,
    stopping,
  }
}
