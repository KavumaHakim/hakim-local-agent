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
import { api } from '../lib/api'
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
  const abort = useRef<AbortController | null>(null)

  // Read inside the stream callback, which would otherwise close over the
  // options from the render that started the turn.
  const latest = useRef(options)
  latest.current = options

  const openConversation = useCallback(async (id: number | null) => {
    abort.current?.abort()
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
    async (prompt: string, overrideModel?: string) => {
      const trimmed = prompt.trim()
      if (!trimmed) return

      // Shown immediately. The real row exists server-side the moment the
      // request is accepted; this is only so the message does not appear to
      // vanish for however long the queue is.
      const optimistic: Message = {
        id: -Date.now(),
        role: 'user',
        content: trimmed,
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
      }

      setTurn({ ...IDLE, phase: 'queued' })
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
            }))
            break

          case 'model':
            setTurn((current) => ({
              ...current,
              phase: event.state === 'loading' ? 'loading' : current.phase,
              modelKey: event.key,
              modelLabel: event.label,
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

  const cancel = useCallback(() => {
    // Stops watching, not the turn itself. The server finishes and stores the
    // answer, which is the right trade when a turn has cost minutes already.
    abort.current?.abort()
    setTurn(IDLE)
  }, [])

  return {
    conversationId,
    messages,
    turn,
    busy: turn.phase !== 'idle' && turn.phase !== 'error',
    send,
    openConversation,
    dismissError,
    cancel,
  }
}
