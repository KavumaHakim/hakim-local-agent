/**
 * Shapes the API returns. Mirrors api/schemas.py by hand.
 *
 * Kept hand-written rather than generated: there are eight endpoints, the
 * generator would be another dependency and another build step, and a
 * mismatch shows up immediately in the one place each type is used.
 */

export interface ToolCall {
  name: string
  ok: boolean
  summary: string
  /**
   * The arguments the model passed, as pretty JSON.
   *
   * Optional because turns recorded before tool detail existed have neither
   * this nor `output`, and those rows are still rendered.
   */
  arguments?: string
  /** The whole result payload the model saw, as pretty JSON. */
  output?: string
  /** True when either was shortened for display. */
  clipped?: boolean
}

export interface Message {
  id: number
  role: 'user' | 'assistant' | 'tool' | string
  content: string
  tools: ToolCall[]
  elapsed: number | null
  model_key: string | null
  created_at: string
  /**
   * The model's thinking for this turn, when it produced any.
   *
   * Client-side only. The server streams it but never stores it, because a
   * thinking trace is per-turn and must not be replayed to the model on the
   * next round. So it survives until the page reloads and no longer.
   */
  reasoning?: string
}

export interface Conversation {
  id: number
  title: string
  model_key: string | null
  created_at: string
  updated_at: string
  message_count: number
}

export interface ConversationDetail extends Conversation {
  messages: Message[]
  /** Whether this conversation has already needed the strong model. */
  escalated: boolean
}

export type ModelState = 'stopped' | 'starting' | 'ready' | 'stopping' | 'failed'

export interface Model {
  key: string
  label: string
  description: string
  port: number
  url: string
  context: number
  threads: number
  min_free_mb: number
  /** False when the GGUF is not on disk. */
  available: boolean
  state: ModelState
  pid: number | null
  error: string
  /** Set when a model started with less free RAM than it wants. */
  warning: string
  /** True when the server was already running and we merely attached. */
  adopted: boolean
  /** "local" for a llama-server here, otherwise the hosted provider's name. */
  provider: string
  remote: boolean
  /** Remote only: whether the API key is present. */
  has_key: boolean
  /** Whether it can be used right now. A remote model needs a key AND network. */
  usable: boolean
}

export interface ModelsResponse {
  models: Model[]
  default_key: string
  active_key: string | null
  router_fast: string
  router_strong: string
  max_active: number
  idle_timeout_seconds: number
  available_ram_mb: number | null
  /** Whether hosted models can be reached at all. */
  online: boolean
}

export interface Tool {
  name: string
  category: string
  description: string
  parameters: Record<string, unknown>
}

export interface DisabledTool {
  category: string
  /** Names the environment variable that would enable it. */
  reason: string
}

export interface ToolSwitch {
  id: string
  label: string
  enabled: boolean
  /** On because the environment says so, so it survives a restart. */
  from_env: boolean
  /** The switch this one is the sharp end of, e.g. python for unrestricted. */
  depends_on: string | null
  /** What the tool can do, and what it cannot protect you from. */
  risk: string
}

export interface ToolsResponse {
  tools: Tool[]
  disabled: DisabledTool[]
  switches: ToolSwitch[]
  workspace: string
}

export interface Health {
  ok: boolean
  busy: boolean
  queue_depth: number
  workspace: string
  db_path: string
}

/* --- streaming events, in the order a healthy turn produces them --- */

export type TurnEvent =
  | { type: 'accepted'; turn_id: string; conversation_id: number; user_message_id: number; position: number }
  | { type: 'queued'; position: number }
  | { type: 'route'; key: string; label: string; reason: string; remote: boolean }
  | { type: 'fallback'; from: string; to: string; reason: string }
  | {
      type: 'model'
      key: string
      label: string
      state: 'loading' | 'ready'
      provider: string
      remote: boolean
      warning?: string
    }
  | { type: 'start'; model_key: string }
  | { type: 'token'; text: string }
  | { type: 'reasoning'; text: string }
  | { type: 'tool'; name: string; ok: boolean; summary: string }
  | { type: 'done'; message_id: number; content: string; tools: ToolCall[]; elapsed: number; model_key: string }
  | {
      type: 'error'
      kind: 'iteration_limit' | 'model' | 'agent' | 'internal'
      message: string
      tools?: ToolCall[]
      /** Only on iteration_limit: whether the strong model is still untried. */
      can_escalate?: boolean
    }

export interface ChatRequest {
  prompt: string
  conversation_id?: number | null
  model_key?: string | null
  enable_thinking?: boolean
  auto_route?: boolean
  /** Agreement to send this turn to a hosted provider. */
  confirm_remote?: boolean
}
