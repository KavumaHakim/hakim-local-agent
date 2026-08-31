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
  /** "chat" drives the agent loop; "ocr" is a vision backend, never a choice. */
  role: string
  /** "local" for a llama-server here, otherwise the hosted provider's name. */
  provider: string
  remote: boolean
  /** Remote only: whether the API key is present. */
  has_key: boolean
  /** Whether it can be used right now. A remote model needs a key AND network. */
  usable: boolean
  /** Found in the models folder rather than declared in models.json. */
  discovered: boolean
  /** Kept in the catalogue so it can be brought back, but not offered. */
  hidden: boolean
  /** Retuned from the settings panel rather than taken from the registry. */
  customised: boolean
  file_mb: number
  /** From the GGUF header: what the model was trained for. */
  training_context: number
  /** What its KV cache costs at the context it is actually running at. */
  kv_cache_mb: number
  /** Things worth knowing before choosing it. Missing weights, capped context. */
  notes: string[]
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
  /** Where to drop a .gguf for it to be picked up. */
  models_dir: string
  /** True when several models exist and none has been chosen as primary. */
  setup_required: boolean
}

export interface RescanResponse {
  success: boolean
  models_dir: string
  /** Keys that were not in the catalogue before this scan. */
  added: string[]
  total: number
  models: ModelsResponse
}

/** What the settings panel may change about one model. */
export interface ModelOverride {
  label?: string
  description?: string
  context?: number
  threads?: number
  min_free_mb?: number
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
  /** Which reader ocr_image uses: "tesseract" or "model". */
  ocr_backend: OcrBackend
  /** Whether that backend could actually run right now. */
  ocr_ready: boolean
  /** What to do about it when it cannot. */
  ocr_hint: string
}

export type OcrBackend = 'tesseract' | 'model'

/** The folder the file tools may reach, and what is pointed at it. */
export interface WorkspaceInfo {
  path: string
  /** What AGENT_WORKSPACE says. A restart returns here. */
  default: string
  /** True while nothing has been chosen in the UI. */
  from_env: boolean
  /** Folders chosen in this process, most recent first. */
  recent: string[]
  /** True when the workspace is the project's own directory. */
  is_project: boolean
  /** Whether any switched-on tool can change files in it. */
  writable: boolean
  /** Labels of the tools that would act on it. Read-only ones are not listed. */
  active_tools: string[]
}

export interface DirectoryEntry {
  name: string
  path: string
}

/** One level of the filesystem, for picking a folder without typing it. */
export interface DirectoryListing {
  path: string
  /** Null at a drive root, which is where walking up stops. */
  parent: string | null
  entries: DirectoryEntry[]
  /** Drives and home: the places a walk can start. */
  roots: DirectoryEntry[]
  /** Set when the folder could be listed only in part, or not at all. */
  note: string
}

/** An image stored in the workspace, ready to be named in a prompt. */
export interface UploadResult {
  /** Workspace-relative, the only form ocr_image accepts. */
  path: string
  name: string
  size: number
  /** False when OCR is off or its server is not running. */
  ocr_ready: boolean
  hint: string
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
  /** Workspace-relative paths from POST /uploads. */
  attachments?: string[]
}
