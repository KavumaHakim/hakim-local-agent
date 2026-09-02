/**
 * Typed wrappers over the API.
 *
 * Every path is relative, so the browser talks to whichever origin served the
 * page: Vite in development (which proxies /api), FastAPI in production. There
 * is no base URL to configure and no cross-origin request to permit.
 */

import type {
  Conversation,
  ConversationDetail,
  DirectoryListing,
  Health,
  HubFiles,
  HubSearchResult,
  ModelDownload,
  ModelOverride,
  ModelsResponse,
  RescanResponse,
  RewindResult,
  OcrBackend,
  SpeechStatus,
  StopTurnResult,
  Transcript,
  ToolsResponse,
  UploadResult,
  WorkspaceInfo,
} from './types'

/** The structured body a 409 carries when a turn would leave the machine. */
export interface RemoteConfirmation {
  kind: 'remote_confirmation_required'
  model_key: string
  label: string
  provider: string
  reason: string
  message: string
}

export class ApiError extends Error {
  status: number
  /**
   * FastAPI's `detail`, unwrapped.
   *
   * Usually a string, but the remote-confirmation refusal sends an object,
   * because "which model, whose servers, and why" cannot be reconstructed
   * from a sentence.
   */
  detail: unknown

  constructor(status: number, message: string, detail?: unknown) {
    super(message)
    this.status = status
    this.detail = detail
    this.name = 'ApiError'
  }

  /** The confirmation request, when that is what this refusal was. */
  get confirmation(): RemoteConfirmation | null {
    const detail = this.detail as RemoteConfirmation | undefined
    return detail?.kind === 'remote_confirmation_required' ? detail : null
  }
}

/** Pull FastAPI's `detail` out of an error body, whatever shape it is. */
export async function readError(
  response: Response,
): Promise<{ message: string; detail: unknown }> {
  try {
    const body = await response.json()
    const detail = body?.detail
    if (typeof detail === 'string') return { message: detail, detail }
    if (detail && typeof detail === 'object') {
      const message = (detail as { message?: string }).message
      return { message: message ?? response.statusText, detail }
    }
  } catch {
    /* not JSON; the status line will have to do */
  }
  return { message: response.statusText, detail: undefined }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })

  if (!response.ok) {
    const { message, detail } = await readError(response)
    throw new ApiError(response.status, message, detail)
  }

  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

export const api = {
  health: () => request<Health>('/health'),

  tools: () => request<ToolsResponse>('/tools'),

  setTool: (id: string, enabled: boolean) =>
    request<ToolsResponse>(`/tools/${encodeURIComponent(id)}`, {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    }),

  /** Choose which reader ocr_image uses. Applies from the next turn. */
  setOcrBackend: (backend: OcrBackend) =>
    request<ToolsResponse>('/ocr-backend', {
      method: 'POST',
      body: JSON.stringify({ backend }),
    }),

  /**
   * End a turn.
   *
   * Answers at once; the turn itself ends at its next checkpoint, and the
   * stream reports that with a `stopped` event. Never a 404 - a turn that
   * finished first is the outcome that was asked for.
   */
  stopTurn: (turnId: string) =>
    request<StopTurnResult>(`/chat/${encodeURIComponent(turnId)}/stop`, {
      method: 'POST',
    }),

  workspace: () => request<WorkspaceInfo>('/workspace'),

  /** Point the file tools at another folder. Applies from the next turn. */
  setWorkspace: (path: string) =>
    request<WorkspaceInfo>('/workspace', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),

  /** Back to the folder AGENT_WORKSPACE names. */
  resetWorkspace: () => request<WorkspaceInfo>('/workspace', { method: 'DELETE' }),

  /**
   * List one folder's sub-folders.
   *
   * The walking happens on the server because it has to: a directory picker in
   * a browser reports names, never the absolute path the tools resolve against.
   */
  browse: (path = '') =>
    request<DirectoryListing>(`/workspace/browse?path=${encodeURIComponent(path)}`),

  models: () => request<ModelsResponse>('/models'),

  /**
   * Look for models on Hugging Face.
   *
   * Sorted by downloads server-side: for any given model there are dozens of
   * re-uploads, and the popular one is overwhelmingly the complete, correctly
   * converted one that is still there next month.
   */
  hubSearch: (query: string, limit = 20) =>
    request<HubSearchResult>(
      `/hub/search?q=${encodeURIComponent(query)}&limit=${limit}`,
    ),

  /** The GGUF files in one repository, with what each would need in RAM. */
  hubFiles: (repo: string) =>
    request<HubFiles>(`/hub/files?repo=${encodeURIComponent(repo)}`),

  /** Fetch one into weights/. Refused up front if the disk cannot take it. */
  startDownload: (repo: string, path: string, sizeBytes: number) =>
    request<ModelDownload>('/hub/download', {
      method: 'POST',
      body: JSON.stringify({ repo, path, size_bytes: sizeBytes }),
    }),

  downloads: () => request<{ downloads: ModelDownload[] }>('/hub/downloads'),

  cancelDownload: (id: string) =>
    request<{ downloads: ModelDownload[] }>(
      `/hub/downloads/${encodeURIComponent(id)}/cancel`,
      { method: 'POST' },
    ),

  /** Re-read the models folder. Cheap: headers only, never a tensor. */
  rescanModels: () =>
    request<RescanResponse>('/models/rescan', { method: 'POST' }),

  /** Choose the model everything defaults to. Does not load it. */
  setPrimaryModel: (key: string) =>
    request<ModelsResponse>('/models/primary', {
      method: 'POST',
      body: JSON.stringify({ key }),
    }),

  setRouter: (fast: string, strong: string) =>
    request<ModelsResponse>('/models/router', {
      method: 'POST',
      body: JSON.stringify({ fast, strong }),
    }),

  /** Retune one model. Applies the next time it starts. */
  overrideModel: (key: string, values: ModelOverride) =>
    request<ModelsResponse>(`/models/${encodeURIComponent(key)}`, {
      method: 'PATCH',
      body: JSON.stringify(values),
    }),

  clearModelOverride: (key: string) =>
    request<ModelsResponse>(`/models/${encodeURIComponent(key)}/override`, {
      method: 'DELETE',
    }),

  setModelHidden: (key: string, hidden: boolean) =>
    request<ModelsResponse>(`/models/${encodeURIComponent(key)}/hidden`, {
      method: 'POST',
      body: JSON.stringify({ hidden }),
    }),

  loadModel: (key: string) =>
    request<ModelsResponse>(`/models/${encodeURIComponent(key)}/load`, {
      method: 'POST',
    }),

  unloadModel: (key: string) =>
    request<ModelsResponse>(`/models/${encodeURIComponent(key)}/unload`, {
      method: 'POST',
    }),

  /**
   * Upload an image for OCR.
   *
   * Deliberately not routed through `request`: that sets a JSON content type,
   * and multipart needs the browser to set its own with the boundary. Setting
   * it by hand produces a body the server cannot parse.
   */
  upload: async (file: File): Promise<UploadResult> => {
    const form = new FormData()
    form.append('file', file)
    const response = await fetch('/api/uploads', { method: 'POST', body: form })
    if (!response.ok) {
      const { message, detail } = await readError(response)
      throw new ApiError(response.status, message, detail)
    }
    return (await response.json()) as UploadResult
  },

  /** Whether speech to text is installed. Asked once, on load. */
  speechStatus: () => request<SpeechStatus>('/speech'),

  /**
   * Transcribe a recorded clip.
   *
   * Multipart for the same reason as `upload`: the browser has to set the
   * content type itself, boundary and all.
   */
  transcribe: async (clip: Blob): Promise<Transcript> => {
    const form = new FormData()
    form.append('file', clip, 'dictation.wav')
    const response = await fetch('/api/speech/transcribe', {
      method: 'POST',
      body: form,
    })
    if (!response.ok) {
      const { message, detail } = await readError(response)
      throw new ApiError(response.status, message, detail)
    }
    return (await response.json()) as Transcript
  },

  conversations: (limit = 30) =>
    request<Conversation[]>(`/conversations?limit=${limit}`),

  conversation: (id: number) => request<ConversationDetail>(`/conversations/${id}`),

  renameConversation: (id: number, title: string) =>
    request<Conversation>(`/conversations/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    }),

  deleteConversation: (id: number) =>
    request<void>(`/conversations/${id}`, { method: 'DELETE' }),

  /**
   * Delete a message and everything after it.
   *
   * What editing a question is built on: the old question and every reply to
   * it go, then the edited text is sent as a new turn. Refused with a 409
   * while any turn is in flight.
   */
  rewind: (conversationId: number, messageId: number) =>
    request<RewindResult>(
      `/conversations/${conversationId}/messages/${messageId}`,
      { method: 'DELETE' },
    ),
}
