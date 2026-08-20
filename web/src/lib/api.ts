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
  Health,
  ModelsResponse,
  ToolsResponse,
  UploadResult,
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

  models: () => request<ModelsResponse>('/models'),

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
}
