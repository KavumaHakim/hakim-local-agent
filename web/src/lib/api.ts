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
} from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })

  if (!response.ok) {
    // FastAPI puts the useful text in `detail`; fall back to the status line
    // rather than showing an empty error.
    let message = response.statusText
    try {
      const body = await response.json()
      if (typeof body?.detail === 'string') message = body.detail
    } catch {
      /* not JSON; the status line will have to do */
    }
    throw new ApiError(response.status, message)
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
