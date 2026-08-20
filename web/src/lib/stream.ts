/**
 * Reads the chat endpoint's server-sent events.
 *
 * `EventSource` cannot be used here: it only issues GET requests and the
 * prompt has to travel in a body. So this is a plain POST whose response is
 * read as a stream and parsed as SSE.
 *
 * The parser is small because the server's format is fixed and narrow: one
 * `event:` line, one `data:` line of JSON, blank line between events, and
 * `:` comment lines used as heartbeats during the long silences while a model
 * loads.
 */

import { ApiError, readError } from './api'
import type { ChatRequest, TurnEvent } from './types'

export interface StreamHandlers {
  onEvent: (event: TurnEvent) => void
  signal?: AbortSignal
}

export async function streamChat(
  body: ChatRequest,
  { onEvent, signal }: StreamHandlers,
): Promise<void> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok) {
    // A 409 here is usually the consent gate: the turn would have gone to a
    // hosted provider. The detail object survives so the caller can ask.
    const { message, detail } = await readError(response)
    throw new ApiError(response.status, message, detail)
  }

  if (!response.body) throw new Error('The server sent no stream.')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // Events are separated by a blank line. Anything after the last separator
    // is a partial event and stays in the buffer until the rest arrives -
    // tokens routinely split across chunks.
    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      const event = parseBlock(block)
      if (event) onEvent(event)
      boundary = buffer.indexOf('\n\n')
    }
  }
}

function parseBlock(block: string): TurnEvent | null {
  let name = ''
  let data = ''

  for (const line of block.split('\n')) {
    // Heartbeat or any other comment.
    if (line.startsWith(':')) continue
    if (line.startsWith('event: ')) name = line.slice(7).trim()
    else if (line.startsWith('data: ')) data = line.slice(6)
  }

  if (!name) return null
  try {
    return { type: name, ...JSON.parse(data || '{}') } as TurnEvent
  } catch {
    // A malformed event is not worth tearing the turn down for; the stream
    // carries on and the missing piece shows up as absent progress.
    return null
  }
}
