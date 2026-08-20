/**
 * Data hooks for the things the sidebar shows: models, conversations, tools.
 *
 * No data-fetching library. There are three resources, each refetched on an
 * explicit action, and a cache layer would be more code than the code it
 * replaced.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import type { Conversation, ModelsResponse, ToolsResponse } from '../lib/types'

/** A ticking count of seconds since `since`, or 0 when it is null. */
export function useElapsed(since: number | null): number {
  const [seconds, setSeconds] = useState(0)

  useEffect(() => {
    if (since === null) {
      setSeconds(0)
      return
    }
    setSeconds(Math.floor((Date.now() - since) / 1000))
    const timer = window.setInterval(
      () => setSeconds(Math.floor((Date.now() - since) / 1000)),
      1000,
    )
    return () => window.clearInterval(timer)
  }, [since])

  return seconds
}

export function useModels() {
  const [data, setData] = useState<ModelsResponse | null>(null)
  const [busyKey, setBusyKey] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setData(await api.models())
      setError(null)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const load = useCallback(async (key: string) => {
    setBusyKey(key)
    setError(null)
    try {
      // Blocks for as long as the load takes - minutes for the 8B - so the
      // caller shows busyKey rather than assuming this returns quickly.
      setData(await api.loadModel(key))
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
    } finally {
      setBusyKey(null)
    }
  }, [])

  const unload = useCallback(async (key: string) => {
    setBusyKey(key)
    setError(null)
    try {
      setData(await api.unloadModel(key))
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
    } finally {
      setBusyKey(null)
    }
  }, [])

  return { models: data, busyKey, error, refresh, load, unload }
}

export function useConversations() {
  const [items, setItems] = useState<Conversation[]>([])

  const refresh = useCallback(async () => {
    try {
      setItems(await api.conversations())
    } catch {
      /* the list is not worth an error banner; it reappears on the next call */
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const remove = useCallback(
    async (id: number) => {
      await api.deleteConversation(id)
      await refresh()
    },
    [refresh],
  )

  const rename = useCallback(
    async (id: number, title: string) => {
      await api.renameConversation(id, title)
      await refresh()
    },
    [refresh],
  )

  return { conversations: items, refresh, remove, rename }
}

export function useTools() {
  const [data, setData] = useState<ToolsResponse | null>(null)

  useEffect(() => {
    let live = true
    void api
      .tools()
      .then((tools) => {
        if (live) setData(tools)
      })
      .catch(() => undefined)
    return () => {
      live = false
    }
  }, [])

  return data
}

/** Fires `handler` on a key chord, e.g. ctrl/cmd+K. */
export function useHotkey(
  key: string,
  handler: () => void,
  { meta = false }: { meta?: boolean } = {},
) {
  const saved = useRef(handler)
  saved.current = handler

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const chord = meta ? event.metaKey || event.ctrlKey : true
      if (chord && event.key.toLowerCase() === key.toLowerCase()) {
        event.preventDefault()
        saved.current()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [key, meta])
}
