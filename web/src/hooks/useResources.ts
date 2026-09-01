/**
 * Data hooks for the things the sidebar shows: models, conversations, tools.
 *
 * No data-fetching library. There are three resources, each refetched on an
 * explicit action, and a cache layer would be more code than the code it
 * replaced.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import type {
  Conversation,
  HubFiles,
  HubModel,
  ModelDownload,
  ModelOverride,
  ModelsResponse,
  OcrBackend,
  ToolsResponse,
  WorkspaceInfo,
} from '../lib/types'

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

  /**
   * Settings that change the catalogue rather than what is running.
   *
   * All of them return the fresh snapshot, so one round trip both applies the
   * change and refreshes the list - there is no second fetch to get out of
   * step with.
   */
  const setPrimary = useCallback(async (key: string) => {
    setBusyKey(key)
    setError(null)
    try {
      setData(await api.setPrimaryModel(key))
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
    } finally {
      setBusyKey(null)
    }
  }, [])

  const rescan = useCallback(async (): Promise<string[]> => {
    setBusyKey('__rescan__')
    setError(null)
    try {
      const result = await api.rescanModels()
      setData(result.models)
      return result.added
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
      return []
    } finally {
      setBusyKey(null)
    }
  }, [])

  const setHidden = useCallback(async (key: string, hidden: boolean) => {
    setBusyKey(key)
    setError(null)
    try {
      setData(await api.setModelHidden(key, hidden))
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
    } finally {
      setBusyKey(null)
    }
  }, [])

  const override = useCallback(
    async (key: string, values: ModelOverride) => {
      setBusyKey(key)
      setError(null)
      try {
        setData(await api.overrideModel(key, values))
      } catch (failure) {
        setError(failure instanceof Error ? failure.message : String(failure))
      } finally {
        setBusyKey(null)
      }
    },
    [],
  )

  const clearOverride = useCallback(async (key: string) => {
    setBusyKey(key)
    setError(null)
    try {
      setData(await api.clearModelOverride(key))
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
    } finally {
      setBusyKey(null)
    }
  }, [])

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

  return {
    models: data,
    busyKey,
    error,
    refresh,
    load,
    unload,
    setPrimary,
    rescan,
    setHidden,
    override,
    clearOverride,
  }
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
  const [pending, setPending] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setData(await api.tools())
    } catch {
      /* the roster reappears on the next call; a banner would say nothing */
    }
  }, [])

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

  const toggle = useCallback(async (id: string, enabled: boolean) => {
    setPending(id)
    setError(null)
    try {
      // The response is the whole roster, so the tool list and the switches
      // update together and cannot disagree about what is on.
      setData(await api.setTool(id, enabled))
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
    } finally {
      setPending(null)
    }
  }, [])

  const setOcrBackend = useCallback(async (backend: OcrBackend) => {
    setPending('ocr-backend')
    setError(null)
    try {
      setData(await api.setOcrBackend(backend))
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
    } finally {
      setPending(null)
    }
  }, [])

  return { data, toggle, pending, error, setOcrBackend, refresh }
}

/**
 * The folder the file tools may reach.
 *
 * Kept apart from `useTools` even though the roster carries the path too: the
 * workspace has its own history, its own default and its own failure modes —
 * a folder that is gone, one Windows will not list — and folding that into the
 * tool roster would make one shape mean two things.
 */
export function useWorkspace() {
  const [data, setData] = useState<WorkspaceInfo | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setData(await api.workspace())
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  /** Returns whether it moved, so the picker knows to close. */
  const choose = useCallback(async (path: string): Promise<boolean> => {
    setPending(true)
    setError(null)
    try {
      setData(await api.setWorkspace(path))
      return true
    } catch (failure) {
      // The server's refusals are the useful ones — "that is a file", "that
      // is the root of a drive" — so they are shown as they arrive.
      setError(failure instanceof Error ? failure.message : String(failure))
      return false
    } finally {
      setPending(false)
    }
  }, [])

  const reset = useCallback(async () => {
    setPending(true)
    setError(null)
    try {
      setData(await api.resetWorkspace())
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
    } finally {
      setPending(false)
    }
  }, [])

  return {
    workspace: data,
    pending,
    error,
    choose,
    reset,
    refresh,
    clearError: useCallback(() => setError(null), []),
  }
}

/**
 * Searching Hugging Face, and fetching what you pick.
 *
 * Downloads are polled rather than streamed. They last minutes to hours, a
 * dropped SSE connection over that span is normal, and the server already
 * keeps the progress - so asking every second is both simpler and more robust
 * than holding a connection open for an hour.
 */
export function useModelHub(onFinished: () => void) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<HubModel[]>([])
  const [searching, setSearching] = useState(false)
  const [files, setFiles] = useState<HubFiles | null>(null)
  const [openRepo, setOpenRepo] = useState<string | null>(null)
  const [downloads, setDownloads] = useState<ModelDownload[]>([])
  const [error, setError] = useState<string | null>(null)

  const search = useCallback(async (text: string) => {
    const trimmed = text.trim()
    setQuery(text)
    if (!trimmed) {
      setResults([])
      return
    }
    setSearching(true)
    setError(null)
    try {
      setResults((await api.hubSearch(trimmed)).models)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
      setResults([])
    } finally {
      setSearching(false)
    }
  }, [])

  const open = useCallback(
    async (repo: string) => {
      if (openRepo === repo) {
        setOpenRepo(null)
        setFiles(null)
        return
      }
      setOpenRepo(repo)
      setFiles(null)
      setError(null)
      try {
        setFiles(await api.hubFiles(repo))
      } catch (failure) {
        setError(failure instanceof Error ? failure.message : String(failure))
      }
    },
    [openRepo],
  )

  const refreshDownloads = useCallback(async () => {
    try {
      setDownloads((await api.downloads()).downloads)
    } catch {
      /* a poll that fails is retried a second later; a banner would flap */
    }
  }, [])

  const download = useCallback(
    async (repo: string, path: string, sizeBytes: number) => {
      setError(null)
      try {
        await api.startDownload(repo, path, sizeBytes)
        await refreshDownloads()
      } catch (failure) {
        // The refusals are the useful ones - not enough disk, already there,
        // one already running - so they are shown exactly as sent.
        setError(failure instanceof Error ? failure.message : String(failure))
      }
    },
    [refreshDownloads],
  )

  const cancel = useCallback(async (id: string) => {
    try {
      setDownloads((await api.cancelDownload(id)).downloads)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure))
    }
  }, [])

  // Poll only while something is running, and rescan once when one lands so
  // the new model appears in the picker without anyone reloading.
  const running = downloads.some((item) => item.state === 'running')
  const settled = useRef(new Set<string>())

  useEffect(() => {
    for (const item of downloads) {
      if (item.state === 'done' && !settled.current.has(item.id)) {
        settled.current.add(item.id)
        onFinished()
      }
    }
  }, [downloads, onFinished])

  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => void refreshDownloads(), 1000)
    return () => window.clearInterval(timer)
  }, [running, refreshDownloads])

  useEffect(() => {
    void refreshDownloads()
  }, [refreshDownloads])

  return {
    query,
    results,
    searching,
    files,
    openRepo,
    downloads,
    error,
    search,
    open,
    download,
    cancel,
    clearError: useCallback(() => setError(null), []),
  }
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
