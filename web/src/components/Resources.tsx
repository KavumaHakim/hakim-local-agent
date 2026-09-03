/**
 * The header's right-hand corner: what the machine has left, and the switch
 * that turns everything off.
 *
 * RAM is the number that decides everything on this hardware - whether a
 * model loads, whether a turn takes thirty seconds or thirty minutes - and
 * until now it was only visible in Task Manager. Polled from an endpoint that
 * takes no lock, so it keeps updating while a model loads, which is exactly
 * when it is most worth watching.
 *
 * The power button is deliberately two clicks. It unloads every model, stops
 * the web server and the API, and there is no undo beyond running start.bat
 * again.
 */

import { useCallback, useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { ResourcesResponse, ShutdownResponse } from '../lib/types'
import { PowerIcon } from './Icons'

const POLL_MS = 5000

function gb(mb: number): string {
  return (mb / 1024).toFixed(1)
}

export function Resources() {
  const [data, setData] = useState<ResourcesResponse | null>(null)
  const [confirming, setConfirming] = useState(false)
  const [pending, setPending] = useState(false)
  const [stopped, setStopped] = useState<ShutdownResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    // A hidden tab does not need a number nobody is looking at.
    if (document.hidden) return
    try {
      setData(await api.resources())
    } catch {
      /* the next tick will try again; a banner for a missed poll would flap */
    }
  }, [])

  useEffect(() => {
    if (stopped) return
    void refresh()
    const timer = window.setInterval(() => void refresh(), POLL_MS)
    return () => window.clearInterval(timer)
  }, [refresh, stopped])

  const shutdown = useCallback(async () => {
    setPending(true)
    setError(null)
    try {
      setStopped(await api.shutdown())
    } catch (failure) {
      // The useful refusal is "a turn is running" - shown as sent.
      setError(failure instanceof Error ? failure.message : String(failure))
      setConfirming(false)
    } finally {
      setPending(false)
    }
  }, [])

  if (stopped) {
    return (
      <span
        className="flex shrink-0 items-center gap-1.5 rounded-full bg-tint px-2.5 py-0.5 text-[11px] text-muted"
        title={stopped.note}
      >
        <PowerIcon className="size-3" />
        Stopped. Close this tab.
      </span>
    )
  }

  const tight =
    data?.load_percent !== null && data?.load_percent !== undefined
      ? data.load_percent >= 85
      : false

  const resident =
    data?.resident_label &&
    (data.resident_state === 'starting'
      ? `loading ${data.resident_label}…`
      : data.resident_label)

  return (
    <div className="flex shrink-0 items-center gap-2">
      {data && data.available_mb !== null && data.total_mb !== null && (
        <span
          className={`flex items-center gap-1.5 text-[11px] tabular-nums ${
            tight ? 'font-semibold text-fg' : 'text-muted'
          }`}
          title={`${data.available_mb.toLocaleString()} MB of ${data.total_mb.toLocaleString()} MB available (${data.load_percent}% in use)${
            resident ? ` · ${resident}` : ' · no model loaded'
          }`}
        >
          <span
            className={`size-[6px] rounded-full ${
              tight ? 'bg-fg' : 'bg-accent'
            }`}
          />
          {gb(data.available_mb)} / {gb(data.total_mb)} GB free
          {resident && (
            <span className="hidden text-faint sm:inline">· {resident}</span>
          )}
        </span>
      )}

      {error && (
        <span className="max-w-[16rem] truncate text-[11px] text-faint" title={error}>
          {error}
        </span>
      )}

      {confirming ? (
        <span className="flex items-center gap-1 text-[11px]">
          <button
            type="button"
            disabled={pending}
            onClick={() => void shutdown()}
            className="rounded-md border border-line px-2 py-0.5 text-fg transition hover:border-accent-line disabled:opacity-50"
          >
            {pending ? 'Stopping…' : 'Stop everything'}
          </button>
          <button
            type="button"
            disabled={pending}
            onClick={() => setConfirming(false)}
            className="rounded-md px-2 py-0.5 text-muted transition hover:text-fg"
          >
            Cancel
          </button>
        </span>
      ) : (
        <button
          type="button"
          onClick={() => setConfirming(true)}
          title="Unload every model and stop both servers"
          className="grid size-7 shrink-0 place-items-center rounded-md text-fg opacity-60 transition hover:bg-tint hover:opacity-100"
        >
          <PowerIcon className="size-4" />
        </button>
      )}
    </div>
  )
}
