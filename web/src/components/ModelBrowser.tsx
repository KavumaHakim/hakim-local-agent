/**
 * Finding and fetching models, without leaving the app.
 *
 * Choosing a model is the one decision this project leaves to the person using
 * it, because it depends on their RAM, their language, and what they want the
 * agent for. That is a reason to help with the choice — not to leave somebody
 * alone with a browser and a hundred near-identical repositories.
 *
 * The number that earns this screen is **needs_ram_mb**: what a file would want
 * once it is running, worked out from its size before a byte is downloaded.
 * Being told a 4.8 GB file wants 6.2 GB free is worth more than any download
 * speed, on a machine that has 8 GB in total.
 *
 * A dialog rather than something in the 262px pane, because a search result, a
 * dozen quantisations and a progress bar do not fit in a sidebar.
 */

import { useEffect, useState } from 'react'
import type { HubFile, HubModel, ModelDownload } from '../lib/types'
import { AlertIcon, CheckIcon, ChevronIcon, SearchIcon, StopIcon } from './Icons'

interface Props {
  hub: {
    query: string
    results: HubModel[]
    searching: boolean
    files: { repo: string; files: HubFile[]; free_ram_mb: number | null } | null
    openRepo: string | null
    downloads: ModelDownload[]
    error: string | null
    search: (text: string) => void
    open: (repo: string) => void
    download: (repo: string, path: string, sizeBytes: number) => void
    cancel: (id: string) => void
    clearError: () => void
  }
  onClose: () => void
}

function gigabytes(bytes: number): string {
  return bytes >= 1e9
    ? `${(bytes / 1e9).toFixed(1)} GB`
    : `${Math.round(bytes / 1e6)} MB`
}

function compact(count: number): string {
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`
  if (count >= 1e3) return `${Math.round(count / 1e3)}k`
  return String(count)
}

function remaining(seconds: number | null): string {
  if (seconds === null) return ''
  if (seconds < 60) return `${seconds}s left`
  const minutes = Math.round(seconds / 60)
  return minutes < 60 ? `${minutes}m left` : `${Math.round(minutes / 60)}h left`
}

export function ModelBrowser({ hub, onClose }: Props) {
  const [typed, setTyped] = useState(hub.query)

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const active = hub.downloads.filter((item) => item.state === 'running')
  const finished = hub.downloads.filter((item) => item.state !== 'running')

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
      onMouseDown={onClose}
    >
      <div
        className="flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-line bg-raised shadow-2xl"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="border-b border-line p-4">
          <h2 className="text-sm font-semibold">Find a model</h2>
          <p className="mt-1 text-xs leading-relaxed text-muted">
            GGUF models from Hugging Face, most downloaded first. Anything you
            fetch lands in <span className="font-mono">weights/</span> and is
            picked up automatically.
          </p>

          <form
            className="mt-3 flex h-9 items-center gap-2 rounded-md border border-transparent bg-sunken px-2.5 focus-within:border-accent-line"
            onSubmit={(event) => {
              event.preventDefault()
              hub.search(typed)
            }}
          >
            <SearchIcon className="size-3.5 shrink-0 text-faint" />
            <input
              value={typed}
              autoFocus
              onChange={(event) => setTyped(event.target.value)}
              placeholder="qwen3 1.7b, llama 3.2, phi-4 …"
              className="min-w-0 flex-1 bg-transparent text-[13px] outline-none placeholder:text-faint"
            />
            <button
              type="submit"
              disabled={hub.searching || !typed.trim()}
              className="h-6 shrink-0 rounded-md border border-line px-2 text-[11px] transition hover:border-accent-line disabled:opacity-30"
            >
              {hub.searching ? 'Looking…' : 'Search'}
            </button>
          </form>
        </div>

        {active.length > 0 && (
          <div className="border-b border-line px-4 py-3">
            {active.map((item) => (
              <div key={item.id} className="mb-2 last:mb-0">
                <div className="flex items-baseline gap-2 text-[11.5px]">
                  <span className="min-w-0 flex-1 truncate font-mono">
                    {item.name}
                  </span>
                  <span className="shrink-0 text-faint">
                    {gigabytes(item.seen_bytes)} / {gigabytes(item.total_bytes)}
                  </span>
                  <button
                    type="button"
                    onClick={() => hub.cancel(item.id)}
                    title="Stop this download"
                    className="shrink-0 text-faint transition hover:text-danger"
                  >
                    <StopIcon className="size-3" />
                  </button>
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-sunken">
                  <div
                    className="h-full rounded-full bg-accent transition-[width] duration-500"
                    style={{ width: `${item.percent}%` }}
                  />
                </div>
                <p className="mt-1 text-[10.5px] text-faint">
                  {item.percent.toFixed(0)}%
                  {item.bytes_per_second > 0 &&
                    ` · ${(item.bytes_per_second / 1e6).toFixed(1)} MB/s`}
                  {item.seconds_left !== null && ` · ${remaining(item.seconds_left)}`}
                </p>
              </div>
            ))}
          </div>
        )}

        {hub.error && (
          <div className="flex items-start gap-2 border-b border-line bg-danger/5 px-4 py-2.5">
            <AlertIcon className="mt-0.5 size-3.5 shrink-0 text-danger" />
            <p className="min-w-0 flex-1 text-[11.5px] leading-relaxed text-danger">
              {hub.error}
            </p>
            <button
              type="button"
              onClick={hub.clearError}
              className="shrink-0 text-faint transition hover:text-fg"
            >
              ✕
            </button>
          </div>
        )}

        <div className="min-h-[220px] flex-1 overflow-y-auto px-2 py-2">
          {hub.results.length === 0 && !hub.searching && (
            <p className="px-2 py-6 text-center text-[12px] text-faint">
              {hub.query
                ? 'Nothing found. Try a shorter query.'
                : 'Search for a model to get started.'}
            </p>
          )}

          {hub.results.map((model) => {
            const open = hub.openRepo === model.id
            return (
              <div key={model.id} className="mb-1">
                <button
                  type="button"
                  onClick={() => hub.open(model.id)}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-left transition hover:bg-tint"
                >
                  <ChevronIcon
                    className={`size-3 shrink-0 text-faint transition ${open ? 'rotate-90' : ''}`}
                  />
                  <span className="min-w-0 flex-1 truncate font-mono text-[12.5px]">
                    {model.id}
                  </span>
                  {model.gated && (
                    <span
                      title="Needs terms accepted on the Hugging Face website"
                      className="shrink-0 rounded border border-warn px-1 text-[10px] text-warn"
                    >
                      gated
                    </span>
                  )}
                  <span className="shrink-0 text-[11px] text-faint">
                    {compact(model.downloads)} ↓
                  </span>
                </button>

                {open && (
                  <FileList
                    files={hub.files?.repo === model.id ? hub.files.files : null}
                    freeRam={hub.files?.free_ram_mb ?? null}
                    downloads={hub.downloads}
                    onDownload={(file) =>
                      hub.download(model.id, file.path, file.size_bytes)
                    }
                  />
                )}
              </div>
            )
          })}
        </div>

        {finished.length > 0 && (
          <div className="border-t border-line px-4 py-2">
            {finished.slice(0, 3).map((item) => (
              <p key={item.id} className="text-[11px] text-faint">
                {item.state === 'done' ? (
                  <span className="text-ok">✓ </span>
                ) : (
                  <span className="text-danger">✕ </span>
                )}
                <span className="font-mono">{item.name}</span>
                {item.state === 'done'
                  ? ' — ready to use'
                  : ` — ${item.state}${item.error ? `: ${item.error}` : ''}`}
              </p>
            ))}
          </div>
        )}

        <div className="flex items-center border-t border-line p-3">
          <p className="flex-1 text-[10.5px] leading-snug text-faint">
            Downloads run one at a time and keep going if you close this.
          </p>
          <button
            type="button"
            onClick={onClose}
            className="h-8 rounded-md px-3 text-[12.5px] text-faint transition hover:text-fg"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

/**
 * The quantisations inside one repository.
 *
 * Each row answers the only question that matters before a long download: will
 * this run on this machine? The verdict compares what the file would need
 * against what is free right now, which is why it is shown here rather than
 * left for the model to fail at later.
 */
function FileList({
  files,
  freeRam,
  downloads,
  onDownload,
}: {
  files: HubFile[] | null
  freeRam: number | null
  downloads: ModelDownload[]
  onDownload: (file: HubFile) => void
}) {
  if (files === null) {
    return <p className="px-8 py-2 text-[11.5px] text-faint">Reading the files…</p>
  }
  if (files.length === 0) {
    return (
      <p className="px-8 py-2 text-[11.5px] text-faint">
        No GGUF files in this repository.
      </p>
    )
  }

  const busy = downloads.some((item) => item.state === 'running')

  return (
    <div className="mb-1 ml-5 border-l border-line pl-2">
      {files.map((file) => {
        const fits = freeRam === null ? null : file.needs_ram_mb <= freeRam
        return (
          <div
            key={file.path}
            className="flex items-center gap-2 rounded-md px-2 py-1 hover:bg-tint"
          >
            <span className="w-[68px] shrink-0 font-mono text-[11px] text-accent-soft">
              {file.quantisation || '—'}
            </span>
            <span className="w-[62px] shrink-0 text-[11px] text-muted">
              {gigabytes(file.size_bytes)}
            </span>
            <span
              title={
                fits === null
                  ? 'Free memory could not be determined'
                  : `Needs about ${file.needs_ram_mb.toLocaleString()} MB free; you have ${freeRam?.toLocaleString()} MB`
              }
              className={`min-w-0 flex-1 truncate text-[11px] ${
                fits === false ? 'text-warn' : 'text-faint'
              }`}
            >
              {fits === false
                ? `wants ~${(file.needs_ram_mb / 1024).toFixed(1)} GB free`
                : `~${(file.needs_ram_mb / 1024).toFixed(1)} GB to run`}
            </span>
            {fits === true && <CheckIcon className="size-3 shrink-0 text-ok" />}
            <button
              type="button"
              disabled={busy}
              onClick={() => onDownload(file)}
              className="h-6 shrink-0 rounded-md border border-line px-2 text-[11px] transition hover:border-accent-line disabled:opacity-30"
            >
              Get
            </button>
          </div>
        )
      })}
    </div>
  )
}
