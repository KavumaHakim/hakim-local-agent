/**
 * Choosing the folder the agent may work in.
 *
 * A dialog rather than something inside the 262px pane, because picking a
 * folder means walking a filesystem and that needs room.
 *
 * The walking is done by the API, not here, and it has to be: a directory
 * picker in a browser hands the page relative names and never the absolute
 * path the tools resolve against. So this navigates by asking the server for
 * one level at a time — which also means the path shown is the real one, not
 * a reconstruction.
 *
 * Typing a path is kept alongside the walk. Pasting one from Explorer is
 * faster than eleven clicks, and someone who knows where they are going should
 * not have to click their way there.
 */

import { useCallback, useEffect, useState } from 'react'
import { api } from '../lib/api'
import { basename } from '../lib/paths'
import type { DirectoryListing, WorkspaceInfo } from '../lib/types'
import { AlertIcon, CheckIcon, ChevronIcon, FolderIcon, HomeIcon } from './Icons'

interface Props {
  workspace: WorkspaceInfo
  pending: boolean
  error: string | null
  onChoose: (path: string) => Promise<boolean>
  onReset: () => void
  onClose: () => void
  onClearError: () => void
}

export function WorkspacePicker({
  workspace,
  pending,
  error,
  onChoose,
  onReset,
  onClose,
  onClearError,
}: Props) {
  const [listing, setListing] = useState<DirectoryListing | null>(null)
  const [loading, setLoading] = useState(true)
  const [typed, setTyped] = useState('')
  const [browseError, setBrowseError] = useState<string | null>(null)

  const open = useCallback(
    async (path: string) => {
      setLoading(true)
      setBrowseError(null)
      try {
        setListing(await api.browse(path))
      } catch (failure) {
        setBrowseError(
          failure instanceof Error ? failure.message : String(failure),
        )
      } finally {
        setLoading(false)
      }
    },
    [],
  )

  // Starts where the agent currently is, which is nearly always the right
  // neighbourhood: you are moving it somewhere near, or back to a recent one.
  useEffect(() => {
    void open(workspace.path)
  }, [open, workspace.path])

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  async function use(path: string) {
    onClearError()
    if (await onChoose(path)) onClose()
  }

  const here = listing?.path ?? workspace.path
  const isCurrent = here === workspace.path

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
      onMouseDown={onClose}
    >
      <div
        className="flex max-h-[80vh] w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-line bg-raised shadow-2xl"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="border-b border-line p-4">
          <h2 className="text-sm font-semibold">Choose a workspace</h2>
          <p className="mt-1 text-xs leading-relaxed text-muted">
            The one folder the agent's file tools may reach. Everything outside
            it is refused, whatever path the model asks for.
          </p>

          {workspace.active_tools.length > 0 && (
            <div className="mt-2.5 flex items-start gap-2 rounded-lg border border-warn/30 bg-warn/5 px-3 py-2">
              <AlertIcon className="mt-0.5 size-3.5 shrink-0 text-warn" />
              <p className="text-[11.5px] leading-relaxed text-muted">
                {workspace.active_tools.join(', ')}{' '}
                {workspace.active_tools.length === 1 ? 'is' : 'are'} switched
                on, so the agent will act on whatever you pick
                {workspace.writable ? ', and can change files there' : ''}.
              </p>
            </div>
          )}
        </div>

        {/* Where the walk can start: home, and the drives that exist. */}
        <div className="flex flex-wrap items-center gap-1.5 border-b border-line px-4 py-2.5">
          {(listing?.roots ?? []).map((root) => (
            <button
              key={root.path}
              type="button"
              onClick={() => void open(root.path)}
              className="flex h-6 items-center gap-1 rounded-md border border-line px-2 text-[11px] text-fg transition hover:border-accent-line"
            >
              {root.name === 'Home' ? (
                <HomeIcon className="size-3 opacity-60" />
              ) : null}
              {root.name}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2 border-b border-line px-4 py-2">
          <button
            type="button"
            disabled={!listing?.parent}
            onClick={() => listing?.parent && void open(listing.parent)}
            title="Up one level"
            className="grid size-6 shrink-0 place-items-center rounded-md text-fg opacity-60 transition hover:bg-tint hover:opacity-100 disabled:opacity-20"
          >
            <ChevronIcon className="size-3.5 -rotate-90" />
          </button>
          <FolderIcon className="size-3.5 shrink-0 text-accent" />
          <span
            title={here}
            className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-muted"
          >
            {here}
          </span>
        </div>

        <div className="min-h-[180px] flex-1 overflow-y-auto px-2 py-1.5">
          {loading && (
            <p className="px-2 py-3 text-[11.5px] text-faint">Reading…</p>
          )}

          {!loading && browseError && (
            <p className="px-2 py-3 text-[11.5px] text-danger">{browseError}</p>
          )}

          {!loading && !browseError && listing?.entries.length === 0 && (
            <p className="px-2 py-3 text-[11.5px] text-faint">
              No folders in here. It can still be the workspace.
            </p>
          )}

          {!loading &&
            listing?.entries.map((entry) => (
              <button
                key={entry.path}
                type="button"
                onClick={() => void open(entry.path)}
                onDoubleClick={() => void use(entry.path)}
                className="flex w-full items-center gap-2 rounded-md px-2 py-[5px] text-left text-[12.5px] transition hover:bg-tint"
              >
                <FolderIcon className="size-3.5 shrink-0 text-faint" />
                <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                {entry.path === workspace.path && (
                  <CheckIcon className="size-3 shrink-0 text-accent" />
                )}
              </button>
            ))}

          {listing?.note && (
            <p className="px-2 py-2 text-[11px] leading-relaxed text-faint">
              {listing.note}
            </p>
          )}
        </div>

        {workspace.recent.length > 1 && (
          <div className="border-t border-line px-4 py-2.5">
            <p className="mb-1.5 text-[10px] tracking-[0.1em] text-faint uppercase">
              Recent
            </p>
            <div className="flex flex-wrap gap-1.5">
              {workspace.recent
                .filter((path) => path !== workspace.path)
                .slice(0, 5)
                .map((path) => (
                  <button
                    key={path}
                    type="button"
                    title={path}
                    onClick={() => void use(path)}
                    className="max-w-full truncate rounded-md border border-line px-2 py-1 text-[11px] transition hover:border-accent-line"
                  >
                    {basename(path)}
                  </button>
                ))}
            </div>
          </div>
        )}

        <div className="border-t border-line px-4 py-2.5">
          <label className="mb-1.5 block text-[10px] tracking-[0.1em] text-faint uppercase">
            Or paste a path
          </label>
          <div className="flex gap-1.5">
            <input
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && typed.trim()) void use(typed.trim())
              }}
              placeholder="C:\Users\you\project"
              spellCheck={false}
              className="h-8 min-w-0 flex-1 rounded-md border border-transparent bg-sunken px-2.5 font-mono text-[11.5px] outline-none placeholder:text-faint focus:border-accent-line"
            />
            <button
              type="button"
              disabled={!typed.trim() || pending}
              onClick={() => void use(typed.trim())}
              className="h-8 shrink-0 rounded-md border border-line px-2.5 text-[11.5px] transition hover:border-accent-line disabled:opacity-30"
            >
              Go
            </button>
          </div>
        </div>

        {error && (
          <p className="border-t border-line px-4 py-2 text-[11.5px] leading-relaxed text-danger">
            {error}
          </p>
        )}

        <div className="flex items-center gap-2 border-t border-line p-3">
          {!workspace.from_env && (
            <button
              type="button"
              onClick={() => {
                onReset()
                onClose()
              }}
              title={workspace.default}
              className="h-8 rounded-md px-2 text-[11.5px] text-faint transition hover:text-fg"
            >
              Back to the default
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            className="ml-auto h-8 rounded-md px-3 text-[12.5px] text-faint transition hover:text-fg"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={pending || isCurrent}
            onClick={() => void use(here)}
            className="h-8 rounded-md border border-accent px-3 text-[12.5px] text-accent transition hover:bg-accent-tint disabled:cursor-not-allowed disabled:opacity-30"
          >
            {isCurrent ? 'Already here' : `Use ${basename(here)}`}
          </button>
        </div>
      </div>
    </div>
  )
}
