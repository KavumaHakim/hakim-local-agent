/**
 * The 56px icon rail.
 *
 * The redesign's main structural change. Everything used to live in one
 * scrolling sidebar — model, settings, history, tools — which meant the panel
 * was always long and nothing in it was ever properly visible. The rail picks
 * *one* subject and the pane beside it shows only that.
 *
 * Clicking the active button collapses the pane, so the rail doubles as the
 * show/hide control rather than needing a separate one.
 */

import type { ReactNode } from 'react'
import {
  ClockIcon,
  FolderIcon,
  MoonIcon,
  PlusIcon,
  SettingsIcon,
  SparkIcon,
  SunIcon,
  ToolIcon,
} from './Icons'

export type PaneId = 'history' | 'tools' | 'workspace' | 'settings'

interface Props {
  active: PaneId
  open: boolean
  onSelect: (pane: PaneId) => void
  onNewConversation: () => void
  theme: 'dark' | 'light'
  onToggleTheme: () => void
  /** Draws the amber dot on Tools: something risky is switched on. */
  toolsWarning: boolean
}

export function Rail({
  active,
  open,
  onSelect,
  onNewConversation,
  theme,
  onToggleTheme,
  toolsWarning,
}: Props) {
  return (
    <nav className="flex w-14 shrink-0 flex-col items-center gap-1 border-r border-line bg-surface pt-3 pb-2.5">
      <div className="mb-2.5 grid size-[30px] shrink-0 place-items-center rounded-[9px] border border-accent-line text-accent">
        <SparkIcon className="size-4" />
      </div>

      <RailButton label="New conversation" onClick={onNewConversation}>
        <PlusIcon className="size-[19px]" />
      </RailButton>

      <RailButton
        label="History"
        selected={open && active === 'history'}
        onClick={() => onSelect('history')}
      >
        <ClockIcon className="size-[19px]" />
      </RailButton>

      <RailButton
        label="Tools"
        selected={open && active === 'tools'}
        onClick={() => onSelect('tools')}
        badge={toolsWarning}
      >
        <ToolIcon className="size-[19px]" />
      </RailButton>

      <RailButton
        label="Workspace"
        selected={open && active === 'workspace'}
        onClick={() => onSelect('workspace')}
      >
        <FolderIcon className="size-[19px]" />
      </RailButton>

      <div className="mt-auto flex flex-col items-center gap-1">
        <RailButton
          label={theme === 'dark' ? 'Switch to light' : 'Switch to dark'}
          onClick={onToggleTheme}
        >
          {theme === 'dark' ? (
            <MoonIcon className="size-[19px]" />
          ) : (
            <SunIcon className="size-[19px]" />
          )}
        </RailButton>

        <RailButton
          label="Settings"
          selected={open && active === 'settings'}
          onClick={() => onSelect('settings')}
        >
          <SettingsIcon className="size-[19px]" />
        </RailButton>
      </div>
    </nav>
  )
}

function RailButton({
  label,
  onClick,
  selected = false,
  badge = false,
  children,
}: {
  label: string
  onClick: () => void
  selected?: boolean
  badge?: boolean
  children: ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      aria-pressed={selected}
      className={`relative grid size-[38px] shrink-0 place-items-center rounded-[10px] transition ${
        selected ? 'text-fg opacity-100' : 'text-fg opacity-75 hover:opacity-100'
      }`}
    >
      {selected && (
        <>
          <span className="absolute inset-0 rounded-[10px] bg-accent-tint" />
          {/* A short accent mark, solid — the system's rules fade at their
              ends, but marks like this stay. */}
          <span className="absolute top-[11px] -left-[9px] h-4 w-[3px] rounded-[2px] bg-accent" />
        </>
      )}
      {!selected && (
        <span className="absolute inset-0 rounded-[10px] transition hover:bg-tint" />
      )}
      <span className="relative">{children}</span>
      {badge && (
        <span className="absolute top-1.5 right-[5px] size-[5px] rounded-full bg-warn" />
      )}
    </button>
  )
}
