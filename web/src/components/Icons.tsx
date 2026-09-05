/**
 * The icons this app uses, as inline SVG.
 *
 * An icon library would be a dependency and a bundle for the dozen glyphs
 * actually needed. These inherit `currentColor`, so they theme themselves.
 */

interface IconProps {
  className?: string
}

function svg(path: React.ReactNode, extra?: Record<string, string>) {
  return function Icon({ className = 'size-4' }: IconProps) {
    return (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
        aria-hidden="true"
        {...extra}
      >
        {path}
      </svg>
    )
  }
}

export const PlusIcon = svg(<path d="M12 5v14M5 12h14" />)

export const SendIcon = svg(<path d="m5 12 14-7-4.5 7L19 19z" />)

export const TrashIcon = svg(
  <>
    <path d="M4 7h16M10 11v6M14 11v6" />
    <path d="M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2" />
  </>,
)

export const ChipIcon = svg(
  <>
    <rect x="7" y="7" width="10" height="10" rx="2" />
    <path d="M4 10h3M4 14h3M17 10h3M17 14h3M10 4v3M14 4v3M10 17v3M14 17v3" />
  </>,
)

export const ToolIcon = svg(
  <path d="M14.7 6.3a4 4 0 0 1-5 5L5 16v3h3l4.7-4.7a4 4 0 0 1 5-5l-2-2 2-2 2 2z" />,
)

export const ChatIcon = svg(
  <path d="M20 15a2 2 0 0 1-2 2H8l-4 4V6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2z" />,
)

export const CommandIcon = svg(
  <path d="M9 6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3z" />,
)

export const SidebarIcon = svg(
  <>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M10 4v16" />
  </>,
)

export const CheckIcon = svg(<path d="m5 13 4 4L19 7" />)

/** A branch splitting from a trunk: this conversation, carried on twice. */
export const ForkIcon = svg(
  <>
    <circle cx="6" cy="18" r="2.5" />
    <circle cx="6" cy="6" r="2.5" />
    <circle cx="18" cy="8" r="2.5" />
    <path d="M6 8.5v7" />
    <path d="M15.5 8.5c-4 0-9 1-9 7" />
  </>,
)

export const PowerIcon = svg(
  <>
    <path d="M12 3v9" />
    <path d="M6.3 6.3a8 8 0 1 0 11.4 0" />
  </>,
)

export const AlertIcon = svg(
  <>
    <path d="M12 9v4M12 17h.01" />
    <path d="M10.3 4.3 2.7 17a2 2 0 0 0 1.7 3h15.2a2 2 0 0 0 1.7-3L13.7 4.3a2 2 0 0 0-3.4 0z" />
  </>,
)

export const StopIcon = svg(<rect x="7" y="7" width="10" height="10" rx="2" />)

export const SparkIcon = svg(
  <path d="M12 3v4M12 17v4M3 12h4M17 12h4M6.3 6.3l2.8 2.8M14.9 14.9l2.8 2.8M17.7 6.3l-2.8 2.8M9.1 14.9l-2.8 2.8" />,
)

export const SpeakerIcon = svg(
  <>
    <path d="M11 5 6 9H3v6h3l5 4z" />
    <path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13" />
  </>,
)

export const MicIcon = svg(
  <>
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
  </>,
)

export const PaperclipIcon = svg(
  <path d="M21 11.5 12.5 20a5 5 0 0 1-7-7l8-8a3.5 3.5 0 1 1 5 5l-8 8a2 2 0 1 1-3-3l7-7" />,
)

export const ImageIcon = svg(
  <>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <circle cx="8.5" cy="9.5" r="1.5" />
    <path d="m4 17 4.5-4.5a2 2 0 0 1 2.8 0L16 17M15 14l1.5-1.5a2 2 0 0 1 2.8 0L20 13.5" />
  </>,
)

export const CopyIcon = svg(
  <>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1" />
  </>,
)

export const ChevronIcon = svg(<path d="m9 6 6 6-6 6" />)

export const CloudIcon = svg(
  <path d="M7 18a4 4 0 0 1-.5-7.97 5.5 5.5 0 0 1 10.6-1.02A3.5 3.5 0 0 1 17.5 18z" />,
)

export const HomeIcon = svg(
  <path d="M4 10.5 12 4l8 6.5V19a1 1 0 0 1-1 1h-4v-6H9v6H5a1 1 0 0 1-1-1z" />,
)

export const BrainIcon = svg(
  <>
    <path d="M9.5 4a2.5 2.5 0 0 0-2.5 2.5A2.5 2.5 0 0 0 5 9a2.5 2.5 0 0 0 1 2 2.5 2.5 0 0 0 .5 4.5A2.5 2.5 0 0 0 9.5 20a2.5 2.5 0 0 0 2.5-2.5v-11A2.5 2.5 0 0 0 9.5 4z" />
    <path d="M14.5 4A2.5 2.5 0 0 1 17 6.5 2.5 2.5 0 0 1 19 9a2.5 2.5 0 0 1-1 2 2.5 2.5 0 0 1-.5 4.5A2.5 2.5 0 0 1 14.5 20a2.5 2.5 0 0 1-2.5-2.5" />
  </>,
)

export const ClockIcon = svg(
  <>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M12 7.5V12l3 1.8" />
  </>,
)

export const MoonIcon = svg(
  <path d="M20.5 14.2A8.8 8.8 0 019.8 3.5a8.8 8.8 0 1010.7 10.7z" />,
)

export const SunIcon = svg(
  <>
    <circle cx="12" cy="12" r="4.2" />
    <path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.5 1.5M16.9 16.9l1.5 1.5M18.4 5.6l-1.5 1.5M7.1 16.9l-1.5 1.5" />
  </>,
)

export const SettingsIcon = svg(
  <>
    <circle cx="12" cy="12" r="3" />
    <path d="M12 4v2M12 18v2M4 12h2M18 12h2M6.3 6.3l1.4 1.4M16.3 16.3l1.4 1.4M17.7 6.3l-1.4 1.4M7.7 16.3l-1.4 1.4" />
  </>,
)

export const RetryIcon = svg(<path d="M19 12a7 7 0 11-2.6-5.4M19 4v4h-4" />)

export const SearchIcon = svg(
  <>
    <circle cx="11" cy="11" r="6.5" />
    <path d="M16 16l4 4" />
  </>,
)

export const CollapseIcon = svg(<path d="M14 7l-5 5 5 5" />)

export const ExpandIcon = svg(<path d="M10 7l5 5-5 5" />)

export const PencilIcon = svg(
  <>
    <path d="M4 20h4l10-10a2.8 2.8 0 0 0-4-4L4 16z" />
    <path d="M13.5 6.5 17.5 10.5" />
  </>,
)

export const FolderIcon = svg(
  <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />,
)
