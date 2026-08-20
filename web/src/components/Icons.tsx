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

export const FolderIcon = svg(
  <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />,
)
