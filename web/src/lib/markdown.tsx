/**
 * A small markdown renderer for assistant replies.
 *
 * Written rather than installed, for the same reason the agent has no agent
 * framework: the subset that actually appears in these replies is small, and
 * a renderer that builds React elements needs no HTML sanitiser, because it
 * never produces HTML. There is no `dangerouslySetInnerHTML` anywhere in this
 * app, so a model that emits `<script>` emits five harmless characters.
 *
 * Supported: fenced code, headings, bullet and numbered lists, blockquotes,
 * rules, paragraphs, and inline code, bold, italic and links.
 */

import type { ReactNode } from 'react'

export function Markdown({ text }: { text: string }) {
  return <>{renderBlocks(text)}</>
}

function renderBlocks(text: string): ReactNode[] {
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  const blocks: ReactNode[] = []
  let index = 0
  let key = 0

  while (index < lines.length) {
    const line = lines[index]

    // Fenced code. Everything inside is literal, including markdown.
    if (line.trimStart().startsWith('```')) {
      const language = line.trim().slice(3).trim()
      const body: string[] = []
      index += 1
      while (index < lines.length && !lines[index].trimStart().startsWith('```')) {
        body.push(lines[index])
        index += 1
      }
      index += 1 // closing fence, or the end of the text
      blocks.push(<CodeBlock key={key++} language={language} code={body.join('\n')} />)
      continue
    }

    if (!line.trim()) {
      index += 1
      continue
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(line)
    if (heading) {
      const level = heading[1].length
      const Tag = `h${Math.min(level + 2, 6)}` as 'h3'
      blocks.push(
        <Tag key={key++} className="mt-4 mb-2 font-semibold text-fg first:mt-0">
          {inline(heading[2])}
        </Tag>,
      )
      index += 1
      continue
    }

    if (/^\s*([-*_])\s*\1\s*\1[\s\S]*$/.test(line) && line.trim().length >= 3) {
      blocks.push(<hr key={key++} className="my-4 border-line" />)
      index += 1
      continue
    }

    if (/^\s*>\s?/.test(line)) {
      const body: string[] = []
      while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
        body.push(lines[index].replace(/^\s*>\s?/, ''))
        index += 1
      }
      blocks.push(
        <blockquote
          key={key++}
          className="my-3 border-l-2 border-accent/50 pl-3 text-muted italic"
        >
          {renderBlocks(body.join('\n'))}
        </blockquote>,
      )
      continue
    }

    const bullet = /^\s*[-*+]\s+/
    const numbered = /^\s*\d+[.)]\s+/
    if (bullet.test(line) || numbered.test(line)) {
      const ordered = numbered.test(line)
      const pattern = ordered ? numbered : bullet
      const items: string[] = []
      while (index < lines.length && pattern.test(lines[index])) {
        items.push(lines[index].replace(pattern, ''))
        index += 1
      }
      const List = ordered ? 'ol' : 'ul'
      blocks.push(
        <List
          key={key++}
          className={`my-3 space-y-1 pl-5 ${ordered ? 'list-decimal' : 'list-disc'} marker:text-muted`}
        >
          {items.map((item, position) => (
            <li key={position}>{inline(item)}</li>
          ))}
        </List>,
      )
      continue
    }

    // Anything else is a paragraph: consecutive non-blank lines that are not
    // the start of another block.
    const paragraph: string[] = []
    while (
      index < lines.length &&
      lines[index].trim() &&
      !lines[index].trimStart().startsWith('```') &&
      !/^(#{1,6})\s+/.test(lines[index]) &&
      !/^\s*>\s?/.test(lines[index]) &&
      !bullet.test(lines[index]) &&
      !numbered.test(lines[index])
    ) {
      paragraph.push(lines[index])
      index += 1
    }
    blocks.push(
      <p key={key++} className="my-3 leading-relaxed first:mt-0 last:mb-0">
        {inline(paragraph.join(' '))}
      </p>,
    )
  }

  return blocks
}

function CodeBlock({ language, code }: { language: string; code: string }) {
  return (
    <figure className="my-3 overflow-hidden rounded-lg border border-line bg-sunken">
      {language && (
        <figcaption className="border-b border-line px-3 py-1.5 font-mono text-[11px] tracking-wide text-muted uppercase">
          {language}
        </figcaption>
      )}
      {/* The only element allowed to scroll sideways; the page never does. */}
      <pre className="overflow-x-auto p-3 text-[13px] leading-relaxed">
        <code className="font-mono">{code}</code>
      </pre>
    </figure>
  )
}

/** Inline spans: `code`, **bold**, *italic*, and [links](url). */
function inline(text: string): ReactNode[] {
  const nodes: ReactNode[] = []
  const pattern =
    /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\n]+\*)|(\[[^\]]+\]\([^)\s]+\))/g

  let cursor = 0
  let key = 0
  let match: RegExpExecArray | null

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) nodes.push(text.slice(cursor, match.index))
    const token = match[0]

    if (token.startsWith('`')) {
      nodes.push(
        <code
          key={key++}
          className="rounded bg-sunken px-1.5 py-0.5 font-mono text-[0.9em] text-accent-soft"
        >
          {token.slice(1, -1)}
        </code>,
      )
    } else if (token.startsWith('**')) {
      nodes.push(
        <strong key={key++} className="font-semibold text-fg">
          {token.slice(2, -2)}
        </strong>,
      )
    } else if (token.startsWith('*')) {
      nodes.push(<em key={key++}>{token.slice(1, -1)}</em>)
    } else {
      const link = /^\[([^\]]+)\]\(([^)\s]+)\)$/.exec(token)
      if (link) nodes.push(renderLink(link[1], link[2], key++))
      else nodes.push(token)
    }

    cursor = match.index + token.length
  }

  if (cursor < text.length) nodes.push(text.slice(cursor))
  return nodes
}

function renderLink(label: string, href: string, key: number): ReactNode {
  // Only http(s). A `javascript:` URL from a model reply is exactly the kind
  // of thing that should render as text rather than become clickable.
  const safe = /^https?:\/\//i.test(href)
  if (!safe) return <span key={key}>{label}</span>
  return (
    <a
      key={key}
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-accent underline underline-offset-2 hover:text-accent-soft"
    >
      {label}
    </a>
  )
}
