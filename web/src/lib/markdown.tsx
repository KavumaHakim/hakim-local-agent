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

    // A table: a row of cells, then the separator row that makes it one.
    // Checked before lists and paragraphs because a pipe row is neither.
    if (isTableStart(lines, index)) {
      const [table, next] = parseTable(lines, index)
      blocks.push(<Table key={key++} {...table} />)
      index = next
      continue
    }

    if (LIST_ITEM.test(line)) {
      // Nested by indentation. A flat regex used to swallow the indent and
      // render every level as one list, which is what 13 of 86 stored
      // replies looked like.
      const [list, next] = parseList(lines, index)
      blocks.push(renderList(list, key++))
      index = next
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
      !LIST_ITEM.test(lines[index]) &&
      !isTableStart(lines, index)
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

// --- lists ------------------------------------------------------------------

/** Any list item: its indent, its marker, and the text after it. */
const LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/

interface ListNode {
  ordered: boolean
  items: { text: string; children: ListNode | null }[]
}

/**
 * Parse consecutive list lines from `start` into a tree, nesting by indent.
 *
 * A deeper indent than the current item opens a child list under it; a
 * shallower one closes back out. Two spaces is enough to count as deeper,
 * which is what models write. Returns the tree and the index after it.
 */
function parseList(lines: string[], start: number): [ListNode, number] {
  let index = start

  function parseLevel(indent: number): ListNode {
    const first = LIST_ITEM.exec(lines[index])!
    const node: ListNode = { ordered: /\d/.test(first[2]), items: [] }

    while (index < lines.length) {
      const match = LIST_ITEM.exec(lines[index])
      if (!match) break
      const depth = match[1].length
      if (depth < indent) break // belongs to an outer list
      if (depth > indent) {
        // Deeper than this level: it is a child of the item just added. A
        // deeper first line with nothing above it is treated as this level.
        const last = node.items[node.items.length - 1]
        if (!last) {
          node.items.push({ text: match[3], children: null })
          index += 1
          continue
        }
        last.children = parseLevel(depth)
        continue
      }
      node.items.push({ text: match[3], children: null })
      index += 1
    }
    return node
  }

  const indent = LIST_ITEM.exec(lines[start])![1].length
  return [parseLevel(indent), index]
}

function renderList(node: ListNode, key: number, nested = false): ReactNode {
  const List = node.ordered ? 'ol' : 'ul'
  return (
    <List
      key={key}
      className={`${nested ? 'mt-1' : 'my-3'} space-y-1 pl-5 ${
        node.ordered ? 'list-decimal' : 'list-disc'
      } marker:text-muted`}
    >
      {node.items.map((item, position) => (
        <li key={position}>
          {inline(item.text)}
          {item.children && renderList(item.children, position, true)}
        </li>
      ))}
    </List>
  )
}

// --- tables -----------------------------------------------------------------

const TABLE_ROW = /^\s*\|.*\|\s*$/
// The row under the header: cells of dashes, with optional colons that set
// the alignment. This row is what makes a run of pipe lines a table.
const TABLE_SEPARATOR = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/

type Align = 'left' | 'center' | 'right'

interface TableData {
  header: string[]
  align: Align[]
  rows: string[][]
}

function isTableStart(lines: string[], index: number): boolean {
  return (
    index + 1 < lines.length &&
    TABLE_ROW.test(lines[index]) &&
    TABLE_SEPARATOR.test(lines[index + 1])
  )
}

/** Split a pipe row into cells. `\|` inside a cell is a literal pipe. */
function splitRow(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '')
  return trimmed
    .split(/(?<!\\)\|/)
    .map((cell) => cell.replace(/\\\|/g, '|').trim())
}

function parseTable(lines: string[], start: number): [TableData, number] {
  const header = splitRow(lines[start])
  const align = splitRow(lines[start + 1]).map((cell): Align => {
    const left = cell.startsWith(':')
    const right = cell.endsWith(':')
    if (left && right) return 'center'
    if (right) return 'right'
    return 'left'
  })

  const rows: string[][] = []
  let index = start + 2
  while (index < lines.length && TABLE_ROW.test(lines[index])) {
    // Ragged rows are padded or cut to the header, so a model that miscounts
    // pipes on one line does not shift every cell after it.
    const cells = splitRow(lines[index])
    rows.push(header.map((_, column) => cells[column] ?? ''))
    index += 1
  }
  return [{ header, align, rows }, index]
}

const ALIGN_CLASS: Record<Align, string> = {
  left: 'text-left',
  center: 'text-center',
  right: 'text-right',
}

function Table({ header, align, rows }: TableData) {
  return (
    // Wide tables scroll inside their own box; the page never does.
    <div className="my-3 overflow-x-auto rounded-lg border border-line">
      <table className="w-full border-collapse text-[13px]">
        <thead className="bg-sunken">
          <tr>
            {header.map((cell, column) => (
              <th
                key={column}
                className={`border-b border-line px-3 py-1.5 font-semibold text-fg ${ALIGN_CLASS[align[column] ?? 'left']}`}
              >
                {inline(cell)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, position) => (
            <tr key={position} className="border-b border-line last:border-b-0">
              {row.map((cell, column) => (
                <td
                  key={column}
                  className={`px-3 py-1.5 align-top ${ALIGN_CLASS[align[column] ?? 'left']}`}
                >
                  {inline(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
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
