import { i18n, ValidLocale } from "../i18n"
import { classNames } from "../util/lang"
import type { QuartzComponent, QuartzComponentProps } from "./types"

function formatDate(date: Date, locale: ValidLocale): string {
  return date.toLocaleDateString(locale, {
    year: "numeric",
    month: "short",
    day: "2-digit",
  })
}

function isInRanges(code: number, ranges: [number, number][]): boolean {
  return ranges.some(([start, end]) => start <= code && code <= end)
}

function isCjk(char: string | undefined): boolean {
  if (!char) return false
  return isInRanges(char.charCodeAt(0), [
    [12352, 12447],
    [19968, 40959],
    [44032, 55203],
  ])
}

function isPunctuation(char: string | undefined): boolean {
  if (!char) return false
  return isInRanges(char.charCodeAt(0), [
    [33, 47],
    [58, 64],
    [91, 96],
    [123, 126],
    [12288, 12351],
    [65280, 65519],
  ])
}

function readingMinutes(text: string): number {
  const isBoundary = (char: string | undefined) => char === undefined || " \n\r\t".includes(char)
  let words = 0

  for (let index = 0; index < text.length; index++) {
    const current = text[index]
    const next = text[index + 1]
    if (isCjk(current) || (!isBoundary(current) && (isBoundary(next) || isCjk(next)))) words++
    if (isCjk(current)) {
      while (
        index < text.length &&
        (isPunctuation(text[index + 1]) || isBoundary(text[index + 1]))
      ) {
        index++
      }
    }
  }

  return Math.max(1, Math.ceil(words / 200))
}

export default (() => {
  const ArticleMeta: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
    if (!fileData.text || fileData.slug === "index") return null

    const locale = cfg.locale ?? "en-US"
    const created = fileData.dates?.created
    const modified = fileData.dates?.modified
    const items = []

    if (created) {
      items.push(
        <span class="article-meta-item">
          <span class="article-meta-label">创建于</span>
          <time datetime={created.toISOString()}>{formatDate(created, locale)}</time>
        </span>,
      )
    }
    if (modified) {
      items.push(
        <span class="article-meta-item">
          <span class="article-meta-label">更新于</span>
          <time datetime={modified.toISOString()}>{formatDate(modified, locale)}</time>
        </span>,
      )
    }

    items.push(
      <span class="article-meta-item">
        {i18n(locale).components.contentMeta.readingTime({
          minutes: readingMinutes(fileData.text),
        })}
      </span>,
    )

    return <p class={classNames(displayClass, "article-meta")}>{items}</p>
  }

  ArticleMeta.css = `
.article-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem 0;
  margin-top: 0;
  color: var(--darkgray);
}

.article-meta-item {
  white-space: nowrap;
}

.article-meta-item:not(:last-child)::after {
  content: "·";
  margin: 0 0.55rem;
  color: var(--gray);
}

.article-meta-label {
  margin-right: 0.3rem;
}
`

  return ArticleMeta
}) satisfies () => QuartzComponent
