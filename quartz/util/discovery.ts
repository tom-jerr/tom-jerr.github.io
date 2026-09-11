import type { QuartzPluginData } from "../plugins/vfile"
import { FullSlug, FilePath, resolveRelative, slugifyFilePath } from "./path"
import type { Root, Element } from "hast"

export const folderColors = ["#6274d9", "#299b83", "#c18a32", "#ad68bd", "#d17463", "#419bb5"]
const roots = ["cuda", "gpu", "llm_inference", "sglang", "notes"]

export function folderColor(folder: string) {
  const root = folder.split("/")[0]
  const index = roots.indexOf(root)
  const hash = Array.from(root).reduce((n, c) => n + c.charCodeAt(0), 0)
  return folderColors[(index < 0 ? hash : index) % folderColors.length]
}

export function articleFolder(slug: string) {
  return slug.includes("/") ? slug.slice(0, slug.lastIndexOf("/")) : "其他"
}

export function isArticle(file: QuartzPluginData) {
  const slug = file.slug ?? ""
  return Boolean(
    slug &&
    slug !== "index" &&
    !slug.endsWith("/index") &&
    !/^(tags|categories)(\/|$)/.test(slug) &&
    !["404", "academy"].includes(slug) &&
    file.frontmatter?.draft !== true &&
    file.unlisted !== true,
  )
}

export function labels(value: unknown): string[] {
  const values = Array.isArray(value) ? value : typeof value === "string" ? [value] : []
  return [
    ...new Set(
      values
        .filter((v): v is string => typeof v === "string")
        .map((v) => v.trim())
        .filter(Boolean),
    ),
  ]
}

export function categories(file: QuartzPluginData) {
  return labels(file.frontmatter?.categories ?? file.frontmatter?.category)
}

// Encoding each segment preserves distinctions such as “C++” and “C#”.
export function categorySlug(category: string) {
  return `categories/${encodeURIComponent(category).replace(/~/g, "%7E").replace(/%/g, "~")}` as FullSlug
}

export function articleDescription(file: QuartzPluginData) {
  const text = String(file.frontmatter?.description ?? file.description ?? "")
  // The Description plugin HTML-escapes its generated summary. JSX escapes again.
  const entities: Record<string, string> = {
    amp: "&",
    lt: "<",
    gt: ">",
    quot: '"',
    apos: "'",
    "#39": "'",
    nbsp: " ",
  }
  return (
    file.frontmatter?.description
      ? text
      : text.replace(/&(amp|lt|gt|quot|apos|#39|nbsp);/g, (_, name: string) => entities[name])
  )
    .replace(/\s+/g, " ")
    .trim()
}

export function articleImage(
  file: QuartzPluginData,
  current: FullSlug,
  assets?: ReadonlySet<string>,
): string | undefined {
  const cover = file.frontmatter?.cover ?? file.frontmatter?.image
  const candidates: { source: string; explicit: boolean }[] = []
  if (typeof cover === "string") candidates.push({ source: cover, explicit: true })
  if (file.htmlAst) {
    const walk = (node: Root | Element) => {
      for (const child of node.children) {
        if (child.type !== "element") continue
        if (
          child.tagName === "img" &&
          typeof child.properties.src === "string" &&
          !/badge|shields\.io|favicon|avatar/i.test(child.properties.src)
        )
          candidates.push({ source: child.properties.src, explicit: false })
        walk(child)
      }
    }
    walk(file.htmlAst)
  }
  for (const { source, explicit } of candidates) {
    if (!source) continue
    if (/^(https?:)?\/\//i.test(source)) return source
    if (/^[a-z][a-z\d+.-]*:/i.test(source) || source.includes("\\")) continue
    // Body images have already passed through CrawlLinks; covers have not.
    const normalized = explicit
      ? (source.startsWith("/") ? "/" : "") + slugifyFilePath(source as FilePath)
      : source
    try {
      const url = new URL(normalized, `https://quartz.local/${file.slug}`)
      const target = decodeURIComponent(url.pathname.slice(1)) as FullSlug
      if (assets && !assets.has(target)) continue
      return resolveRelative(current, target) + url.search + url.hash
    } catch {
      /* Ignore malformed legacy image URLs and try the next image. */
    }
  }
}
