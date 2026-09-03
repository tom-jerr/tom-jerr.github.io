// Explicit, idempotent migration. Preview with node scripts/migrate-blog-posts.mjs;
// pass --apply to relocate the listed articles and copy only their referenced assets.
import fs from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { parseDocument } from "yaml"
import { slugifyFilePath } from "@quartz-community/utils"

export const moves = [
  [
    "上篇：初识 Nebula Graph —— 向量类型支持.md",
    "notes/nebula-vsearch/上篇：初识 Nebula Graph —— 向量类型支持.md",
  ],
  [
    "中篇：Vector 类型的 DDL&DML 适配.md",
    "notes/nebula-vsearch/中篇：Vector 类型的 DDL&DML 适配.md",
  ],
  [
    "下篇：向量索引与相似度搜索 —— Nebula Graph 的 ANN 实现之路.md",
    "notes/nebula-vsearch/下篇：向量索引与相似度搜索 —— Nebula Graph 的 ANN 实现之路.md",
  ],
  ["bision debug.md", "notes/nebula-vsearch/bison-debug.md"],
  ["bustub通关指北.md", "notes/CMU15445/bustub通关指北.md"],
  ["C++异步方案.md", "notes/C++/C++异步方案.md"],
  ["Implement of Concurrent.md", "notes/C++Concurrency/Implement of Concurrent.md"],
  ["CS144.md", "notes/CS144/CS144.md"],
  ["FlashAttention 原理 v1-v2.md", "llm_inference/FlashAttention 原理 v1-v2.md"],
  ["PageAttention.md", "llm_inference/PageAttention.md"],
].map(([name, destination]) => [`blogs/posts/${name}`, destination])

const root = fileURLToPath(new URL("../content/", import.meta.url))
const apply = process.argv.includes("--apply")
const safePath = (relative) => {
  const result = path.resolve(root, relative)
  const checked = path.relative(root, result)
  if (!checked || checked.startsWith("..") || path.isAbsolute(checked)) {
    throw new Error(`Target is outside content: ${relative}`)
  }
  return result
}
const exists = async (file) =>
  fs.access(file).then(
    () => true,
    () => false,
  )
const slash = (value) => value.replaceAll(path.sep, "/")
const encodePath = (value) => value.split("/").map(encodeURIComponent).join("/")
const assets = new Map()
const planned = []

// Preflight every source, destination and image before moving anything.
for (const [source, destination] of moves) {
  const from = safePath(source)
  const to = safePath(destination)
  if (!(await exists(from))) {
    if (!(await exists(to))) throw new Error(`Missing source and destination: ${source}`)
    continue
  }
  if (await exists(to)) throw new Error(`Refusing to overwrite article: ${destination}`)
  let text = await fs.readFile(from, "utf8")
  const fm = text.match(/^---\r?\n([\s\S]*?)\r?\n---/)
  if (!fm) throw new Error(`Missing frontmatter: ${source}`)
  const doc = parseDocument(fm[1])
  if (doc.errors.length) throw doc.errors[0]
  const previous = doc.toJS().aliases ?? []
  const aliases = Array.isArray(previous) ? previous : [previous]
  doc.set("aliases", [...new Set([...aliases, source.replace(/\.md$/, "")])])
  text = `---\n${doc.toString()}---${text.slice(fm[0].length)}`
  // These legacy articles use simple img/ paths in Markdown, HTML and cover metadata.
  const references = [
    ...new Set(text.match(/(?<![\/\w])img\/[^\s)"'<>]+\.(?:png|jpg|jpeg|gif|svg|webp)/gi) ?? []),
  ]
  if (text.includes("(flashattentionv2.png)")) references.push("flashattentionv2.png")
  for (const reference of references) {
    let assetFrom = safePath(path.join(path.dirname(source), reference))
    // Former theme covers were rooted in overrides/img, now migrated to content/img.
    if (!(await exists(assetFrom)) && reference === doc.toJS().cover) {
      assetFrom = safePath(reference)
    }
    const newReference = `img/blog-posts/${path.basename(reference)}`
    const relativeTarget = path.join(path.dirname(destination), newReference)
    const assetTo = safePath(relativeTarget)
    const bytes = await fs.readFile(assetFrom)
    if (await exists(assetTo)) {
      if (!bytes.equals(await fs.readFile(assetTo)))
        throw new Error(`Conflicting asset: ${relativeTarget}`)
    }
    if (assets.has(assetTo) && !assets.get(assetTo).bytes.equals(bytes)) {
      throw new Error(`Conflicting planned asset: ${relativeTarget}`)
    }
    assets.set(assetTo, { from: assetFrom, bytes })
    text = text.replaceAll(reference, newReference)
  }
  planned.push({ source, destination, from, to, text })
}

async function markdownFiles(directory) {
  const result = []
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    const absolute = path.join(directory, entry.name)
    if (entry.isDirectory()) result.push(...(await markdownFiles(absolute)))
    else if (entry.name.endsWith(".md")) result.push(absolute)
  }
  return result
}

function rewriteLinks(text, relativeFile) {
  for (const [source, destination] of moves) {
    const oldRelative = slash(path.relative(path.dirname(relativeFile), source))
    const newRelative = slash(path.relative(path.dirname(relativeFile), destination))
    text = text
      .replaceAll(`(${oldRelative})`, `(<${newRelative}>)`)
      .replaceAll(`(${encodePath(oldRelative)})`, `(<${newRelative}>)`)
    const legacyRoute = source.replace(/\.md$/, "")
    for (const oldRoute of [legacyRoute, encodePath(legacyRoute), slugifyFilePath(source)]) {
      text = text.replaceAll(`https://tom-jerr.github.io/${oldRoute}/`, `<${newRelative}>`)
    }
    // Quartz's relative link resolver expects Markdown file paths, not root slugs.
    text = text.replaceAll(`(/${slugifyFilePath(destination)})`, `(<${newRelative}>)`)
    // Angle destinations preserve spaces and '&' for Quartz's slug normalization.
    text = text.replaceAll(`(${encodePath(newRelative)})`, `(<${newRelative}>)`)
  }
  return text
}

for (const item of planned) item.text = rewriteLinks(item.text, item.destination)
const updates = []
for (const file of await markdownFiles(root)) {
  if (planned.some((item) => item.from === file)) continue
  const before = await fs.readFile(file, "utf8")
  const after = rewriteLinks(before, slash(path.relative(root, file)))
  if (after !== before) updates.push({ file, text: after })
}

console.log(
  `${apply ? "Apply" : "Preview"}: ${planned.length} articles, ${assets.size} referenced assets, ${updates.length} incoming-link updates.`,
)
for (const { source, destination } of planned) console.log(`${source} -> ${destination}`)
if (apply) {
  for (const [to, { from }] of assets) {
    await fs.mkdir(path.dirname(to), { recursive: true })
    if (!(await exists(to))) await fs.copyFile(from, to, fs.constants.COPYFILE_EXCL)
  }
  for (const item of planned) {
    await fs.mkdir(path.dirname(item.to), { recursive: true })
    await fs.rename(item.from, item.to)
    await fs.writeFile(item.to, item.text)
  }
  for (const { file, text } of updates) await fs.writeFile(file, text)
}
