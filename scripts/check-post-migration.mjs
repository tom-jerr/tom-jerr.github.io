import assert from "node:assert/strict"
import fs from "node:fs/promises"
import path from "node:path"
import { parse } from "yaml"
import { slugifyFilePath, simplifySlug } from "@quartz-community/utils"
import { moves } from "./migrate-blog-posts.mjs"

for (const [source, destination] of moves) {
  const oldExists = await fs.access(path.join("content", source)).then(
    () => true,
    () => false,
  )
  assert.equal(oldExists, false, `Article still at old location: ${source}`)
  const content = await fs.readFile(path.join("content", destination), "utf8")
  const frontmatter = parse(content.match(/^---\r?\n([\s\S]*?)\r?\n---/)[1])
  assert.ok(frontmatter.aliases.includes(source.replace(/\.md$/, "")))
  assert.ok(frontmatter.tags?.length, `Missing tags: ${destination}`)
  const alias = `${slugifyFilePath(source)}.html`
  const html = await fs.readFile(path.join("public", alias), "utf8")
  const redirect = html.match(/rel="canonical" href="([^"]+)"/)[1]
  const target = new URL(redirect, `https://example.com/${alias}`).pathname
  assert.equal(decodeURIComponent(target), `/${simplifySlug(slugifyFilePath(destination))}`)
  for (const image of new Set(content.match(/img\/blog-posts\/[^\s)"'<>]+/g) ?? [])) {
    await fs.access(path.join("content", path.dirname(destination), image))
  }
}
assert.equal(
  (await fs.readdir("content/blogs/posts")).filter((name) => name.endsWith(".md")).length,
  0,
)
console.log(
  `Verified ${moves.length} moved posts, their tags, local images and generated old-URL redirects.`,
)
