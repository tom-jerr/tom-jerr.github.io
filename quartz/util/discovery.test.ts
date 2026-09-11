import assert from "node:assert/strict"
import { test } from "node:test"
import { articleImage, categories, categorySlug, isArticle } from "./discovery"
import type { QuartzPluginData } from "../plugins/vfile"
import type { FullSlug } from "./path"

const file = (frontmatter: Record<string, unknown> = {}) =>
  ({
    slug: "notes/project/post" as FullSlug,
    frontmatter: { title: "Post", tags: [], ...frontmatter },
  }) as QuartzPluginData

test("taxonomy accepts scalar and list categories without conflating punctuation", () => {
  assert.deepEqual(categories(file({ categories: "Project" })), ["Project"])
  assert.deepEqual(categories(file({ categories: ["CUDA", "CUDA", " "] })), ["CUDA"])
  assert.notEqual(categorySlug("C++"), categorySlug("C#"))
  assert.ok(!/[#%/]/.test(categorySlug("中文/分类").slice("categories/".length)))
})

test("article covers resolve from the source article, including parent and root paths", () => {
  assert.equal(
    articleImage(file({ cover: "img/kernel figure.png" }), "tags/cuda" as FullSlug),
    "../notes/project/img/kernel-figure.png",
  )
  assert.equal(
    articleImage(file({ cover: "../img/kernel.png" }), "index" as FullSlug),
    "./notes/img/kernel.png",
  )
  assert.equal(
    articleImage(file({ cover: "/static/cover.png" }), "index" as FullSlug),
    "./static/cover.png",
  )
  assert.equal(
    articleImage(file({ cover: "https://example.com/cover.png" }), "index" as FullSlug),
    "https://example.com/cover.png",
  )
  assert.equal(articleImage(file(), "index" as FullSlug), undefined)
})

test("article listings exclude unlisted, draft and generated index pages", () => {
  assert.equal(isArticle(file()), true)
  assert.equal(isArticle({ ...file(), unlisted: true }), false)
  assert.equal(isArticle(file({ draft: true })), false)
  assert.equal(isArticle({ ...file(), slug: "notes/index" as FullSlug }), false)
})

test("broken covers and legacy local images fall back to an existing body image", () => {
  const page = file({ cover: "img/missing.png" })
  page.htmlAst = {
    type: "root",
    children: [
      {
        type: "element",
        tagName: "img",
        properties: { src: "C:\\Users\\me\\diagram.png" },
        children: [],
      },
      { type: "element", tagName: "img", properties: { src: "./img/diagram.svg" }, children: [] },
    ],
  }
  assert.equal(
    articleImage(page, "tags/cuda" as FullSlug, new Set(["notes/project/img/diagram.svg"])),
    "../notes/project/img/diagram.svg",
  )
  assert.equal(articleImage(page, "index" as FullSlug, new Set()), undefined)
})
