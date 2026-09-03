import assert from "node:assert/strict"
import { test } from "node:test"
import { explorerOptions, recentNotesOptions } from "../quartz.navigation"

const node = (slug: string, isFolder = true) => ({
  slugSegment: slug.split("/").at(-1),
  slugSegments: slug.split("/"),
  displayName: slug,
  isFolder,
  data: null,
  children: [],
})

test("Explorer filters, maps and orders only the five requested roots", () => {
  // Reconstruct exactly as Explorer does in the browser: no lexical closure.
  const filter = new Function("node", `return (${explorerOptions.filterFn!.toString()})(node)`)
  const sort = new Function("a", "b", `return (${explorerOptions.sortFn!.toString()})(a,b)`)
  const map = new Function("node", `return (${explorerOptions.mapFn!.toString()})(node)`)
  const roots = [
    "notes",
    "blogs",
    "tags",
    "sglang",
    "about",
    "cuda",
    "index",
    "llm_inference",
    "gpu",
  ]
    .map((slug) => node(slug))
    .filter((n) => filter(n))
    .sort((a, b) => sort(a, b))
    .map((n) => map(n))
  assert.deepEqual(
    roots.map((n) => n.displayName),
    ["cuda", "gpu", "llm_inference", "sglang", "notes"],
  )
  assert.equal(filter(node("notes/nebula-vsearch/article", false)), true)
  assert.equal(filter(node("nebula-vsearch/article", false)), false)
})

test("Recently updated notes use the same section boundary", () => {
  const filter = recentNotesOptions.filter!
  assert.equal(filter({ slug: "sglang/hicache" } as Parameters<typeof filter>[0]), true)
  assert.equal(filter({ slug: "academy" } as Parameters<typeof filter>[0]), false)
  assert.equal(filter({ slug: "blogs/index" } as Parameters<typeof filter>[0]), false)
})
