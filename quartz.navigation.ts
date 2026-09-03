import type { ExplorerOptions } from "@quartz-community/explorer"
import type { RecentNotesOptions } from "@quartz-community/recent-notes"

// Explorer sends these functions to the browser as source strings. Keep each
// function self-contained: module-level constants are not available there.
export const explorerOptions: Partial<ExplorerOptions> = {
  filterFn: (node) =>
    ["cuda", "gpu", "llm_inference", "sglang", "notes"].includes(node.slugSegments?.[0] ?? ""),
  mapFn: (node) => {
    if (node.slugSegments?.length === 1) node.displayName = node.slugSegment
    return node
  },
  sortFn: (a, b) => {
    if (a.slugSegments?.length === 1 && b.slugSegments?.length === 1) {
      const sections = ["cuda", "gpu", "llm_inference", "sglang", "notes"]
      return sections.indexOf(a.slugSegment ?? "") - sections.indexOf(b.slugSegment ?? "")
    }
    if (a.isFolder !== b.isFolder) return a.isFolder ? -1 : 1
    return (a.displayName ?? "").localeCompare(b.displayName ?? "", "zh-CN", {
      numeric: true,
      sensitivity: "base",
    })
  },
}

export const recentNotesOptions: Partial<RecentNotesOptions> = {
  filter: (file) =>
    ["cuda", "gpu", "llm_inference", "sglang", "notes"].includes((file.slug ?? "").split("/")[0]),
}
