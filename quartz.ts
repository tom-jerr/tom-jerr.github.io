import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import { componentRegistry } from "./quartz/components/registry"
import { explorerOptions, recentNotesOptions } from "./quartz.navigation"
import { Taxonomy, RecentCards, DiscoveryListing } from "./quartz/components/Discovery"
import { FolderGraph } from "./quartz/components/FolderGraph"
import { PageTypeDispatcher } from "./quartz/plugins/pageTypes/dispatcher"
import { categories, categorySlug, isArticle } from "./quartz/util/discovery"
import type { QuartzPageTypePluginInstance } from "./quartz/plugins/types"

// Config loading also constructs the page dispatcher, so overrides must be
// registered before it captures the layouts used by emitted pages.
componentRegistry.setOptionOverrides("@quartz-community/explorer", explorerOptions)
componentRegistry.setOptionOverrides("@quartz-community/recent-notes", recentNotesOptions)
const config = await loadQuartzConfig()
export const layout = await loadQuartzLayout()

for (const section of [layout.defaults, ...Object.values(layout.byPageType)]) {
  if (section.left?.length) section.left = [...section.left, Taxonomy, RecentCards]
  if (section.right?.length) section.right = [FolderGraph, ...section.right]
}
for (const pageType of config.plugins.pageTypes ?? []) {
  if (["FolderPage", "TagPage"].includes(pageType.name)) pageType.body = () => DiscoveryListing
}
const categoryPages: QuartzPageTypePluginInstance = {
  name: "CategoryPage",
  priority: 30,
  match: ({ slug }) => slug.startsWith("categories/"),
  generate: ({ content }) => {
    const names = new Set(
      content
        .map(([, file]) => file.data)
        .filter(isArticle)
        .flatMap(categories),
    )
    return [
      { slug: "categories/index", title: "分类", data: {} },
      ...[...names].map((name) => ({ slug: categorySlug(name), title: `分类：${name}`, data: {} })),
    ]
  },
  layout: "folder",
  body: () => DiscoveryListing,
}
config.plugins.pageTypes!.push(categoryPages)
config.plugins.emitters = config.plugins.emitters.filter(
  (emitter) => emitter.name !== "PageTypeDispatcher",
)
config.plugins.emitters.push(PageTypeDispatcher(layout))
export default config
