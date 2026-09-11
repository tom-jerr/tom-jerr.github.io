import type { QuartzComponent } from "./types"
import { FullSlug, resolveRelative, slugTag } from "../util/path"
import { categories, categorySlug, isArticle, labels } from "../util/discovery"
import { ArticleCards } from "./ArticleCards"
import { htmlToJsx } from "../util/jsx"
import { byDateAndAlphabetical } from "./PageList"
// @ts-ignore: Quartz bundles inline scripts as strings.
import script from "./scripts/discovery.inline"

export const Taxonomy: QuartzComponent = ({ allFiles, fileData }) => {
  const files = allFiles.filter(isArticle)
  const groups = [
    {
      title: "Tags · 标签",
      values: (f: typeof fileData) => labels(f.frontmatter?.tags),
      slug: (s: string) => `tags/${slugTag(s)}` as FullSlug,
      index: "tags/index",
    },
    {
      title: "Categories · 分类",
      values: categories,
      slug: categorySlug,
      index: "categories/index",
    },
  ]
  return (
    <nav class="taxonomy" aria-label="标签与分类">
      {groups.map((group) => {
        const counts = new Map<string, number>()
        files.forEach((file) =>
          group.values(file).forEach((value) => counts.set(value, (counts.get(value) ?? 0) + 1)),
        )
        return (
          <details open>
            <summary>
              {group.title} <span>{counts.size}</span>
            </summary>
            <div class="taxonomy-labels">
              {[...counts]
                .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
                .map(([label, count]) => (
                  <a
                    class="internal taxonomy-chip"
                    href={resolveRelative(fileData.slug!, group.slug(label))}
                    title={`${label} · ${count} 篇文章`}
                  >
                    {label}
                    <small>{count}</small>
                  </a>
                ))}
              {!counts.size && <span class="meta">暂无分类</span>}
            </div>
            <a
              class="internal taxonomy-all"
              href={resolveRelative(fileData.slug!, group.index as FullSlug)}
            >
              查看全部 →
            </a>
          </details>
        )
      })}
    </nav>
  )
}
Taxonomy.afterDOMLoaded = script

export const RecentCards: QuartzComponent = (props) => {
  const files = props.allFiles
    .filter(
      (f) =>
        isArticle(f) &&
        ["cuda", "gpu", "llm_inference", "sglang", "notes"].includes(f.slug!.split("/")[0]),
    )
    .sort(byDateAndAlphabetical())
    .slice(0, 8)
  return (
    <section class="recent-notes">
      <h2>最近更新</h2>
      <ArticleCards {...props} allFiles={files} />
    </section>
  )
}

export const DiscoveryListing: QuartzComponent = (props) => {
  const slug = props.fileData.slug!
  const all = props.allFiles.filter(isArticle)
  const tagPage = slug.startsWith("tags/")
  const categoryPage = slug.startsWith("categories/")
  const overview = slug === "tags/index" || slug === "categories/index"
  const key = slug.replace(/^(tags|categories)\//, "").replace(/\/index$/, "")
  const folder = slug.replace(/\/index$/, "")
  const files = overview
    ? all
    : all.filter((file) =>
        tagPage
          ? labels(file.frontmatter?.tags).some(
              (t) => slugTag(t) === key || slugTag(t).startsWith(`${key}/`),
            )
          : categoryPage
            ? categories(file).some((c) => categorySlug(c) === slug)
            : file.slug!.startsWith(`${folder}/`),
      )
  return (
    <div class="discovery-listing popover-hint">
      {!tagPage && !categoryPage && (
        <article>{htmlToJsx(props.fileData.filePath!, props.tree)}</article>
      )}
      {overview && <Taxonomy {...props} />}
      <p class="listing-count">{files.length} 篇文章</p>
      <ArticleCards {...props} allFiles={files} />
      {!files.length && <p>这个标签下还没有文章。</p>}
    </div>
  )
}
