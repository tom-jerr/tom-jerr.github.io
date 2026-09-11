import type { QuartzComponent, QuartzComponentProps } from "./types"
import { Date, getDate } from "./Date"
import { FullSlug, resolveRelative, slugTag } from "../util/path"
import { articleDescription, articleImage, categories, categorySlug } from "../util/discovery"
import { byDateAndAlphabetical } from "./PageList"

const assetCache = new WeakMap<FullSlug[], Set<string>>()

export const ArticleCards: QuartzComponent = (props: QuartzComponentProps) => {
  const { fileData, allFiles, cfg } = props
  let assets = assetCache.get(props.ctx.allSlugs)
  if (!assets) {
    assets = new Set(props.ctx.allSlugs)
    assetCache.set(props.ctx.allSlugs, assets)
  }
  return (
    <ul class="article-cards section-ul">
      {[...allFiles].sort(byDateAndAlphabetical()).map((page) => {
        const href = resolveRelative(fileData.slug!, page.slug!)
        const image = articleImage(page, fileData.slug!, assets)
        const description = articleDescription(page)
        return (
          <li class="article-card section-li" key={page.slug}>
            {image && (
              <a class="article-cover internal" href={href} tabIndex={-1} aria-hidden="true">
                <img src={image} alt="" loading="lazy" decoding="async" />
              </a>
            )}
            <div class="article-card-body desc">
              <h3>
                <a class="internal" href={href}>
                  {page.frontmatter?.title}
                </a>
              </h3>
              {page.dates && (
                <p class="meta">
                  <Date date={getDate(page)!} locale={cfg.locale} />
                </p>
              )}
              {description && <p class="article-description">{description}</p>}
              <ul class="tags">
                {(page.frontmatter?.tags ?? []).map((tag) => (
                  <li>
                    <a
                      class="internal tag-link"
                      href={resolveRelative(fileData.slug!, `tags/${slugTag(tag)}` as FullSlug)}
                    >
                      #{tag}
                    </a>
                  </li>
                ))}
                {categories(page).map((category) => (
                  <li>
                    <a
                      class="internal category-link"
                      href={resolveRelative(fileData.slug!, categorySlug(category))}
                    >
                      {category}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </li>
        )
      })}
    </ul>
  )
}
