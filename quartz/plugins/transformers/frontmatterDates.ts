import type { QuartzTransformerPlugin } from "../types"

type Frontmatter = Record<string, unknown>

function isPresent(value: unknown): boolean {
  return value !== undefined && value !== null && value !== ""
}

/** Normalize date aliases before CreatedModifiedDate reads the frontmatter. */
export function normalizeFrontmatterDates(frontmatter: Frontmatter): void {
  if (isPresent(frontmatter.modified)) return

  if (isPresent(frontmatter.updated)) {
    frontmatter.modified = frontmatter.updated
  } else if (isPresent(frontmatter.update)) {
    frontmatter.modified = frontmatter.update
  }
}

export const FrontmatterDateAliases: QuartzTransformerPlugin = () => ({
  name: "FrontmatterDateAliases",
  markdownPlugins() {
    return [
      () => (_tree, file) => {
        const frontmatter = file.data.frontmatter as Frontmatter | undefined
        if (frontmatter) normalizeFrontmatterDates(frontmatter)
      },
    ]
  },
})
