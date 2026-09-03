import type {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "@quartz-community/types"
import style from "./styles/utterances.scss"
// @ts-expect-error The plugin build turns inline scripts into browser JavaScript strings.
import script from "./scripts/utterances.inline"

export interface UtterancesCommentsOptions {
  repo: `${string}/${string}`
  issueTerm?: "pathname" | "url" | "title" | "og:title"
  label?: string
  lightTheme?: string
  darkTheme?: string
}

export default ((opts?: UtterancesCommentsOptions) => {
  const options: UtterancesCommentsOptions = {
    repo: "tom-jerr/blog-comments",
    issueTerm: "pathname",
    label: "comments",
    lightTheme: "github-light",
    darkTheme: "github-dark",
    ...opts,
  }

  const UtterancesComments: QuartzComponent = ({
    displayClass,
    fileData,
  }: QuartzComponentProps) => {
    const comments = fileData.frontmatter?.comments
    if (comments === false || comments === "false") return <></>

    return (
      <div
        class={[displayClass, "utterances-comments"].filter(Boolean).join(" ")}
        data-repo={options.repo}
        data-issue-term={options.issueTerm}
        data-label={options.label}
        data-light-theme={options.lightTheme}
        data-dark-theme={options.darkTheme}
      />
    )
  }

  UtterancesComments.css = style
  UtterancesComments.afterDOMLoaded = script
  return UtterancesComments
}) satisfies QuartzComponentConstructor<UtterancesCommentsOptions>
