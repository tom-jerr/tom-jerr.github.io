// @ts-nocheck

const utterancesTheme = (container) => {
  const theme = document.documentElement.getAttribute("saved-theme") ?? "light"
  return theme === "dark" ? container.dataset.darkTheme : container.dataset.lightTheme
}

const updateUtterancesTheme = (event) => {
  const container = document.querySelector(".utterances-comments")
  const frame = container?.querySelector(".utterances-frame")
  if (!container || !frame?.contentWindow) return

  const theme = event?.detail?.theme ?? document.documentElement.getAttribute("saved-theme")
  frame.contentWindow.postMessage(
    {
      type: "set-theme",
      theme: theme === "dark" ? container.dataset.darkTheme : container.dataset.lightTheme,
    },
    "https://utteranc.es",
  )
}

const setupUtterances = () => {
  const container = document.querySelector(".utterances-comments")
  if (!container) return

  container.replaceChildren()
  const commentsScript = document.createElement("script")
  commentsScript.src = "https://utteranc.es/client.js"
  commentsScript.async = true
  commentsScript.crossOrigin = "anonymous"
  commentsScript.setAttribute("repo", container.dataset.repo)
  commentsScript.setAttribute("issue-term", container.dataset.issueTerm ?? "pathname")
  commentsScript.setAttribute("theme", utterancesTheme(container))
  if (container.dataset.label) commentsScript.setAttribute("label", container.dataset.label)
  container.appendChild(commentsScript)

  document.addEventListener("themechange", updateUtterancesTheme)
  window.addCleanup?.(() => document.removeEventListener("themechange", updateUtterancesTheme))
}

document.addEventListener("nav", setupUtterances)
document.addEventListener("render", setupUtterances)
