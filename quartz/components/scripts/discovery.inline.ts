import { folderColor } from "../../util/discovery"

document.addEventListener("nav", () => {
  const decorate = () => {
    document
      .querySelectorAll<HTMLElement>(".folder-container[data-folderpath]")
      .forEach((folder) => {
        folder.style.setProperty("--folder-color", folderColor(folder.dataset.folderpath!))
        const title = folder.querySelector(".folder-title")
        if (title && !folder.querySelector(".folder-glyph")) {
          const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg")
          icon.setAttribute("viewBox", "0 0 24 24")
          icon.setAttribute("class", "folder-glyph")
          icon.setAttribute("aria-hidden", "true")
          icon.innerHTML =
            '<path d="M3 5a2 2 0 0 1 2-2h5l3 3h6a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/>'
          title.before(icon)
        }
      })
  }
  document.querySelectorAll(".explorer-ul").forEach((tree) => {
    const observer = new MutationObserver(decorate)
    observer.observe(tree, { childList: true, subtree: true })
    window.addCleanup(() => observer.disconnect())
  })
  decorate()
  const taxonomy = document.querySelector<HTMLElement>(".sidebar.left > .taxonomy")
  if (taxonomy) {
    const resize = new ResizeObserver(() => {
      taxonomy.parentElement!.style.setProperty(
        "--mobile-taxonomy-height",
        `${taxonomy.offsetHeight}px`,
      )
    })
    resize.observe(taxonomy)
    window.addCleanup(() => resize.disconnect())
  }
  document.querySelectorAll<HTMLImageElement>(".article-cover img").forEach((img) => {
    const hide = () => {
      img.closest<HTMLElement>(".article-cover")!.hidden = true
    }
    img.addEventListener("error", hide)
    if (img.complete && !img.naturalWidth) hide()
    window.addCleanup(() => img.removeEventListener("error", hide))
  })
})
