import { articleFolder, folderColor } from "../../util/discovery"
import { resolveRelative, FullSlug } from "../../util/path"

type Entry = { title: string; links?: string[]; tags?: string[] }
type GraphNode = Entry & { slug: string; x: number; y: number; folder: string }
const ns = "http://www.w3.org/2000/svg"

document.addEventListener("nav", async () => {
  const root = document.querySelector<HTMLElement>(".folder-graph")
  if (!root) return
  let disposed = false
  const cleanups: (() => void)[] = []
  window.addCleanup(() => {
    disposed = true
    cleanups.forEach((fn) => fn())
  })
  const raw = await fetchData
  if (disposed) return
  const data = ((raw as unknown as { content?: Record<string, Entry> }).content ?? raw) as Record<
    string,
    Entry
  >
  const current = document.body.dataset.slug ?? "index"
  const all = Object.entries(data).filter(
    ([slug]) =>
      slug !== "index" &&
      !slug.endsWith("/index") &&
      !/^(tags|categories)(\/|$)/.test(slug) &&
      !["404", "academy"].includes(slug),
  )
  const currentFolder = articleFolder(current)
  const related = new Set(data[current]?.links ?? [])
  all.forEach(([slug, entry]) => {
    if (entry.links?.includes(current)) related.add(slug)
  })
  const on = (target: EventTarget, event: string, handler: EventListener) => {
    target.addEventListener(event, handler)
    cleanups.push(() => target.removeEventListener(event, handler))
  }
  const draw = (container: HTMLElement, global: boolean) => {
    container.replaceChildren()
    let entries =
      global || current === "index"
        ? all
        : all.filter(
            ([slug]) =>
              articleFolder(slug) === currentFolder || slug === current || related.has(slug),
          )
    // Keep the small home preview readable; the expanded graph includes every folder.
    if (!global && current === "index") {
      const counts = new Map<string, number>()
      entries.forEach(([slug]) =>
        counts.set(articleFolder(slug), (counts.get(articleFolder(slug)) ?? 0) + 1),
      )
      const previewFolders = new Set(
        [...counts]
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)
          .map(([folder]) => folder),
      )
      entries = entries.filter(([slug]) => previewFolders.has(articleFolder(slug)))
    }
    if (!entries.length) {
      container.textContent = "暂无相关文章"
      return
    }
    const groups = new Map<string, typeof entries>()
    entries.forEach((entry) => {
      const folder = articleFolder(entry[0])
      if (!groups.has(folder)) groups.set(folder, [])
      groups.get(folder)!.push(entry)
    })
    const sorted = [...groups].sort(([a], [b]) => a.localeCompare(b))
    const columns = Math.ceil(Math.sqrt(sorted.length))
    const cell = Math.max(
      160,
      Math.sqrt(Math.max(...sorted.map(([, files]) => files.length))) * 30 + 65,
    )
    const width = columns * cell,
      height = Math.ceil(sorted.length / columns) * cell
    const svg = document.createElementNS(ns, "svg")
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`)
    svg.setAttribute("aria-label", "按文件夹聚类的文章引用图，节点可聚焦并打开文章")
    const make = (tag: string, attrs: Record<string, string | number>, parent: Element = svg) => {
      const el = document.createElementNS(ns, tag)
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)))
      parent.appendChild(el)
      return el
    }
    const nodes: GraphNode[] = []
    sorted.forEach(([folder, files], i) => {
      const cx = ((i % columns) + 0.5) * cell,
        cy = (Math.floor(i / columns) + 0.5) * cell
      const radius = cell * 0.42
      make("circle", {
        cx,
        cy,
        r: radius,
        fill: folderColor(folder),
        "fill-opacity": 0.07,
        stroke: folderColor(folder),
        "stroke-opacity": 0.3,
        "stroke-dasharray": "3 5",
        "data-cluster": folder,
      })
      const label = make("text", {
        x: cx,
        y: cy - radius + 17,
        "text-anchor": "middle",
        class: "cluster-label",
        "font-size": 12,
      })
      label.textContent = folder
      files
        .sort(([a], [b]) => a.localeCompare(b))
        .forEach(([slug, entry], j) => {
          const angle = j * 2.399963229728653
          const r = files.length === 1 ? 0 : Math.sqrt((j + 0.5) / files.length) * (radius - 35)
          nodes.push({
            ...entry,
            slug,
            folder,
            x: cx + Math.cos(angle) * r,
            y: cy + 10 + Math.sin(angle) * r,
          })
        })
    })
    const bySlug = new Map(nodes.map((n) => [n.slug, n]))
    const edges = make("g", { class: "graph-edges" })
    const seen = new Set<string>()
    nodes.forEach((node) =>
      node.links?.forEach((slug) => {
        const target = bySlug.get(slug)
        const key = [node.slug, slug].sort().join("|")
        if (!target || seen.has(key) || target === node) return
        seen.add(key)
        make(
          "line",
          {
            x1: node.x,
            y1: node.y,
            x2: target.x,
            y2: target.y,
            "data-source": node.slug,
            "data-target": slug,
          },
          edges,
        )
      }),
    )
    const tooltip = document.createElement("div")
    tooltip.className = "graph-tooltip"
    tooltip.setAttribute("role", "status")
    tooltip.hidden = true
    const hide = () => {
      tooltip.hidden = true
      svg.querySelectorAll(".highlight").forEach((el) => el.classList.remove("highlight"))
    }
    nodes.forEach((node) => {
      const link = make("a", {
        href: resolveRelative(current as FullSlug, node.slug as FullSlug),
        class: "internal graph-node",
        tabindex: 0,
        "aria-label": node.title,
        "data-slug": node.slug,
        "data-folder": node.folder,
      })
      const title = make("title", {}, link)
      title.textContent = node.title
      make(
        "circle",
        {
          cx: node.x,
          cy: node.y,
          r: node.slug === current ? 6.5 : 4.5,
          fill: folderColor(node.folder),
          class: node.slug === current ? "current-node" : "",
        },
        link,
      )
      const show = () => {
        tooltip.replaceChildren()
        const heading = document.createElement("strong")
        heading.textContent = node.title
        const folder = document.createElement("small")
        folder.textContent = node.folder
        tooltip.append(heading, folder)
        const neighbors = nodes.filter(
          (n) => node.links?.includes(n.slug) || n.links?.includes(node.slug),
        )
        if (neighbors.length) {
          const names = document.createElement("span")
          names.textContent = `相关文章：${neighbors
            .slice(0, 4)
            .map((n) => n.title)
            .join("、")}${neighbors.length > 4 ? ` 等 ${neighbors.length} 篇` : ""}`
          tooltip.append(names)
        }
        tooltip.hidden = false
        edges
          .querySelectorAll("line")
          .forEach((edge) =>
            edge.classList.toggle(
              "highlight",
              edge.getAttribute("data-source") === node.slug ||
                edge.getAttribute("data-target") === node.slug,
            ),
          )
      }
      on(link, "pointerenter", show)
      on(link, "pointerleave", hide)
      on(link, "focus", show)
      on(link, "blur", hide)
      on(link, "click", ((event: MouseEvent) => {
        if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return
        event.preventDefault()
        event.stopPropagation()
        window.spaNavigate(new URL(link.getAttribute("href")!, location.href))
      }) as EventListener)
      on(link, "keydown", ((event: KeyboardEvent) => {
        if (event.key === "Enter") {
          event.preventDefault()
          window.spaNavigate(new URL(link.getAttribute("href")!, location.href))
        }
      }) as EventListener)
    })
    container.append(svg, tooltip)
    if (global) {
      let box = { x: 0, y: 0, w: width, h: height }
      const update = () => svg.setAttribute("viewBox", `${box.x} ${box.y} ${box.w} ${box.h}`)
      const controls = document.createElement("div")
      controls.className = "graph-controls"
      for (const [label, factor] of [
        ["放大", 0.75],
        ["缩小", 1.333],
        ["复位", 0],
      ] as const) {
        const button = document.createElement("button")
        button.type = "button"
        button.textContent = label
        on(button, "click", () => {
          if (!factor) box = { x: 0, y: 0, w: width, h: height }
          else {
            const next = Math.min(width * 4, Math.max(width / 16, box.w * factor))
            const ratio = next / box.w
            box = {
              x: box.x + (box.w * (1 - ratio)) / 2,
              y: box.y + (box.h * (1 - ratio)) / 2,
              w: next,
              h: box.h * ratio,
            }
          }
          hide()
          update()
        })
        controls.append(button)
      }
      container.append(controls)
      let start: { x: number; y: number; bx: number; by: number } | undefined
      on(svg, "pointerdown", ((e: PointerEvent) => {
        if ((e.target as Element).closest("a")) return
        start = { x: e.clientX, y: e.clientY, bx: box.x, by: box.y }
        svg.setPointerCapture(e.pointerId)
      }) as EventListener)
      on(svg, "pointermove", ((e: PointerEvent) => {
        if (!start) return
        const scale = Math.max(box.w / svg.clientWidth, box.h / svg.clientHeight)
        box.x = start.bx - (e.clientX - start.x) * scale
        box.y = start.by - (e.clientY - start.y) * scale
        update()
      }) as EventListener)
      on(svg, "pointerup", () => {
        start = undefined
      })
      on(svg, "pointercancel", () => {
        start = undefined
      })
    }
  }
  draw(root.querySelector('.cluster-graph[data-global="false"]')!, false)
  const dialog = root.querySelector<HTMLDialogElement>("dialog")!
  const expand = root.querySelector<HTMLButtonElement>(".graph-expand")!
  let drawn = false
  on(expand, "click", () => {
    dialog.showModal()
    if (!drawn) {
      draw(dialog.querySelector(".cluster-graph")!, true)
      drawn = true
    }
  })
  on(root.querySelector(".graph-close")!, "click", () => dialog.close())
  on(dialog, "click", (event) => {
    if (event.target === dialog) dialog.close()
  })
})
