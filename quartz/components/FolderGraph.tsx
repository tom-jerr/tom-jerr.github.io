import type { QuartzComponent } from "./types"
// @ts-ignore: Quartz bundles inline scripts as strings.
import script from "./scripts/folderGraph.inline"

export const FolderGraph: QuartzComponent = () => (
  <section class="folder-graph">
    <div class="graph-heading">
      <h2>知识图谱</h2>
      <button class="graph-expand" type="button" aria-label="展开全站知识图谱">
        ⤢
      </button>
    </div>
    <p class="graph-hint">同目录聚类 · 悬浮查看文章</p>
    <div class="cluster-graph" data-global="false" />
    <dialog class="graph-dialog" aria-label="全站知识图谱">
      <div class="graph-heading">
        <h2>全站知识图谱</h2>
        <button class="graph-close" type="button" aria-label="关闭知识图谱">
          ×
        </button>
      </div>
      <p class="graph-hint">颜色对应目录 · 连线表示文章引用 · 拖动平移，按钮缩放</p>
      <div class="cluster-graph" data-global="true" />
    </dialog>
  </section>
)
FolderGraph.afterDOMLoaded = script
