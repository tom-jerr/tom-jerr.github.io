// Local visual QA previews only; SVGs remain the published, editable assets.
import fs from "node:fs/promises"
import path from "node:path"
import sharp from "sharp"

const source = path.resolve("content/img/projects")
const target = path.resolve("tmp/project-diagram-check")
await fs.mkdir(target, { recursive: true })
const files = (await fs.readdir(source)).filter((name) => name.endsWith("-architecture.svg"))
const tiles = []
for (const [index, file] of files.entries()) {
  const png = await sharp(path.join(source, file)).png().toBuffer()
  await fs.writeFile(path.join(target, file.replace(/\.svg$/, ".png")), png)
  tiles.push({ input: png, left: (index % 2) * 800, top: Math.floor(index / 2) * 400 })
}
await sharp({
  create: {
    width: 1600,
    height: Math.ceil(files.length / 2) * 400,
    channels: 4,
    background: "#f8fafd",
  },
})
  .composite(tiles)
  .png()
  .toFile(path.join(target, "project-diagrams.png"))
console.log(`Rendered ${files.length} diagrams and contact sheet in ${target}`)
