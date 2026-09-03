import { readFile, readdir } from "node:fs/promises"
import path from "node:path"

const outputRoot = path.resolve("public")
const checkedExtensions = new Set([".html", ".css"])
const ignoredSchemes = /^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true })
  const files = []

  for (const entry of entries) {
    const absolute = path.join(directory, entry.name)
    if (entry.isDirectory()) files.push(...(await walk(absolute)))
    else files.push(absolute)
  }

  return files
}

function outputCandidates(source, reference) {
  const withoutFragment = reference.split(/[?#]/, 1)[0].replaceAll("&amp;", "&")
  if (!withoutFragment || ignoredSchemes.test(withoutFragment)) return []

  let decoded
  try {
    decoded = decodeURIComponent(withoutFragment)
  } catch {
    decoded = withoutFragment
  }

  const target = decoded.startsWith("/")
    ? path.resolve(outputRoot, `.${decoded}`)
    : path.resolve(path.dirname(source), decoded)

  const relative = path.relative(outputRoot, target)
  if (relative.startsWith("..") || path.isAbsolute(relative)) return [target]

  return [target, `${target}.html`, path.join(target, "index.html")]
}

const files = await walk(outputRoot)
const outputFiles = new Set(files.map((file) => path.resolve(file).toLowerCase()))
const checkedFiles = files.filter((file) => checkedExtensions.has(path.extname(file)))
const missing = new Map()

for (const file of checkedFiles) {
  const text = await readFile(file, "utf8")
  const references = []

  if (path.extname(file) === ".html") {
    for (const match of text.matchAll(/\b(?:href|src)=["']([^"']+)["']/gi)) {
      references.push(match[1])
    }
  } else {
    for (const match of text.matchAll(/url\(\s*["']?([^"')]+)["']?\s*\)/gi)) {
      references.push(match[1])
    }
  }

  for (const reference of references) {
    const candidates = outputCandidates(file, reference)
    if (candidates.length === 0) continue

    if (candidates.some((candidate) => outputFiles.has(path.resolve(candidate).toLowerCase())))
      continue

    const source = path.relative(outputRoot, file).replaceAll(path.sep, "/")
    const key = `${source}\0${reference}`
    missing.set(key, { source, reference })
  }
}

if (missing.size > 0) {
  console.error(`Found ${missing.size} unresolved local references:`)
  for (const { source, reference } of missing.values()) {
    console.error(`- ${source}: ${reference}`)
  }
  process.exitCode = 1
} else {
  console.log(
    `Checked ${checkedFiles.length} generated HTML/CSS files; all local references resolve.`,
  )
}
