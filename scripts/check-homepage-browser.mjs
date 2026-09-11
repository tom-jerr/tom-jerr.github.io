// Read-only UI acceptance against localhost; launches an isolated Chrome profile.
// Usage: node scripts/check-homepage-browser.mjs [chrome.exe] [baseUrl]
import assert from "node:assert/strict"
import fs from "node:fs/promises"
import path from "node:path"
import { spawn } from "node:child_process"

const executable = process.argv[2] ?? "C:/Program Files/Google/Chrome/Application/chrome.exe"
const base = process.argv[3] ?? "http://localhost:8080"
const output = path.resolve("tmp/homepage-qa")
await fs.mkdir(output, { recursive: true })
const profile = await fs.mkdtemp(path.join(output, "chrome-profile-"))
const chrome = spawn(
  executable,
  [
    "--headless=new",
    "--remote-debugging-port=0",
    `--user-data-dir=${profile}`,
    "--no-first-run",
    "--no-default-browser-check",
    "about:blank",
  ],
  { windowsHide: true, stdio: ["ignore", "ignore", "pipe"] },
)
const endpoint = await new Promise((resolve, reject) => {
  let logs = ""
  const timer = setTimeout(() => reject(new Error("Chrome startup timed out")), 15000)
  chrome.once("error", reject)
  chrome.stderr.on("data", (chunk) => {
    logs += chunk
    const match = logs.match(/DevTools listening on (ws:\/\/[^\s]+)/)
    if (match) {
      clearTimeout(timer)
      resolve(match[1])
    }
  })
})
const socket = new WebSocket(endpoint)
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true })
  socket.addEventListener("error", reject, { once: true })
})
let id = 0
const pending = new Map()
const runtimeErrors = []
socket.addEventListener("message", ({ data }) => {
  const message = JSON.parse(data)
  if (message.method === "Runtime.exceptionThrown")
    runtimeErrors.push(message.params.exceptionDetails)
  if (!pending.has(message.id)) return
  const { resolve, reject, timer } = pending.get(message.id)
  clearTimeout(timer)
  pending.delete(message.id)
  if (message.error) reject(new Error(JSON.stringify(message.error)))
  else resolve(message.result)
})
function send(method, params = {}, sessionId) {
  return new Promise((resolve, reject) => {
    const messageId = ++id
    const timer = setTimeout(() => {
      pending.delete(messageId)
      reject(new Error(`${method} timed out`))
    }, 20000)
    pending.set(messageId, { resolve, reject, timer })
    socket.send(JSON.stringify({ id: messageId, method, params, sessionId }))
  })
}

try {
  const { targetId } = await send("Target.createTarget", { url: "about:blank" })
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true })
  const command = (method, params = {}) => send(method, params, sessionId)
  const evaluate = async (expression) => {
    const result = await command("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: true,
    })
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails))
    return result.result.value
  }
  const waitFor = async (expression) => {
    for (let i = 0; i < 80; i++) {
      if (await evaluate(expression)) return
      await new Promise((resolve) => setTimeout(resolve, 150))
    }
    await screenshot("failure")
    console.log(
      await evaluate(
        `JSON.stringify({title:document.title, home:!!document.querySelector('.home-contact'), roots:Array.from(document.querySelectorAll('.explorer-ul > li > .folder-container .folder-title'),e=>e.textContent), data:document.querySelector('.explorer')?.dataset.dataFns})`,
      ),
    )
    console.log(runtimeErrors)
    throw new Error(`Condition timed out: ${expression}`)
  }
  const screenshot = async (name) => {
    const result = await command("Page.captureScreenshot", { format: "png" })
    await fs.writeFile(path.join(output, `${name}.png`), Buffer.from(result.data, "base64"))
  }
  const roots = `Array.from(document.querySelectorAll('.explorer-ul > li > .folder-container .folder-title'), e => e.textContent)`
  await command("Runtime.enable")
  await command("Page.enable")
  await command("Network.enable")
  await command("Network.setBlockedURLs", {
    // Keep UI acceptance independent of analytics and the remote comments client.
    urls: ["*googletagmanager*", "*google-analytics*", "*clustrmaps*", "*utteranc.es*"],
  })
  await command("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 1000,
    deviceScaleFactor: 1,
    mobile: false,
  })
  await command("Page.navigate", { url: base })
  await waitFor(`document.querySelector('.home-contact') && (${roots}).length === 5`)
  const expected = ["cuda", "gpu", "llm_inference", "sglang", "notes"]
  assert.deepEqual(await evaluate(roots), expected)
  const serialized = await evaluate(
    `JSON.parse(document.querySelector('.explorer').dataset.dataFns)`,
  )
  assert.ok(
    serialized.filterFn.includes("llm_inference"),
    "Configured Explorer filter was not emitted",
  )
  assert.equal(await evaluate(`document.querySelectorAll('.project-card').length`), 8)
  assert.equal(
    await evaluate(`document.querySelectorAll('.home-contact svg:not(.external-icon)').length`),
    6,
  )
  assert.ok(await evaluate(`document.querySelector('article').textContent.includes('秋招机会')`))
  assert.ok(await evaluate(`!document.querySelector('article').textContent.includes('实习机会')`))
  await evaluate(
    `Promise.all(Array.from(document.querySelectorAll('.project-card img'), img => { img.loading='eager'; return img.decode() }))`,
  )
  assert.equal(
    await evaluate(
      `document.querySelectorAll('.project-card img').length === Array.from(document.querySelectorAll('.project-card img')).filter(i=>i.naturalWidth===800).length`,
    ),
    true,
  )
  const recent = await evaluate(
    `Array.from(document.querySelectorAll('.recent-notes .article-card .desc h3 a'), a=>a.getAttribute('href'))`,
  )
  assert.equal(recent.length, 8)
  assert.ok(
    recent.every((href) =>
      expected.includes(href.replace(/^\.\//, "").replace(/^\//, "").split("/")[0]),
    ),
    JSON.stringify(recent),
  )
  await screenshot("desktop-home")
  await waitFor(
    `document.querySelectorAll('.folder-glyph').length >= 5 && document.querySelectorAll('.graph-node').length > 0`,
  )
  assert.equal(
    await evaluate(
      `new Set(Array.from(document.querySelectorAll('.explorer-ul > li > .folder-container'), el => el.style.getPropertyValue('--folder-color'))).size`,
    ),
    5,
  )
  assert.ok(
    await evaluate(`document.querySelector('.sidebar .taxonomy').textContent.includes('Project')`),
  )
  await evaluate(
    `document.querySelector('.graph-node').dispatchEvent(new PointerEvent('pointerenter'))`,
  )
  assert.ok(
    await evaluate(
      `!document.querySelector('.graph-tooltip').hidden && document.querySelector('.graph-tooltip strong').textContent.length > 0`,
    ),
  )
  await evaluate(`document.querySelector('.graph-expand').click()`)
  await waitFor(
    `document.querySelector('.graph-dialog').open && document.querySelectorAll('.graph-dialog [data-cluster]').length > 1`,
  )
  assert.ok(
    await evaluate(
      `Array.from(document.querySelectorAll('.graph-dialog .graph-node')).every(n => n.dataset.folder === n.dataset.slug.slice(0,n.dataset.slug.lastIndexOf('/')) || n.dataset.folder === '其他')`,
    ),
  )
  await screenshot("desktop-graph")
  const initialViewBox = await evaluate(
    `document.querySelector('.graph-dialog svg').getAttribute('viewBox')`,
  )
  await evaluate(`document.querySelector('.graph-controls button').click()`)
  assert.notEqual(
    await evaluate(`document.querySelector('.graph-dialog svg').getAttribute('viewBox')`),
    initialViewBox,
  )
  await command("Input.dispatchKeyEvent", {
    type: "keyDown",
    key: "Escape",
    code: "Escape",
    windowsVirtualKeyCode: 27,
  })
  await command("Input.dispatchKeyEvent", {
    type: "keyUp",
    key: "Escape",
    code: "Escape",
    windowsVirtualKeyCode: 27,
  })
  await waitFor(`!document.querySelector('.graph-dialog').open`)
  const graphTarget = await evaluate(
    `(() => { const n = document.querySelector('.cluster-graph[data-global="false"] .graph-node'); const r = n.querySelector('circle').getBoundingClientRect(); return {x:r.x+r.width/2,y:r.y+r.height/2,slug:n.dataset.slug,title:n.getAttribute('aria-label')} })()`,
  )
  await command("Input.dispatchMouseEvent", {
    type: "mouseMoved",
    x: graphTarget.x,
    y: graphTarget.y,
  })
  await waitFor(
    `document.querySelector('.graph-tooltip strong')?.textContent === ${JSON.stringify(graphTarget.title)} && !document.querySelector('.graph-tooltip').hidden`,
  )
  await screenshot("desktop-graph-hover")
  await command("Input.dispatchMouseEvent", {
    type: "mousePressed",
    x: graphTarget.x,
    y: graphTarget.y,
    button: "left",
    clickCount: 1,
  })
  await command("Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x: graphTarget.x,
    y: graphTarget.y,
    button: "left",
    clickCount: 1,
  })
  await waitFor(`document.body.dataset.slug === ${JSON.stringify(graphTarget.slug)}`)
  await waitFor(`document.querySelector('.current-node')`)
  await evaluate(
    `document.querySelector('.folder-container[data-folderpath="cuda/index"] a').click()`,
  )
  await waitFor(
    `document.body.dataset.slug === 'cuda/index' && document.querySelector('.discovery-listing .article-card')`,
  )
  assert.ok(
    await evaluate(
      `document.querySelectorAll('.discovery-listing .article-description').length > 3`,
    ),
  )
  await evaluate(
    `Promise.all(Array.from(document.querySelectorAll('.discovery-listing .article-cover img'), img => {img.loading='eager'; return img.decode()}))`,
  )
  assert.ok(
    await evaluate(`document.querySelectorAll('.discovery-listing .article-cover img').length > 3`),
  )
  await screenshot("desktop-folder-cards")
  await evaluate(`document.querySelector('.taxonomy a[href$="tags/cuda"]').click()`)
  await waitFor(
    `document.body.dataset.slug === 'tags/cuda' && document.querySelector('.discovery-listing .article-card')`,
  )
  assert.ok(
    await evaluate(
      `Array.from(document.querySelectorAll('.discovery-listing .article-card')).every(card => Array.from(card.querySelectorAll('.tag-link')).some(a => a.textContent === '#cuda'))`,
    ),
  )
  await evaluate(`document.querySelector('.taxonomy a[href$="categories/Project"]').click()`)
  await waitFor(`document.body.dataset.slug === 'categories/Project'`)
  assert.equal(
    await evaluate(`document.querySelectorAll('.discovery-listing .article-card').length`),
    4,
  )
  await screenshot("desktop-category")
  await evaluate(`document.documentElement.setAttribute('saved-theme','dark')`)
  await screenshot("desktop-category-dark")
  await evaluate(`document.documentElement.setAttribute('saved-theme','light')`)
  await command("Page.navigate", { url: base })
  await waitFor(`document.querySelector('.home-contact') && (${roots}).length === 5`)
  assert.ok(
    await evaluate(
      `Array.from(document.querySelectorAll('.project-card img')).every(i=>Math.abs(i.clientWidth / i.clientHeight - 2) < .02)`,
    ),
    "Architecture images must preserve 2:1 ratio",
  )
  await evaluate(`document.querySelector('.project-grid').scrollIntoView({behavior:'instant'})`)
  await waitFor(`document.querySelector('.project-grid').getBoundingClientRect().top < 100`)
  await screenshot("desktop-projects")
  for (const [name, width] of [
    ["tablet", 1000],
    ["mobile", 390],
  ]) {
    await command("Emulation.setDeviceMetricsOverride", {
      width,
      height: 844,
      deviceScaleFactor: 1,
      mobile: false,
    })
    await command("Page.navigate", { url: base })
    await waitFor(`document.querySelector('.home-contact') && (${roots}).length === 5`)
    await evaluate("scrollTo({top:0,behavior:'instant'})")
    await waitFor("document.documentElement.scrollWidth <= innerWidth + 1")
    await screenshot(`${name}-home`)
    if (name === "mobile") {
      await evaluate(`document.querySelector('.mobile-explorer').click()`)
      await waitFor(`!document.querySelector('.explorer').classList.contains('collapsed')`)
      await evaluate(
        `Promise.all(document.querySelector('.explorer').getAnimations({subtree:true}).map(a=>a.finished.catch(()=>{})))`,
      )
      assert.deepEqual(await evaluate(roots), expected)
      assert.equal(
        await evaluate(`getComputedStyle(document.querySelector('.recent-notes')).display`),
        "none",
      )
      await screenshot("mobile-navigation")
      assert.ok(await evaluate(`document.querySelector('.sidebar > .taxonomy').checkVisibility()`))
      await evaluate(`document.querySelector('.mobile-explorer').click()`)
      await evaluate(
        `Promise.all(document.querySelector('.explorer').getAnimations({subtree:true}).map(a=>a.finished.catch(()=>{})))`,
      )
      await evaluate(`document.querySelector('.project-grid').scrollIntoView({behavior:'instant'})`)
      await waitFor(`document.querySelector('.project-grid').getBoundingClientRect().top < 100`)
      await screenshot("mobile-projects")
      await evaluate(`document.documentElement.setAttribute('saved-theme','dark')`)
      await screenshot("mobile-projects-dark")
      await evaluate(`document.documentElement.setAttribute('saved-theme','light')`)
    }
  }
  // A real in-site link activates Quartz SPA routing; then verify the same filter.
  await evaluate(`document.querySelector('a[href*="bustub通关指北"]').click()`)
  await waitFor(
    `!document.querySelector('.home-contact') && document.querySelector('article')?.textContent.includes('Buffer')`,
  )
  assert.deepEqual(await evaluate(roots), expected)
  await command("Page.navigate", { url: `${base}/sglang/` })
  await waitFor(`document.querySelector('.discovery-listing .article-card')`)
  await evaluate(
    `Promise.all(Array.from(document.querySelectorAll('.discovery-listing .article-cover img'), img => {img.loading='eager'; return img.decode()}))`,
  )
  assert.ok(await evaluate(`document.documentElement.scrollWidth <= innerWidth + 1`))
  await screenshot("mobile-article-cards")
  assert.equal(runtimeErrors.length, 0, JSON.stringify(runtimeErrors))
  console.log(
    JSON.stringify(
      {
        roots: expected,
        projectCards: 8,
        contactIcons: 6,
        viewports: [1440, 1000, 390],
        spaNavigation: "passed",
        runtimeErrors,
        output,
      },
      null,
      2,
    ),
  )
} finally {
  await send("Browser.close").catch(() => {})
  socket.close()
  chrome.kill()
}
