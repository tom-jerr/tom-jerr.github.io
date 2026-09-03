import { defineConfig } from "tsup"
import type { Plugin } from "esbuild"
import path from "node:path"

const inlineScriptPlugin: Plugin = {
  name: "inline-script-loader",
  setup(parentBuild) {
    const absWorkingDir = parentBuild.initialOptions.absWorkingDir ?? process.cwd()

    parentBuild.onLoad({ filter: /\.scss$/ }, async (args) => {
      const fs = await import("node:fs")
      const text = await fs.promises.readFile(args.path, "utf8")
      return { contents: text, loader: "text" }
    })

    parentBuild.onLoad({ filter: /\.inline\.ts$/ }, async (args) => {
      const esbuild = await import("esbuild")
      const fs = await import("node:fs")
      const text = await fs.promises.readFile(args.path, "utf8")
      const result = await esbuild.build({
        stdin: {
          contents: text,
          loader: "ts",
          resolveDir: path.dirname(args.path),
          sourcefile: path.relative(absWorkingDir, args.path),
        },
        write: false,
        bundle: true,
        minify: true,
        platform: "browser",
        format: "esm",
        target: "es2020",
      })

      const script = result.outputFiles?.[0]?.text
      if (!script) throw new Error(`Unable to compile ${args.path}`)
      return { contents: script, loader: "text" }
    })
  },
}

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "components/index": "src/components/index.ts",
  },
  format: ["esm"],
  dts: true,
  tsconfig: "tsconfig.build.json",
  sourcemap: true,
  clean: true,
  treeshake: true,
  target: "es2022",
  splitting: false,
  noExternal: [/.*/],
  external: [
    "preact",
    "preact/hooks",
    "preact/jsx-runtime",
    "preact/compat",
    "@jackyzha0/quartz",
    "@jackyzha0/quartz/*",
    "vfile",
    "vfile/*",
    "unified",
  ],
  outDir: "dist",
  platform: "node",
  esbuildOptions(options) {
    options.jsx = "automatic"
    options.jsxImportSource = "preact"
  },
  esbuildPlugins: [inlineScriptPlugin],
})
