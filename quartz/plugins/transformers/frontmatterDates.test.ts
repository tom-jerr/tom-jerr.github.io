import assert from "node:assert/strict"
import test from "node:test"
import { normalizeFrontmatterDates } from "./frontmatterDates"

test("maps updated to the canonical modified date", () => {
  const frontmatter: Record<string, unknown> = {
    created: "2025-09-28",
    updated: "2025-12-13",
  }

  normalizeFrontmatterDates(frontmatter)

  assert.equal(frontmatter.modified, "2025-12-13")
})

test("keeps an explicit modified date", () => {
  const frontmatter: Record<string, unknown> = {
    modified: "2025-12-12",
    updated: "2025-12-13",
  }

  normalizeFrontmatterDates(frontmatter)

  assert.equal(frontmatter.modified, "2025-12-12")
})

test("supports the legacy update alias when it has a value", () => {
  const frontmatter: Record<string, unknown> = { update: "2025-12-13" }

  normalizeFrontmatterDates(frontmatter)

  assert.equal(frontmatter.modified, "2025-12-13")
})
