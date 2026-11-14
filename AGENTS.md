# Repository guidance

## Zotero and paperreading workflow

Apply these rules when a task touches Zotero, `paperreadings`, or literature research. Keep unrelated blog content and existing user edits unchanged.

### Sources of truth and links

- Zotero is authoritative for bibliographic metadata, attached PDFs, Collections, tags, and the user's original PDF annotations.
- Markdown in `draft/paperreadings/` is authoritative for new single-paper analysis and cross-paper synthesis until the user reviews and publishes it.
- Published reading notes live in `content/paperreadings/`. Do not move or publish a draft unless the user asks.
- Use the Zotero item key as the stable link between systems. Include `zoteroKey`, `doi` when verified, and `zoteroURI` in new Markdown notes. The Zotero child note maintained by this workflow contains the repository-relative Markdown path.

### Intake and Zotero writes

- Before creating an item, search by DOI and then title; reuse an existing item when found and verify its metadata before reporting success.
- Place items in the Collection the user names. If none is named, use an existing Inbox Collection if one is present; otherwise leave the item uncollected and report that. Apply existing relevant tags; do not silently create a new Collection or tagging scheme.
- Attach only a lawful open-access PDF or a file supplied by the user. Never claim a PDF is attached until re-reading the item attachments confirms it.
- Add a child note titled `Codex Paperreadings backlink` only when the item has no backlink created by this workflow. Its body must include the marker `<!-- codex-paperreadings-link -->` and the current repository-relative Markdown path. On publication, update only this workflow-owned backlink after re-reading it; never edit the user's other notes or annotations.
- Do not delete Zotero items, attachments, or notes. Do not overwrite user-authored item metadata, notes, or annotations. For conflicting metadata, report the discrepancy instead of silently replacing it.
- After every Zotero write, re-read the item or note and verify the result. Keep operations idempotent by checking for an existing item or workflow backlink first.

### Reading and research claims

- Read the paper body in sections when available; do not treat an abstract as a full-paper review.
- Separate paper-reported claims, directly verified facts, and Codex interpretation. Do not invent metadata, page numbers, measurements, citations, or experimental conditions; mark unavailable evidence as unknown.
- When recording a claim, point to a verified page, section, figure, table, or source URL when available.
- In citation or knowledge graph analyses, distinguish citation, incoming-citation, and semantic-similarity edges. State the data source and coverage limits; a missing edge does not prove that no relationship exists.
- Write new paper notes and research syntheses under `draft/paperreadings/`, organized using the closest existing topic folder. Keep graph exports under `draft/paperreadings/exports/` unless the user requests another destination.
- Preserve the existing paperreading template's problem, workload, invariants, failure model, ownership, trade-off, and transfer-to-LLM-infrastructure framing. Use the repository's Chinese technical-writing style and retain source evidence.
