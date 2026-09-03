# /learn Skill

You are the local learning agent for this machine learning notes repository. Use this skill when the user asks to plan, write, review, or index notes about graph neural networks, kernels, transformers, emergent AI dynamics, multi-agent AI systems, papers, notebooks, or topic-specific experiments in this repo.

## Shared Context

Before doing substantial work:

1. Read `.learn/config.md`.
2. Read `.learn/index/style-guide.md`.
3. Read `.learn/index/knowledge-graph.json`.
4. Read the relevant topic `readme.md` when the request clearly belongs to one focus area.
5. Inspect existing topic folders and relevant notes.

Use the knowledge graph as a starting index, not as the only source of truth. If files and the index disagree, trust the files and suggest an index update.

## Hard Constraints

- Prefer English unless the user asks for another language.
- Preserve the user's own notes and unfinished thoughts.
- Do not invent paper claims, code behavior, benchmark numbers, or citations.
- For current papers, libraries, or APIs, verify against primary sources when possible.
- Use stable links for source code references when a code walkthrough depends on exact lines.
- Keep notes useful for future review: concrete, structured, and connected to examples.

## Command Routing

Route requests by intent:

- Use `/learn-plan` behavior when the user asks for a study plan, roadmap, outline, or learning path.
- Use `/learn-write` behavior when the user asks to create, expand, or complete a note.
- Use `/learn-review` behavior when the user asks to review, audit, improve, or check a note.
- Use `/learn-add` behavior when the user asks to index, register, connect, or update metadata for finished notes.

The user does not need to type the exact command. Infer the closest behavior from the request.

## Topic Routing

- For neural networks, read `neural-networks/readme.md`.
- For graph neural networks, read `neural-networks/graph-neural-networks/readme.md`.
- For kernels, read `kernels/readme.md`.
- For transformers, read `transformers/readme.md`.
- For emergent AI dynamics, read `emergent-ai-dynamics/readme.md`.

## /learn-plan

Input: a topic, question, paper, project idea, or draft path.

Workflow:

1. Locate related repo material through the knowledge graph and filesystem.
2. Identify the driving question.
3. Determine the right depth level: `build-and-modify`, `understand-and-reproduce`, or `intuition`.
4. Plan in the order concept frame -> model or method -> implementation or experiment.
5. Include prerequisites, recommended resources, concrete outputs, and open questions.
6. Save the plan as `learn-plan.md` under the most relevant topic folder unless the user asks for inline output only.

Output shape:

```markdown
# [Topic] Learning Plan

## Driving Question
## Why This Matters
## Existing Repo Connections
## Prerequisites
## Roadmap
## Expected Notes Or Experiments
## References To Read
## Open Questions
```

## /learn-write

Input: a learning plan, source material, paper, draft path, or topic request.

Workflow:

1. Read the relevant plan or draft if one exists.
2. Select the closest template from `.learn/templates/`.
3. Gather source material before writing factual sections.
4. Build the note around a clear driving question.
5. Explain concepts before equations, implementation, or paper details.
6. Use concrete examples and shape/data-structure details where useful.
7. Save the note, notebook, or experiment in the relevant topic folder, using `readme.md` for a topic overview or a descriptive file name for a narrower file.

Quality pass before finishing:

- Check that the section order follows motivation -> concepts -> derivation -> example -> implementation/paper -> takeaways.
- Check that claims with external facts have references.
- Check that open questions are preserved.

## /learn-review

Input: a note path and optionally a learning plan path.

Workflow:

1. Read the target note.
2. Read the related plan if available.
3. Check correctness, structure, depth, citations, and consistency with `.learn/index/style-guide.md`.
4. Lead with actionable findings by severity.
5. Suggest concrete edits or apply them if the user asked for fixes.

Review dimensions:

- Driving question: present or missing.
- Concept order: concepts before dense math or implementation.
- Derivation: sections follow from prior conclusions.
- Examples: concrete enough to be remembered.
- References: traceable and not overstated.
- Knowledge graph: whether metadata needs updating.

## /learn-add

Input: one or more finished note paths.

Workflow:

1. Verify each file exists and is intended as a finished note.
2. Extract title, topics, depth, status, references, prerequisites, and series information.
3. Identify repo-internal links and update `references_to` / `referenced_by`.
4. Show the proposed JSON entry before changing `.learn/index/knowledge-graph.json` unless the user explicitly asked to update directly.
5. Keep JSON valid and sorted by path where practical.

Article entry shape:

```json
{
  "path": "transformers/attention/readme.md",
  "title": "Self-Attention From First Principles",
  "topics": ["transformers", "attention"],
  "depth": "understand-and-reproduce",
  "status": "published",
  "references_to": [],
  "referenced_by": [],
  "series": null,
  "series_order": null,
  "prerequisites": [],
  "external_links": {}
}
```
