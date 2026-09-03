---
name: ml-notes-learning
description: Plan, write, review, and index machine learning notes in this repository, especially for graph neural networks, kernel methods, transformers, emergent AI dynamics, multi-agent AI systems, papers, notebooks, topic-specific experiments, and reusable learning roadmaps. Use when the user asks for /learn-plan, /learn-write, /learn-review, /learn-add, study plans, note drafting, note review, or knowledge graph updates for this repo.
---

# ML Notes Learning

Use this skill as the Codex-discoverable entrypoint for the repo-local `.learn/` system.

## Load Context

Before doing substantial work:

1. Read `.learn/skill.md`.
2. Read `.learn/config.md`.
3. Read `.learn/index/style-guide.md`.
4. Read `.learn/index/knowledge-graph.json`.
5. Read the relevant root topic `readme.md` when the request clearly belongs to graph neural networks, kernels, transformers, or emergent AI dynamics.
6. Inspect existing notes in the relevant topic folder.

If a `.learn/` file is missing, continue with the closest available instructions and mention the missing file.

## Invocation Mapping

Treat these user phrases as commands:

- `/learn-plan ...` means use the `/learn-plan` workflow in `.learn/skill.md`.
- `/learn-write ...` means use the `/learn-write` workflow in `.learn/skill.md`.
- `/learn-review ...` means use the `/learn-review` workflow in `.learn/skill.md`.
- `/learn-add ...` means use the `/learn-add` workflow in `.learn/skill.md`.

Codex may also use this skill implicitly for requests like:

- "Create a learning plan for GraphSAGE."
- "Write a note about the kernel trick."
- "Review my transformer attention note."
- "Add this finished note to the knowledge graph."

## Output Behavior

- Save generated plans and notes to the repo unless the user asks for inline output only.
- Prefer English unless the user asks for another language.
- Do not invent citations, paper results, code behavior, or benchmark numbers.
- Keep outputs concrete, structured, and useful for later review.

## Topic Guidance Locations

- Neural networks: `neural-networks/readme.md`
- Graph neural networks: `neural-networks/graph-neural-networks/readme.md`
- Kernels: `kernels/readme.md`
- Transformers: `transformers/readme.md`
- Emergent AI dynamics: `emergent-ai-dynamics/readme.md`
