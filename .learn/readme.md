# /learn Agent For ML Notes

This directory defines a repo-local learning agent for maintaining this machine learning notes repository.

The goal is to turn the repo from a passive folder of notes into an active study system. The agent should help plan learning paths, draft notes, review notes, and maintain a lightweight knowledge graph across topics.

## Learning Philosophy

The notes should follow four principles:

1. Motivation first: start from a concrete confusion, project idea, paper, or implementation question.
2. Concepts before mechanics: build the mental model before code or equations become dense.
3. Derivation over listing: each section should follow naturally from the previous section.
4. Notes should become reusable: every article should leave behind concepts, examples, references, and takeaways that are easy to revisit.

## Agent Skills

The local agent supports four workflows:

- `/learn-plan`: create a learning plan for a new topic.
- `/learn-write`: write or extend a note from a plan, draft, or source material.
- `/learn-review`: review a note for correctness, structure, depth, and style.
- `/learn-add`: add a finished note to the knowledge graph.

## Directory Structure

```text
.learn/
├── readme.md
├── skill.md
├── config.md
├── index/
│   ├── knowledge-graph.json
│   └── style-guide.md
└── templates/
    ├── concept-note.md
    ├── paper-reading.md
    ├── code-walkthrough.md
    └── simulation-idea.md
```

## Current Focus Areas

- Graph neural networks
- Kernel methods
- Transformers
- Emergent AI dynamics

Topic-specific study guidance lives in the root topic folders:

- `graph-neural-networks/`
- `kernels/readme.md`
- `transformers/readme.md`
- `emergent-ai-dynamics/readme.md`
