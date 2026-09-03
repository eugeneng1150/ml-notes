# ML Notes Style Guide

## Core Shape

Use this sequence for substantial notes:

1. Introduction: explain the idea in plain language and why it matters.
2. Intuition: use an analogy, small example, or visual mental model.
3. How it works: walk through the mechanism step by step.
4. Mathematical formulation or implementation detail: add formalism only after the intuition is clear.
5. Complexity, comparison, or applications: choose the sections that fit the topic.
6. Key takeaways: end with the points worth remembering.

## Writing Rules

- Prefer precise explanations over broad surveys.
- Do not write a checklist when a causal explanation is possible.
- Make section transitions specific: mention the exact previous conclusion that motivates the next section.
- Use equations only when each symbol is defined.
- Include tensor shapes, graph sizes, matrix dimensions, or simulation state variables when they clarify the idea.
- Use tables for comparisons and mermaid for diagrams.
- Avoid ASCII art diagrams.
- Keep personal motivation natural and brief.
- Prefer `Follow-Up Topics` when a note needs to point to future reading.

## Depth Calibration

Use `build-and-modify` when the topic is central to an implementation or experiment.

Use `understand-and-reproduce` when the topic should be reproducible in a notebook or small script.

Use `intuition` when the topic is mainly needed to orient future reading.

## Topic-Specific Guidance

### Graph Neural Networks

Anchor explanations around the graph object: node features, edge features, adjacency, neighborhoods, and message passing. When possible, show how a node representation changes after one layer.

### Kernels

Move from similarity intuition to the kernel trick, then to concrete algorithms such as SVMs or Gaussian processes. Do not introduce RKHS machinery until the simple similarity view has done its job.

### Transformers

Track shapes carefully. For attention notes, always specify the dimensions of `Q`, `K`, `V`, attention logits, and output activations.

### Emergent AI Dynamics

State the agents, environment, state variables, action space, observation space, update rules, metrics, and failure modes. Distinguish simulation rules from learned behavior.

## Review Checklist

- Is there a driving question?
- Does the note build concepts before details?
- Does each major section follow from the previous one?
- Are examples concrete enough?
- Are references and claims traceable?
- Are the key takeaways specific enough to be useful later?
