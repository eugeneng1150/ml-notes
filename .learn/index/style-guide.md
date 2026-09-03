# ML Notes Style Guide

## Core Shape

Use this sequence for substantial notes:

1. Motivation: explain the concrete question or confusion that makes the topic worth studying.
2. Concept frame: define the minimum mental model needed before details.
3. Derivation path: show how the next concept follows from the previous one.
4. Example: use a toy example, numeric example, diagram, or code sketch.
5. Implementation or paper connection: connect the idea to real code, a paper, or an experiment.
6. Takeaways and open questions: record what is known, what is still unclear, and what to revisit.

## Writing Rules

- Prefer precise explanations over broad surveys.
- Do not write a checklist when a causal explanation is possible.
- Make section transitions specific: mention the exact previous conclusion that motivates the next section.
- Use equations only when each symbol is defined.
- Include tensor shapes, graph sizes, matrix dimensions, or simulation state variables when they clarify the idea.
- Use tables for comparisons and mermaid for diagrams.
- Avoid ASCII art diagrams.
- Keep personal motivation natural and brief.

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
- Are open questions explicit?
