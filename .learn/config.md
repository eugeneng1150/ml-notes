# /learn Config

## Language

- Primary note language: English
- Acceptable style: concise, technical, first-person when explaining motivation
- Avoid: generic textbook summaries that do not connect to examples or projects

## Topic Roots

```yaml
topic_roots:
  neural-networks: neural-networks/
  graph-neural-networks: neural-networks/graph-neural-networks/
  kernels: kernels/
  transformers: transformers/
  emergent-ai-dynamics: emergent-ai-dynamics/
  papers: papers/
```

## Depth Levels

```yaml
depth_levels:
  build-and-modify:
    meaning: Understand deeply enough to implement, debug, modify, or extend.
    examples:
      - transformer attention implementation
      - message passing GNN layer
      - custom simulation environment
  understand-and-reproduce:
    meaning: Understand the method well enough to reproduce a small version or explain tradeoffs.
    examples:
      - SVM with an RBF kernel
      - GraphSAGE sampling
      - small agent-based simulation
  intuition:
    meaning: Build a conceptual model without proving every theorem or implementing every detail.
    examples:
      - RKHS intuition
      - scaling laws
      - emergent behavior in multi-agent systems
```

## Citation Rules

- Prefer primary sources: papers, official documentation, and source code.
- Include links for important claims.
- For code references, prefer stable commits or tagged versions when available.
- Separate facts from personal interpretation.

## Review Standards

- A note should have a clear driving question.
- Important concepts should be explained with at least one concrete example.
- Math should support the explanation instead of replacing it.
- Implementation notes should mention assumptions, shapes, data structures, or failure modes.
- Open questions should be preserved rather than hidden.
