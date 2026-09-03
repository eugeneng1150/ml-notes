# Machine Learning Notes

This repository is my working notebook for machine learning. I use it to consolidate concepts, paper notes, derivations, implementation details, and project ideas into something I can revisit and build on over time.

I started this because research can make it easy to jump from paper to paper without retaining the deeper structure behind the ideas. These notes are meant to slow that process down: clarify the core concepts, connect them to examples, and turn scattered reading into a durable learning system.

Contributions, corrections, and suggested resources are welcome. I am still learning, and this repository is intentionally written as an evolving study record rather than a finished textbook.

## Goals

- Build clear conceptual notes that are easy to revisit.
- Track mathematical foundations, algorithms, and implementation patterns.
- Summarize important papers and connect them to practical ideas.
- Develop small experiments that turn theory into working code.
- Maintain a backlog of emergent AI dynamics ideas and research directions.

## Learning Areas

### Neural Networks

Topics to study:

- Multilayer perceptrons
- Convolutional neural networks
- Recurrent neural networks
- Graph representations: nodes, edges, adjacency matrices, incidence matrices
- Message passing neural networks
- Graph convolutional networks
- Graph attention networks
- GraphSAGE and neighborhood sampling
- Positional encodings for graphs
- Over-smoothing and over-squashing
- Autoencoders
- Neural ODEs
- Applications in molecules, recommendation systems, knowledge graphs, and social networks

### Kernels

Topics to study:

- Kernel trick
- Positive definite kernels
- Reproducing kernel Hilbert spaces
- Support vector machines
- Gaussian processes
- Polynomial, RBF, Laplacian, and string kernels
- Kernel methods for structured data
- Connections between kernels and neural networks

### Transformers

Topics to study:

- Self-attention and multi-head attention
- Positional encodings and rotary embeddings
- Encoder, decoder, and encoder-decoder architectures
- Training objectives for language models
- Scaling laws
- Retrieval-augmented generation
- Efficient attention variants
- Interpretability and mechanistic analysis

### Emergent AI Dynamics

Topics to explore:

- Multi-agent simulations
- Emergent behavior from simple agent rules
- Learning environments for decision-making agents
- LLM-driven agents with memory and planning
- Simulated economies, markets, societies, and ecosystems
- Reinforcement learning in custom environments
- Evaluation methods for simulation quality and agent behavior

## Agent-Assisted Workflow

This repo includes a local `.learn/` directory and a Codex skill at `.agents/skills/ml-notes-learning/`. Use it for prompts like:

- `$ml-notes-learning /learn-plan I want to understand GraphSAGE`
- `$ml-notes-learning /learn-write based on this paper, create a note under kernels/`
- `$ml-notes-learning /learn-review transformers/attention/readme.md`
- `$ml-notes-learning /learn-add neural-networks/graph-neural-networks/message-passing.md`

In Codex CLI or the IDE extension, you can also type `/skills` and select `ml-notes-learning`.

### Available Codex Skill

#### `$ml-notes-learning`

Use this skill to manage the learning workflow for this repository.

Available workflows:

| Workflow | What it does | Example |
| --- | --- | --- |
| `/learn-plan` | Creates a learning plan for a topic, including prerequisites, roadmap, expected notes or experiments, references, and open questions. | `$ml-notes-learning /learn-plan transformers from first principles` |
| `/learn-write` | Writes or expands a note using the repo templates and saves it under the relevant topic folder. | `$ml-notes-learning /learn-write a concept note on message passing in GNNs` |
| `/learn-review` | Reviews an existing note for correctness, structure, examples, citations, depth, and consistency with the style guide. | `$ml-notes-learning /learn-review transformers/readme.md` |
| `/learn-add` | Adds a finished note to `.learn/index/knowledge-graph.json` so relationships between notes can be tracked. | `$ml-notes-learning /learn-add kernels/kernel-trick.md` |

The topic folders are not Codex skills. They are normal note areas:

- `neural-networks/`
- `kernels/`
- `transformers/`
- `emergent-ai-dynamics/`

The actual Codex skill lives at `.agents/skills/ml-notes-learning/SKILL.md`. The `.learn/` directory contains supporting workflow instructions, templates, style rules, and the knowledge graph index used by that skill.
