# Machine Learning Notes

This repository is my working notebook for machine learning. I use it to consolidate concepts, paper notes, derivations, implementation details, and project ideas into something I can revisit and build on over time.

I started this because research can make it easy to jump from paper to paper without retaining the deeper structure behind the ideas. These notes are meant to slow that process down: clarify the core concepts, connect them to examples, and turn scattered reading into a durable learning system.

Contributions, corrections, and suggested resources are welcome. I am still learning, and this repository is intentionally written as an evolving study record rather than a finished textbook.

## Goals

* Build clear conceptual notes that are easy to revisit.
* Track mathematical foundations, algorithms, and implementation patterns.
* Summarize important papers and connect them to practical ideas.
* Develop small experiments that turn theory into working code.
* Maintain a backlog of emergent AI dynamics ideas and research directions.

## Learning Areas

### Machine Learning Foundations

Resources:

* [Stanford CS229: Machine Learning](https://cs229.stanford.edu/summer2020/syllabus-summer2020.html)

Topics to study:

* Linear regression
* Logistic regression
* Generalized linear models
* Generative learning algorithms
* Support vector machines
* Kernel methods
* Gaussian processes
* Bias-variance tradeoff
* Regularization
* Learning theory
* Clustering
* Expectation-maximization
* Principal component analysis

### Neural Networks

Resources:

* [Stanford CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/2025/schedule.html)
* [Stanford CS224W: Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/)

Topics to study:

* Multilayer perceptrons
* Backpropagation
* Optimization
* Convolutional neural networks
* Recurrent neural networks
* LSTMs and GRUs
* Graph representations: nodes, edges, adjacency matrices, incidence matrices
* Message passing neural networks
* Graph convolutional networks
* Graph attention networks
* GraphSAGE and neighborhood sampling
* Positional encodings for graphs
* Weisfeiler-Lehman test and GNN expressivity
* Over-smoothing and over-squashing
* Autoencoders
* Variational autoencoders
* Generative adversarial networks
* Diffusion models
* Neural ODEs
* Applications in molecules, recommendation systems, knowledge graphs, and social networks

### Kernels

Resources:

* [MVA: Machine Learning with Kernel Methods](https://mva-kernel-methods.github.io/course-page/lectures/)
* [MVA 2021–2022: Machine Learning with Kernel Methods](https://mva-kernel-methods.github.io/course-2021-2022/lectures/)
* [Stanford CS229: Machine Learning](https://cs229.stanford.edu/summer2020/syllabus-summer2020.html)
* [Stanford STATS305C: Reproducing Kernel Hilbert Spaces](https://web.stanford.edu/class/stats305c/lectures/RKHS.html)

Topics to study:

* Kernel trick
* Feature maps
* Positive definite kernels
* Gram matrices
* Reproducing kernel Hilbert spaces
* Representer theorem
* Support vector machines
* Kernel ridge regression
* Kernel logistic regression
* Gaussian processes
* Mercer theorem
* Bochner theorem
* Polynomial kernels
* RBF kernels
* Laplacian kernels
* String kernels
* Graph kernels
* Weisfeiler-Lehman graph kernels
* Kernels for structured data
* Kernel representations of probability distributions
* Kernel approximation methods
* Random features
* Connections between kernels and neural networks
* Neural tangent kernels
* Deep kernel methods
* Graph convolutional kernel networks

### Transformers and Language Models

Resources:

* [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
* [Stanford CS25: Transformers United](https://web.stanford.edu/class/cs25/)

Topics to study:

* Self-attention
* Scaled dot-product attention
* Multi-head attention
* Query, key, and value projections
* Positional encodings
* Rotary positional embeddings
* Encoder architectures
* Decoder architectures
* Encoder-decoder architectures
* Transformer blocks
* Residual connections
* Layer normalization
* Feed-forward networks
* Tokenization
* Training objectives for language models
* Autoregressive language modeling
* Scaling laws
* Pretraining
* Instruction tuning
* Preference optimization
* Retrieval-augmented generation
* KV caching
* Efficient attention variants
* Sliding-window attention
* Sparse attention
* State-space models
* Interpretability
* Mechanistic interpretability
* Language model evaluation
* Language model systems and inference

### Reinforcement Learning

Resources:

* [Stanford CS234: Reinforcement Learning](https://web.stanford.edu/class/cs234/modules.html)
* [Berkeley CS285: Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/)

Topics to study:

* Markov decision processes
* Bellman equations
* Dynamic programming
* Policy evaluation
* Policy iteration
* Value iteration
* Monte Carlo methods
* Temporal-difference learning
* Q-learning
* SARSA
* Function approximation
* Deep Q-networks
* Policy gradients
* Actor-critic methods
* Advantage estimation
* Model-based reinforcement learning
* Exploration
* Offline reinforcement learning
* Imitation learning
* Multi-task reinforcement learning
* Meta reinforcement learning
* Reinforcement learning for language models

### Emergent AI Dynamics

Resources:

* [Stanford CS329X: Human-Centered LLMs](https://web.stanford.edu/class/cs329x/)
* [Stanford CS234: Reinforcement Learning](https://web.stanford.edu/class/cs234/modules.html)
* [Berkeley CS285: Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/)
* [Stanford CS25: Transformers United](https://web.stanford.edu/class/cs25/)

Topics to explore:

* Multi-agent simulations
* Multi-agent reinforcement learning
* Emergent behavior from simple agent rules
* Cooperation and competition
* Social dilemmas
* Game-theoretic interactions
* Self-interested agents
* Reputation systems
* Contracting mechanisms
* Mediation
* Governance mechanisms
* Sanctions and incentives
* Network formation and network rewiring
* Learning environments for decision-making agents
* LLM-driven agents with memory and planning
* Agent communication
* Tool-using agents
* Simulated economies
* Simulated markets
* Simulated societies
* Simulated ecosystems
* Reinforcement learning in custom environments
* Human-AI interaction
* Alignment and preference learning
* Evaluation methods for simulation quality
* Evaluation of cooperation and agent behavior
* Sustainability and long-horizon dynamics


## Agent-Assisted Workflow

This repo includes a local `.learn/` directory and a Codex skill at `.agents/skills/ml-notes-learning/`.

Use it for prompts like:

* `$ml-notes-learning /learn-plan I want to understand GraphSAGE`
* `$ml-notes-learning /learn-write based on this paper, create a note under kernels/`
* `$ml-notes-learning /learn-review transformers/attention/readme.md`
* `$ml-notes-learning /learn-add neural-networks/graph-neural-networks/message-passing.md`

In Codex CLI or the IDE extension, you can also type `/skills` and select `ml-notes-learning`.

### Available Codex Skill

#### `$ml-notes-learning`

Use this skill to manage the learning workflow for this repository.

Available workflows:

| Workflow        | What it does                                                                                                                          | Example                                                                     |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `/learn-plan`   | Creates a learning plan for a topic, including prerequisites, roadmap, expected notes or experiments, references, and open questions. | `$ml-notes-learning /learn-plan transformers from first principles`         |
| `/learn-write`  | Writes or expands a note using the repo templates and saves it under the relevant topic folder.                                       | `$ml-notes-learning /learn-write a concept note on message passing in GNNs` |
| `/learn-review` | Reviews an existing note for correctness, structure, examples, citations, depth, and consistency with the style guide.                | `$ml-notes-learning /learn-review transformers/readme.md`                   |
| `/learn-add`    | Adds a finished note to `.learn/index/knowledge-graph.json` so relationships between notes can be tracked.                            | `$ml-notes-learning /learn-add kernels/kernel-trick.md`                     |

The topic folders are not Codex skills. They are normal note areas:

* `machine-learning-foundations/`
* `neural-networks/`
* `kernels/`
* `transformers/`
* `reinforcement-learning/`
* `emergent-ai-dynamics/`

The actual Codex skill lives at `.agents/skills/ml-notes-learning/SKILL.md`.

The `.learn/` directory contains supporting workflow instructions, templates, style rules, and the knowledge graph index used by that skill.
