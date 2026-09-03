# Graph Neural Networks

## Introduction

Graph neural networks are models for learning from graph-structured data, where the input is made of nodes, edges, and relationships rather than a plain vector or sequence. I started this topic from two directions: Weisfeiler-Lehman graph kernels and introductory material on graph neural networks for machine learning engineers. That pairing is useful because graph kernels show a classical way to compare graphs through structured features, while GNNs show how to learn those representations directly with neural networks.

## Intuition

The central idea is simple: each node updates its representation by looking at its neighbors. If a node is connected to other nodes that carry useful information, then the node should be able to absorb that information over multiple message-passing steps.

The intuition is easiest to see in a small graph:

1. Every node starts with an initial feature vector.
2. Each layer collects messages from neighboring nodes.
3. The messages are aggregated into a single summary.
4. The node updates its own embedding using that summary.
5. A readout step turns node embeddings into node-level, edge-level, or graph-level predictions.

This is close to the Weisfeiler-Lehman view of graphs. WL relabeling updates a node using its current label and the multiset of neighbor labels. GNNs do the same kind of local neighborhood aggregation, but with learned continuous transformations instead of fixed discrete labels.

## How It Works

Most message-passing GNNs follow the same pattern:

1. Initialize node features.
2. Build messages from neighboring nodes and edges.
3. Aggregate the messages with a permutation-invariant operator such as sum, mean, or max.
4. Update node embeddings with a neural network.
5. Repeat for several layers.

For a graph with node features `h_v^(l)` at layer `l`, a common form is:

$$
m_v^{(l+1)} = \operatorname{AGGREGATE}\left(\left\{ M\left(h_v^{(l)}, h_u^{(l)}, e_{vu}\right) : u \in \mathcal{N}(v) \right\}\right)
$$

$$
h_v^{(l+1)} = \operatorname{UPDATE}\left(h_v^{(l)}, m_v^{(l+1)}\right)
$$

Where:

- `N(v)` is the neighborhood of node `v`
- `M` is a message function
- `AGGREGATE` is permutation invariant
- `UPDATE` is usually an MLP, GRU-style update, or residual transformation

Graph-level prediction adds a readout step after the final layer:

$$
h_G = \operatorname{READOUT}\left(\left\{ h_v^{(L)} : v \in V \right\}\right)
$$

This is the core mechanism behind graph convolutional networks, GraphSAGE, graph attention networks, and many related variants.

## Mathematical Formulation

A graph can be written as `G = (V, E)` with optional node features `X` and edge features `E_feat`.

One common message-passing layer can be written as:

$$
h_v^{(l+1)} = U^{(l)}\left(h_v^{(l)}, \operatorname{AGGREGATE}\left(\left\{ M^{(l)}\left(h_v^{(l)}, h_u^{(l)}, e_{vu}\right) : u \in \mathcal{N}(v) \right\}\right)\right)
$$

In a graph convolutional network, a simplified version often looks like:

$$
H^{(l+1)} = \sigma\left(D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)}\right)
$$

Where:

- `A` is the adjacency matrix with self-loops
- `D` is the degree matrix
- `H^(l)` is the matrix of node embeddings at layer `l`
- `W^(l)` is a learnable weight matrix
- `sigma` is a nonlinearity

This formal view makes the main design choices explicit: what information is exchanged, how it is aggregated, and how much structure the model can preserve.

## Complexity

### Time Complexity

For sparse graphs, one message-passing layer is usually linear in the number of edges plus the number of nodes, or roughly `O(|E| + |V|)`, up to the cost of the per-edge and per-node neural transformations.

### Space Complexity

The main memory cost is storing node embeddings, edge information, and the graph connectivity structure. For minibatched or sampled training, the actual memory use also depends on neighborhood sampling and batch size.

## Why It Matters

GNNs are useful because many real problems are naturally graph-shaped. Molecules, social networks, recommendation systems, knowledge graphs, traffic systems, and relational databases all have structure that a standard MLP does not see directly.

The graph view also explains common failure modes. If the graph is too deep, node embeddings can become too similar. If information has to flow through narrow bottlenecks, long-range signals can get compressed. If the graph is heterophilic, naive neighborhood aggregation can become misleading.

## Comparison

| Aspect | Graph kernels | Graph neural networks |
| --- | --- | --- |
| Representation | Hand-designed structural features | Learned node and graph embeddings |
| Update rule | Fixed feature extraction | Learned message passing |
| Training style | Often separate feature generation + classifier | End-to-end optimization |
| Strength | Good interpretability and strong classical baselines | Flexible feature learning and task adaptation |
| Limitation | Less adaptive to data-rich tasks | Can oversmooth or oversquash on deep graphs |

## Applications

1. Molecule property prediction and drug discovery
2. Recommendation systems and user-item graphs
3. Knowledge graphs and relational reasoning
4. Social network analysis and community structure
5. Traffic, routing, and infrastructure modeling

## Key Takeaways

- GNNs learn from graph structure by repeatedly aggregating information from neighboring nodes.
- Weisfeiler-Lehman graph kernels are a useful mental bridge because they also build representations from local neighborhoods.
- The main design choices are the message function, aggregation rule, update function, and readout function.
- The main practical issues are expressivity, graph depth, homophily, oversmoothing, and oversquashing.

## References

- Nino Shervashidze, Pascal Schweitzer, Erik Jan van Leeuwen, Kurt Mehlhorn, and Karsten M. Borgwardt. [Weisfeiler-Lehman Graph Kernels](https://www.jmlr.org/papers/v12/shervashidze11a.html). JMLR, 2011.
- James Tanis, Chris Giannella, Adrian Mariano, and Daoud Meerzaman. [Introduction to Graph Neural Networks for Machine Learning Engineers](https://dl.acm.org/doi/10.1145/3816725). ACM Computing Surveys, 2026.
