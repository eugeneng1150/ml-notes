# Graph Neural Networks

This folder collects my notes on graph neural networks and graph representation learning. The main question I want to keep returning to is:

> How do we learn useful representations when the input is not a vector, image, or sequence, but a graph of entities and relationships?

I started this topic from two directions: Weisfeiler-Lehman graph kernels and introductory GNN material for machine learning engineers. That makes a useful learning path because graph kernels show one classical way to compare graphs through structured features, while GNNs show how neural networks can learn graph representations through message passing.

## Mental Model

A graph is made of nodes, edges, and optional features on either of them. A GNN updates each node representation by repeatedly mixing information from its neighbors.
gi
At a high level:

1. Each node starts with an initial feature vector.
2. Each layer gathers messages from neighboring nodes.
3. The model aggregates those messages.
4. The node updates its representation.
5. A decoder turns node, edge, or graph representations into predictions.

This means the graph structure is not just metadata. It determines which information can move, how far it can move, and what the model is able to distinguish.

## Why Weisfeiler-Lehman Matters

The Weisfeiler-Lehman idea gives a clean bridge from graph kernels to GNNs.

In the WL relabeling procedure, each node updates its label using its current label and the multiset of labels from its neighbors. After several iterations, the graph has labels that encode increasingly larger local neighborhoods. WL graph kernels use these labels to compare graphs by counting matching patterns.

That sounds very close to message passing:

| Classical WL / graph kernels | Graph neural networks |
| --- | --- |
| Discrete node labels | Continuous node embeddings |
| Neighbor label multiset | Neighbor message aggregation |
| Relabeling step | Neural update function |
| Count matching graph patterns | Learn task-specific representations |
| Fixed feature extraction | End-to-end learned features |

The important connection is that both approaches build node information from local neighborhoods. The difference is that graph kernels usually define the features by hand, while GNNs learn how to transform and combine features for a task.

## Core Concepts To Learn

- Graph representation: nodes, edges, adjacency, edge index, node features, edge features
- Prediction tasks: node classification, graph classification, link prediction, edge prediction
- Message passing: message function, aggregation function, update function
- Readout functions: turning node embeddings into graph-level predictions
- Expressivity: what graph structures a GNN can or cannot distinguish
- Homophily and heterophily: whether connected nodes tend to share labels or differ
- Oversmoothing: deep layers can make node embeddings too similar
- Oversquashing: long-range information can be compressed through narrow graph bottlenecks
- Inductive vs transductive learning: generalizing to new graphs or only known nodes

## Learning Path

1. Review graph basics: node features, edge features, adjacency matrices, edge lists, neighborhoods, and graph-level labels.
2. Understand WL graph kernels as a classical similarity-based approach to graph learning.
3. Map WL relabeling to the message-passing view of GNNs.
4. Study a basic graph convolutional network and track how node embeddings change after one layer.
5. Study GraphSAGE to understand inductive learning and neighborhood sampling.
6. Study graph attention networks to understand learned neighbor weighting.
7. Compare node classification, graph classification, and link prediction decoders.
8. Investigate common failure modes: low homophily, label scarcity, oversmoothing, and oversquashing.
9. Implement small examples with a toy graph before moving to benchmark datasets.

## Planned Notes

- `weisfeiler-lehman-graph-kernels.md`
- `message-passing.md`
- `graph-convolutional-networks.md`
- `graphsage.md`
- `graph-attention-networks.md`
- `oversmoothing-and-oversquashing.md`
- `node-vs-graph-vs-link-prediction.md`

## Open Questions

- How exactly does WL expressivity limit what message-passing GNNs can distinguish?
- When is a graph kernel still preferable to a GNN?
- How much depth is useful before oversmoothing or oversquashing becomes harmful?
- What changes when the graph is heterophilic instead of homophilic?
- How should graph structure be combined with rich node text or image features?

## References

- Nino Shervashidze, Pascal Schweitzer, Erik Jan van Leeuwen, Kurt Mehlhorn, and Karsten M. Borgwardt. [Weisfeiler-Lehman Graph Kernels](https://www.jmlr.org/papers/v12/shervashidze11a.html). JMLR, 2011.
- James Tanis, Chris Giannella, Adrian Mariano, and Daoud Meerzaman. [Introduction to Graph Neural Networks for Machine Learning Engineers](https://dl.acm.org/doi/10.1145/3816725). ACM Computing Surveys, 2026.
