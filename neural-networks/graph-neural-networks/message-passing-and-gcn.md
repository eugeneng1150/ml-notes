# Message Passing and Graph Convolutional Networks

## Questions

1. Why are ordinary neural network architectures awkward to apply directly to graphs?
2. What information does a node have at layer 0 of a GCN?
3. What does it mean for a node to aggregate information from its neighbors?
4. What is a node's computation graph?
5. How does one GCN layer update a node representation?
6. What is the role of the neighbor aggregation term?
7. What is the role of the node's own embedding in the update?
8. Why do we usually apply a learnable linear transformation to aggregated information?
9. Why do we apply a non-linearity such as ReLU after the update?
10. What does the equation h_v^(k+1) = sigma(W_k * mean_{u in N(v)} h_u^(k) + B_k h_v^(k)) mean in words?
11. Why is mean aggregation permutation invariant?
12. Why are the same weight matrices shared across different nodes?
13. How is learning shared GCN weights different from learning one embedding vector per node?
14. What information can a node access after one GCN layer?
15. What information can it access after two layers?
16. How does stacking K layers relate to the K-hop neighborhood of a node?
17. Why can stacking layers be viewed as progressively enlarging the receptive field?
18. How does the final node embedding depend on both the node's own features and its neighborhood?
19. How can GCN embeddings be used for node-level prediction?
20. How can node embeddings be combined to produce graph-level predictions?
21. Why does message passing naturally support graphs with different numbers of nodes and different node degrees?
22. What are the main differences between a shallow node embedding method and a GCN?
23. In what sense is a GCN learning a function for generating embeddings rather than memorizing embeddings?
24. What assumptions does simple neighborhood averaging make about useful graph structure?
25. What are the main takeaways I should remember about message passing and GCNs?
