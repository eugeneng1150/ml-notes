# Permutation Invariance and Equivariance

## Questions

1. Why does a graph not have a canonical ordering of its nodes?
2. What changes in the adjacency matrix and node feature matrix when node order is permuted?
3. What properties should a graph-learning function have when node labels or ordering change?
4. What does permutation invariance mean?
5. What does permutation equivariance mean?
6. What is the difference between permutation invariance and permutation equivariance?
7. Why should a whole-graph representation usually be permutation invariant?
8. Why should node-level representations usually be permutation equivariant?
9. What does the equation f(A, X) = f(PAP^T, PX) mean?
10. What does the equation Pf(A, X) = f(PAP^T, PX) mean?
11. Why is the embedding computation for one specific node invariant to the order of its neighbors?
12. Why is aggregation with sum or mean permutation invariant?
13. If the embedding of each individual node stays the same, why is the output for all nodes called permutation equivariant rather than invariant?
14. What exactly changes in the output embedding matrix after the input node order is permuted?
15. Why is it important that rows of the node feature matrix and rows of the output embedding matrix remain aligned?
16. Why is an ordinary MLP on a flattened graph representation sensitive to node ordering?
17. How does permutation equivariance help a GNN treat different node orderings as the same underlying graph?
18. Can I explain invariance and equivariance using a three-node example without relying only on the formal equations?
19. When would I want invariance instead of equivariance in a GNN pipeline?
20. What are the main takeaways I should remember about invariance and equivariance?
