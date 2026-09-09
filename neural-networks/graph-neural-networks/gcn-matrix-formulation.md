# GCN Matrix Formulation

## Questions

1. Why do we want to rewrite the node-by-node GCN update in matrix form?
2. What does H^(k) represent?
3. What is the shape of H^(k) if there are |V| nodes and each node embedding has dimension d_k?
4. What does the adjacency matrix A represent in the GCN computation?
5. What happens when we multiply A by H^(k)?
6. Why does the row for node v in AH contain information from v's neighbors?
7. Can I work through a three-node numerical example of AH by hand?
8. Why does AH compute a sum of neighbor embeddings rather than a mean?
9. What is the degree matrix D?
10. What does D^(-1)A do?
11. Why does A_tilde = D^(-1)A correspond to mean aggregation?
12. What does A_tilde H^(k) represent for each node?
13. What is the role of W_k in A_tilde H^(k) W_k^T?
14. What is the role of H^(k) B_k^T?
15. Why are the neighbor transformation and self transformation added together?
16. What does the full equation H^(k+1) = sigma(A_tilde H^(k) W_k^T + H^(k) B_k^T) mean in words?
17. What are the dimensions of A_tilde, H^(k), W_k, B_k, and H^(k+1)?
18. Why does the order of matrix multiplication matter in the GCN equation?
19. How is the matrix equation equivalent to computing the GCN update separately for every node?
20. Why is sparse matrix multiplication useful for GCNs?
21. Why is the adjacency matrix typically sparse for real-world graphs?
22. What part of the matrix formulation performs neighborhood aggregation?
23. What part preserves and transforms the node's own information?
24. How does this matrix formulation connect back to permutation equivariance?
25. Why can some more complex GNN aggregation functions not be written in such a simple matrix form?
26. What are the main takeaways I should remember about the matrix formulation of a GCN?
