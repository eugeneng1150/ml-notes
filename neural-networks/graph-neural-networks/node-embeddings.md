# Node Embeddings

## Questions

1. Why do we want to learn node embeddings instead of hand-engineering graph features every time?
2. What does it mean to map a node into a vector space?
3. What should a good node embedding preserve about the original graph?
4. How can similarity in the graph be represented as similarity in embedding space?
5. What is the encoder-decoder framework for learning node embeddings?
6. What does the encoder do?
7. What does the decoder do?
8. Why is the definition of node similarity important?
9. What are different possible meanings of node similarity, such as being connected, sharing neighbors, or having similar structural roles?
10. What is a shallow node embedding?
11. Why can a shallow encoder be viewed as an embedding lookup table?
12. What are the learnable parameters in a shallow embedding method?
13. How do DeepWalk-style methods use random walks to define node neighborhoods?
14. What is the intuition behind predicting nodes that appear in the same random-walk context?
15. What is the optimization objective trying to achieve in random-walk-based node embeddings?
16. Why is negative sampling useful when training node embeddings?
17. What is node2vec trying to improve over a basic random walk?
18. What is the intuition behind BFS-like and DFS-like exploration in node2vec?
19. What roles do the parameters p and q play in node2vec?
20. Why does the choice of random-walk strategy change what kind of similarity the embedding captures?
21. Why are methods such as DeepWalk and node2vec transductive?
22. What happens when a new node is added after training?
23. Why can random-walk embeddings fail to capture structural similarity between distant nodes?
24. What limitations of shallow node embeddings motivate graph neural networks?
25. What are the main takeaways I should remember about node embeddings before moving on to GNNs?
