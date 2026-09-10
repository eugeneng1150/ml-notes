### Graph Convolutional Network (GCN)

### GraphSAGE

### Graph Attention Networks

- **Idea:** Compute the embedding $h_v^{(l)}$ of each node using an **attention strategy**.

    - Nodes attend over their neighbors' messages.
    - Different neighbors can be assigned different levels of importance.

- Compute an **attention coefficient** $e_{vu}$ for each pair of connected nodes $v,u$.

    - $e_{vu}$ represents the importance of node $u$'s message to node $v$.
    - The attention coefficient is computed using an attention function $a$:

```math
e_{vu} = a(m_u^{(l)}, m_v^{(l)})
```

- Normalize the attention coefficients using **softmax** to obtain the final attention weights $\alpha_{vu}$.

```math
\alpha_{vu}
=
\frac{\exp(e_{vu})}
{\sum_{k \in N(v)} \exp(e_{vk})}
```

The attention weights over all neighbors of $v$ sum to 1.

- Use the attention weights to perform a **weighted aggregation** of the neighbors' messages:

```math
h_v^{(l)}
=
\sigma\left(
\sum_{u \in N(v)}
\alpha_{vu} W^{(l)}h_u^{(l-1)}
\right)
```

- The **attention function** $a$ can be implemented as a small neural network.

    - The parameters of $a$ are learned during training.
    - This allows the model to learn which neighbors are more important to a particular node.

- **Multi-head attention** can be used to stabilize training.

    - Multiple attention mechanisms are learned with different parameters.
    - Each attention head produces its own node vector.
    - The outputs of the different heads are combined using concatenation or summation.