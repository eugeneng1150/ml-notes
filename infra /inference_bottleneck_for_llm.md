# Understanding Inference Scaling for LLMs: Bottlenecks, Trade-offs, and Performance Principles

## Metadata
- Paper: Understanding Inference Scaling for LLMs: Bottlenecks, Trade-offs, and Performance Principles
- Authors: Moiz Arif, Avinash Maurya, Sudharshan Vazhkudai, Bogdan Nicolae
- Year: 2026
- Venue: ISCA '26 Industry Track
- DOI: 10.1109/ISCA66397.2026.00084
- arXiv: 2605.19775
- Link: https://arxiv.org/pdf/2605.19775
- Area: LLM inference scaling, inference systems, parallelism, KV cache, reasoning workloads
- Status: reading


## Summary
Data Parallelism is throughput efficient for small models but hits a capacity trap on reasoning workloads as KV-Cache forces early throttling resulting in sub optimal compute utilization.   
Tensor parallelism unlocks stranded memory and delivers sublinear gains near the 32B crossover.  
Dense models are interconnect and memory bandwidth bound and favor high degree of Tensor parallelism while sparse MoE are limited by routing and synchronization latency and benefit from hybrid strategies.

## Problem
Due to models having long chain of thoughts (CoT), the generation of thousands of thinking tokens creates massive persistent KV cache that saturates High Bandwith Memory (HDB). We cannot solely rely on Data Parallelism as it will result in memory thrashing, schedular preemption and non linear latency spikes (Might need to read up on these). Thus, there is a need to search through combindations of Data, Tensor and Pipeline parallelism which is tailored specifically to the sparsity and sequence length characteristic of the model. There are 3 main problems:  
1) Capacity wall: Reasoning traces with long Output sequence length cause KV cache to grow linearly, exhausting the HBM of the GPUs. 
2) Parallelism Efficiency Gap: Although Tensor Parallelism (TP) alleviates memory pressure by using HDM of multiple GPUs, it incurs communication overhead.
3) Architectural Divergence: Different models have different architecture, hence infernece needs to be tailored to it.

## Core Idea and background
TODO: Build the background needed to understand the paper.
1. Prefill phase: model process the user prompt in parallel and is strictly **compute bound** 
2. Decoding phase: **Bandwidth bound** each token generatd requires reading the entire model weight and active KV cache from the HBM. 
3. Dense Architectures (Grouped-Query Attention, GQA)

- In standard **Multi-Head Attention (MHA)**, each query head has its own corresponding key and value head:

  $$
  Q_1 \rightarrow (K_1, V_1), \quad
  Q_2 \rightarrow (K_2, V_2), \quad
  Q_3 \rightarrow (K_3, V_3), \quad
  Q_4 \rightarrow (K_4, V_4)
  $$

- **GQA reduces KV-cache memory** by allowing multiple query heads to **share a smaller number of key/value heads**. For example:

  $$
  Q_1, Q_2 \rightarrow (K_1, V_1)
  $$

  $$
  Q_3, Q_4 \rightarrow (K_2, V_2)
  $$

  Hence, **4 query heads only require 2 KV heads instead of 4**.

- This reduces the **KV-cache footprint per token**, which mitigates GPU memory pressure.

- However, GQA **does not eliminate KV-cache growth**. A K/V representation still needs to be stored for every token at every layer, so KV-cache memory still grows **linearly with sequence length and the number of layers**.

- Therefore, for very long reasoning sequences, the KV cache can still eventually saturate GPU memory even with GQA.
4. Sparse Architectures (Multi-Head Latent Attention)

- **MLA compresses the KV cache into low-rank latent vectors.**

- In **Multi-Head Attention (MHA)**, each attention head has its own K and V representations:

  $$
  [K_1,V_1,K_2,V_2,K_3,V_3,K_4,V_4]
  $$

  Therefore, the **KV-cache size grows with the number of attention heads**.

- In **Multi-Head Latent Attention (MLA)**, instead of caching all the individual K and V representations, the KV information is compressed into a **low-dimensional latent representation** $c^{KV}$.

  You can think of it as compressing:

  $$
  \underbrace{[K_1,V_1,K_2,V_2,K_3,V_3,K_4,V_4]}_{\text{large}}
  $$

  into:

  $$
  \underbrace{c^{KV}}_{\text{small compressed representation}}
  $$

- Therefore, the **KV-cache size does not scale directly with the number of attention heads**, significantly reducing the memory required per token.

- However, MLA does **not** eliminate KV-cache growth with sequence length. A latent representation still needs to be cached for each token, so the KV cache still grows as the sequence becomes longer.

5. **Data Parallelism (DP):** As seen in the other paper [OrionInfer](./orioninfer.md), DP is employed when there are a large number of concurrent requests. Each GPU holds a full copy of the model, allowing different GPUs to process different requests independently and achieve higher throughput. However, this paper focuses on the **memory limitation** of DP: since every GPU must store a full copy of the model weights, less GPU memory is available for the KV cache. This does not contradict OrionInfer; DP is still beneficial for high request loads, but its effectiveness can eventually be limited by KV-cache saturation for long reasoning workloads.  

6. **Tensor Parallelism (TP):** Splits the computation **within each individual layer** across multiple GPUs. Therefore, multiple GPUs work together to process the same layer for a request. This allows the model to use the aggregate memory and compute of multiple GPUs, but introduces **high communication overhead** because the GPUs must synchronize their partial results at every layer.

7. **Pipeline Parallelism (PP):** Splits the model **between layers**, where different groups of layers are assigned sequentially to different GPUs. For example, GPU 1 may hold Layers 1–10 while GPU 2 holds Layers 11–20. PP reduces memory usage per GPU without the high communication frequency of TP. However, it can introduce **pipeline bubbles**, where some GPUs remain idle while waiting for work from another pipeline stage. Multiple concurrent requests can fill these bubbles, but waiting for enough requests may introduce unacceptable **queuing latency**.

|                | DP                            | TP                             | PP                                 |
| -------------- | ----------------------------- | ------------------------------ | ---------------------------------- |
| What is split? | **Requests**                  | **Computation within layers**  | **Layers**                         |
| Model on GPUs  | Full model replica            | Each layer sharded             | Different layers on different GPUs |
| One request    | One replica/GPU group         | Multiple GPUs simultaneously   | Moves GPU → GPU sequentially       |
| Communication  | Low                           | **High**                       | Lower than TP                      |
| Main advantage | Throughput                    | Low latency + aggregate memory | Memory + lower communication       |
| Main problem   | Model replication / KV memory | All-Reduce overhead            | **Pipeline bubbles**               |

Useful concepts to define:
1. Prefill vs decoding
2. Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT)
3. KV cache growth and fragmentation
4. Data parallelism (DP), tensor parallelism (TP), and pipeline parallelism (PP)
5. Dense models vs sparse Mixture-of-Experts (MoE) models
6. Reasoning-centric workloads and Chain-of-Thought token generation


## How It Works
TODO: Walk through the paper's characterization methodology and main findings.

Suggested structure:
1. Experimental setup
    - Models evaluated
    - GPU cluster setup
    - Workloads and prompt/output token regimes
    - Metrics

2. Bottleneck regimes
    - Compute-bound prefill
    - Memory-bandwidth-bound decoding
    - Capacity-bound reasoning workloads
    - Interconnect and synchronization limits

3. Parallelism tradeoffs
    - When DP is efficient
    - When DP hits the KV-cache capacity trap
    - When TP unlocks stranded memory
    - When high-degree TP becomes necessary
    - Why MoE models need hybrid strategies

4. Decision framework
    - Small models
    - Mid-scale dense models
    - Frontier dense models
    - Sparse MoE models
    - Long reasoning workloads


## What I Learned
TODO: Fill this after reading the paper.

Possible takeaways to check against the paper:
1. Reasoning workloads change the dominant bottleneck from prefill compute to decoding capacity.
2. DP can look throughput-efficient for small models but fail once KV cache pressure and fragmentation dominate.
3. TP can improve effective memory capacity and performance around larger model sizes, but gains are sublinear.
4. Dense frontier models and sparse MoE models bottleneck differently.
5. Inference infrastructure needs workload-aware parallelism rather than one fixed serving strategy.
