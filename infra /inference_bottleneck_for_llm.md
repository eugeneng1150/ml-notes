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
Dense models are interconnect and memory bandwidth bound and favor high degree of Tensor parallelism while sparse MoE are limited by routing and synchronization latency and benefit from hybrid strategies

## Problem
TODO: Explain the specific inference scaling problem this paper addresses.


## Core Idea and background
TODO: Build the background needed to understand the paper.

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
