# Transformers

Notes about transformer architecture, self-attention, positional encodings, language models, efficient attention, RAG, interpretability, and transformer implementation.

## Study Focus

- Start with the sequence modeling problem and the need for token-to-token interaction.
- Build from embeddings -> attention -> MLP -> residual stream -> layer stack.
- Track tensor shapes throughout explanations.
- Separate architecture, training objective, inference behavior, and systems concerns.

## Note Checklist

- Annotate attention shapes: `Q`, `K`, `V`, attention logits, probabilities, and output.
- Explain why scaling by `sqrt(d_k)` appears before using the formula heavily.
- Distinguish encoder, decoder, and encoder-decoder use cases.
- For implementation notes, mention masking, KV cache, batching, precision, and memory.
- For modern LLM topics, verify model or library details with primary sources.

## Subtopics

- [Attention](attention/readme.md)
- [Flash Attention](flash_attn/readme.md)
- [Omni](omni/readme.md)
- [Special Tokens](special_tokens/readme.md)
