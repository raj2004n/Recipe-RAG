import gc
import time
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import numpy as np

def profile_model_memory(model_id: str, input_tokens: int) -> dict:
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot profile VRAM accurately.")
        return

    # clear CUDA cache before starting
    torch.cuda.empty_cache()
    gc.collect()

    initial_memory = torch.cuda.memory_allocated() / (1024**3)
    print(f"Initial Baseline VRAM: {initial_memory:.3f} GB")

    # track static weight memory
    print("Loading model and tokenizer...")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    load_time = time.time() - start_time
    post_load_memory = torch.cuda.memory_allocated() / (1024**3)
    static_weight_vram = post_load_memory - initial_memory

    print(f"Model loaded in: {load_time:.2f} seconds")
    print(f"Post-Load VRAM: {post_load_memory:.3f} GB")
    print(f"-> Static Weight VRAM Footprint: {static_weight_vram:.3f} GB")

    # track Dynamic Memory (Prefill & KV Cache)
    print(f"\nSimulating a retrieval prompt ({input_tokens} tokens)...")

    # construct dummy input tensor representing a long retrieval context
    dummy_input_ids = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(1, input_tokens),
        device=model.device,
    )

    # reset max memory tracker to isolate the forward pass peak
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        # complete a single prefill forward pass to construct the initial KV cache
        outputs = model(dummy_input_ids, use_cache=True)

    peak_prefill_vram = torch.cuda.max_memory_allocated() / (1024**3)
    dynamic_overhead = peak_prefill_vram - post_load_memory

    print(f"Peak VRAM during prefill pass: {peak_prefill_vram:.3f} GB")
    print(
        f"-> Dynamic Overhead (KV Cache + Activation Tensors): {dynamic_overhead:.3f} GB"
    )

    return {
        "input_tokens": input_tokens,
        "load_time_s": round(load_time, 2),
        "initial_memory_gb": round(initial_memory, 2),
        "post_load_vram_gb": round(post_load_memory, 2),
        "static_weight_vram_gb": round(static_weight_vram, 2),
        "dynamic_overhead_gb": round(dynamic_overhead, 2),
        "peak_prefill_vram_gb": round(peak_prefill_vram, 2),
    }

def profile_dataset_tokenization(model_id: str, chunks: dict) -> dict:
    # load config and tokenizer
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # process dataset tokenization lengths
    token_counts = []

    # extract text from chunks (is what is being tokenized)
    texts = [c["text"] for c in chunks]

    for text in texts:
        # add_special_tokens=True to account for BOS/EOS tokens
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_counts.append(len(tokens))

    # max positon embedding
    max_context = getattr(config, "max_position_embeddings", None)

    # calculate distribution statistics
    total_seqs = len(token_counts)
    total_tokens = sum(token_counts)
    mean_tokens_per_chunk = np.mean(token_counts)
    median_tokens_per_chunk = np.median(token_counts)
    std_tokens_per_chunk = np.std(token_counts)
    max_tokens_per_chunk = np.max(token_counts)
    min_tokens_per_chunk = np.min(token_counts)
    p95_tokens_per_chunk = np.percentile(token_counts, 95)
    p99_tokens_per_chunk = np.percentile(token_counts, 99)
    num_truncated = sum(n > max_context for n in token_counts)

    print(f"Total Sequences Evaluated : {total_seqs}")
    print(f"Total Tokens Evaluated    : {total_tokens}")
    print(f"Mean Tokens per Chunk     : {mean_tokens_per_chunk:.2f}")
    print(f"Median Tokens per Chunk   : {median_tokens_per_chunk:.1f}")
    print(f"Std Tokens per Chunk      : {std_tokens_per_chunk:.2f}")
    print(f"Max Tokens per Chunk      : {max_tokens_per_chunk:.1f}")
    print(f"Min Tokens per Chunk      : {min_tokens_per_chunk:.1f}")
    print(f"95th Percentile Tokens    : {p95_tokens_per_chunk:.1f}")
    print(f"99th Percentile Tokens    : {p99_tokens_per_chunk:.1f}")
    print(f"Num Truncated Chunks      : {num_truncated}")

    return {
        "total_seqs": total_seqs,
        "total_tokens": total_tokens,
        "mean_tokens_per_chunk": mean_tokens_per_chunk,
        "median_tokens_per_chunk": median_tokens_per_chunk,
        "std_tokens_per_chunk": std_tokens_per_chunk,
        "max_tokens_per_chunk": max_tokens_per_chunk,
        "min_tokens_per_chunk": min_tokens_per_chunk,
        "p95_tokens_per_chunk": p95_tokens_per_chunk,
        "p99_tokens_per_chunk": p99_tokens_per_chunk,
        "num_truncated": num_truncated, 
    }