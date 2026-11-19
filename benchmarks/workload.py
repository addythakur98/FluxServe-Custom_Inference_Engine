"""Benchmark workload: a stratified prompt suite with a prefix-shared cohort."""

# Shared system prompt (~320 tokens = 20 full KV blocks of 16 tokens) used by the
# prefix-shared cohort, so warm requests can skip most of their prompt compute.
SHARED_SYSTEM_PROMPT = (
    "You are a principal distributed-systems and GPU inference architect. "
    "You design high-throughput, low-tail-latency serving engines for large language models "
    "deployed on NVIDIA A100 and H100 clusters. Follow these mandatory engineering standards in every answer:\n"
    "1. Memory Hierarchy & Paging: Always account for HBM bandwidth vs SRAM capacity, KV-cache block size, "
    "internal/external fragmentation, reference-counted prefix-cache radix trees, and copy-on-write semantics.\n"
    "2. Scheduling & Batching: Analyze iteration-level continuous batching, chunked prefill token budgets, "
    "decode-priority vs prefill-priority head-of-line blocking, and recompute vs swap preemption trade-offs.\n"
    "3. Kernel Execution & Launch Overhead: Quantify Tensor Core tile utilization, Arithmetic Intensity (FLOPs/Byte), "
    "roofline regimes (memory-bandwidth bound decode vs compute-bound prefill), and CUDA Graph static buffer capture.\n"
    "4. Observability & SLOs: Frame all recommendations in terms of P50/P90/P99 Time-To-First-Token (TTFT), "
    "Time-Per-Output-Token (TPOT), Inter-Token Latency (ITL), and goodput under strict latency SLOs.\n"
    "Provide concise, mathematically grounded explanations and production-ready code.\n\n"
)


def build_benchmark_prompts(num_requests: int = 100):
    """
    Build a statistically meaningful benchmark suite (default N=100 requests)
    stratified into 4 equal cohorts:
      1. Short prompts       (25% — ~15-30 tokens)
      2. Medium prompts      (25% — ~60-140 tokens)
      3. Long prompts        (25% — ~220-400 tokens)
      4. Prefix-shared suite (25% — ~340 tokens each sharing SHARED_SYSTEM_PROMPT)
    """
    short_templates = [
        "Explain the difference between latency and throughput in {topic}.",
        "What is the primary bottleneck of {topic} on modern GPUs?",
        "Define {topic} in two concise sentences for a systems engineer.",
        "Why does {topic} matter for tail latency at P99?",
        "Give a concrete production example of {topic}.",
    ]
    medium_templates = [
        "You are a systems engineer reviewing a production service. Explain how {topic} interacts with memory hierarchy, thread scheduling, and tail latency under bursty traffic. Include concrete trade-offs.",
        "Write an idiomatic Python implementation demonstrating {topic}, including type annotations, edge-case handling, and a brief analysis of time and space complexity.",
        "Compare two industry approaches to {topic} in large-scale distributed infrastructure. When does the simpler design outperform the complex one?",
    ]
    long_templates = [
        "You are leading an architecture review for a multi-tenant AI serving platform handling 5,000 requests per second. "
        "The service currently suffers from severe P99 latency spikes whenever long-context requests arrive alongside short interactive chat queries. "
        "Specifically, engineers suspect {topic} is interacting poorly with GPU memory allocation, kernel launch scheduling, and network serialization. "
        "Provide a step-by-step diagnostic plan, identify the exact hardware and software bottlenecks, and propose a concrete redesign with expected P50 and P99 improvements.",
    ]
    prefix_questions = [
        "User: Why does PagedAttention eliminate external fragmentation in KV caches?\nAssistant:",
        "User: How does a Radix Tree prefix cache decide which leaf blocks to evict under memory pressure?\nAssistant:",
        "User: Why do CUDA Graphs reduce TPOT more at small batch sizes than at very large batch sizes?\nAssistant:",
        "User: How should max_batched_tokens be tuned to balance prefill TTFT against decode ITL spikes?\nAssistant:",
        "User: Compare recompute preemption vs PCIe host-swap preemption when KV blocks run out.\nAssistant:",
        "User: How does Grouped-Query Attention (GQA) change the arithmetic intensity of decode attention?\nAssistant:",
        "User: Explain the online softmax rescaling invariant in FlashAttention-2.\nAssistant:",
        "User: Why must the last prompt token still be computed even on a 100% prefix-cache match?\nAssistant:",
        "User: How do padding rows in bucketed CUDA Graphs avoid corrupting live KV cache blocks?\nAssistant:",
        "User: What determines the maximum concurrent sequences before KV preemption begins?\nAssistant:",
    ]
    topics = [
        "PagedAttention block tables", "FlashAttention SRAM tiling", "CUDA Graph kernel launch amortization",
        "Grouped-Query Attention (GQA)", "Radix-tree prefix caching", "iteration-level continuous batching",
        "RoPE rotary position embeddings", "SwiGLU activation fusion", "HBM memory bandwidth saturation",
        "Tensor Core warp scheduling", "KV cache quantization", "speculative decoding",
        "token-budget chunked prefill", "LRU block eviction", "dynamic batch bucketing",
        "PCIe pinned memory transfers", "top-k Mixture-of-Experts routing", "RMSNorm numerical stability",
        "ring attention across GPUs", "tail-latency SLO admission control", "zero-copy tensor slicing",
        "asynchronous CUDA streams", "PyTorch SDPA kernel dispatch", "autoregressive decode roof-line",
        "multi-turn conversation KV reuse",
    ]

    n_cohort = max(1, num_requests // 4)
    prompts = []

    for i in range(n_cohort):
        t = topics[i % len(topics)]
        prompts.append(short_templates[i % len(short_templates)].format(topic=t))

    for i in range(n_cohort):
        t = topics[(i + 5) % len(topics)]
        prompts.append(medium_templates[i % len(medium_templates)].format(topic=t))

    for i in range(n_cohort):
        t = topics[(i + 11) % len(topics)]
        prompts.append(long_templates[i % len(long_templates)].format(topic=t) + f" (Case #{i+1})")

    while len(prompts) < num_requests:
        idx = len(prompts) - 3 * n_cohort
        q = prefix_questions[idx % len(prefix_questions)]
        # Keep the shared prefix identical across all prefix-cohort requests; vary only the user suffix
        prompts.append(SHARED_SYSTEM_PROMPT + f"[Query #{idx+1}] " + q)

    return prompts[:num_requests]
