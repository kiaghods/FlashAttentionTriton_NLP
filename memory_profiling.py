import torch, gc, time, logging
from mqa_forward import mqa_flash_attention
from flashattention_kernel import flash_attention
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("mem")

GB = 1024 ** 3
def gb(x): return x / GB

# HELPERS ------------------------------------
def print_mem(tag):
    log.info(f"{tag:<28} "
             f"allocated={gb(torch.cuda.memory_allocated()):5.2f} GB   "
             f"reserved={gb(torch.cuda.memory_reserved()):5.2f} GB")

def top_blocks(snapshot, k=8):
    blocks = [b for seg in snapshot for b in seg["blocks"] if b["state"] == "active_allocated"]
    blocks.sort(key=lambda b: b["size"], reverse=True)
    return blocks[:k]

def segment_breakdown(snapshot):
    seg_bytes = defaultdict(int)
    for seg in snapshot:
        seg_bytes[seg["segment_type"]] += seg["active_size"]
    return {k: gb(v) for k, v in seg_bytes.items()}

# MAIN PROFILER --------------------------------
def profile_attention(attention_fn, name, batch_size=4, num_heads=16, seq_len=1024, head_dim=64, dtype=torch.float16, causal=False):
    torch.cuda.empty_cache(); gc.collect()
    
    log.info(f"\n--- Profiling {name} seq={seq_len} ---")
    print_mem("startup")

    if name == "MQA":
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch_size, 1, seq_len, head_dim, dtype=dtype, device="cuda")
        v = torch.randn_like(k)
    else:  # Regular FlashAttention
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="cuda")
        v = torch.randn_like(k)

    print_mem("after tensor setup")

    with torch.no_grad():
        if name == "MQA":
            _ = mqa_flash_attention(q[:1], k[:1], v[:1], causal=causal)
        else:
            _ = flash_attention(q[:1], k[:1], v[:1], causal=causal)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.reset_accumulated_memory_stats()
    start = time.time()
    with torch.no_grad():
        if name == "MQA":
            out = mqa_flash_attention(q, k, v, causal=causal)
        else:
            out = flash_attention(q, k, v, causal=causal)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    alloc_peak = gb(torch.cuda.max_memory_allocated())
    reserv_peak = gb(torch.cuda.max_memory_reserved())
    alloc_after = gb(torch.cuda.memory_allocated())
    reserv_after = gb(torch.cuda.memory_reserved())

    print_mem("after forward")
    log.info(f"peak-allocated            = {alloc_peak:5.2f} GB")
    log.info(f"peak-reserved             = {reserv_peak:5.2f} GB")
    log.info(f"forward wall-time         = {elapsed:6.3f} s")

    del q, k, v, out
    torch.cuda.empty_cache(); gc.collect()
    
    return {
        "alloc_after": alloc_after,
        "reserv_after": reserv_after,
        "alloc_peak": alloc_peak,
        "reserv_peak": reserv_peak
    }

# RUNNING -------------------------------------------

if __name__ == "__main__":
    seq_lens = [1024, 2048, 4096, 8192]
    results = {}
    
    for seq_len in seq_lens:
        results[seq_len] = {
            "MQA": profile_attention(mqa_flash_attention, "MQA", seq_len=seq_len, causal=False),
            "FlashAttention": profile_attention(flash_attention, "FlashAttention", seq_len=seq_len, causal=False)
        }
    
    # summary
    log.info("\n" + "="*80)
    log.info("MEMORY USAGE SUMMARY (GB)")
    log.info("="*80)
    log.info(f"{'Seq Len':>8} | {'Model':<15} | {'Alloc After':>12} | {'Reserv After':>12} | {'Peak Alloc':>12} | {'Peak Reserv':>12}")
    log.info("-"*80)
    
    for seq_len in seq_lens:
        for model in ["MQA", "FlashAttention"]:
            r = results[seq_len][model]
            log.info(f"{seq_len:8d} | {model:<15} | {r['alloc_after']:12.2f} | {r['reserv_after']:12.2f} | {r['alloc_peak']:12.2f} | {r['reserv_peak']:12.2f}")
