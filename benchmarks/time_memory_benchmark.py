import torch
import triton
import triton.language as tl 
import pandas as pd
import numpy as np
import argparse
import math
import os
import logging
import sys 

# Get the directory of the current script (benchmarks/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (workspace root FlashAttention_NLP/)
parent_dir = os.path.dirname(current_script_dir)
# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# --- End of sys.path modification ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Try to import PIP FlashAttention ---
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func as pip_flashattention_functional
    # Previously was: from flash_attn.flash_attn_triton import flash_attn_func as pip_flashattention_functional
    PIP_FLASH_ATTENTION_AVAILABLE = True
    logger.info("Successfully imported 'flash_attn_unpadded_func' from 'flash_attn.flash_attn_interface'.")
except ImportError:
    logger.warning(
        "Could not import 'flash_attn_unpadded_func' from 'flash_attn.flash_attn_interface'. "
        "Pip FlashAttention providers will be skipped. Ensure 'flash-attn==1.0.0' is installed."
    )
    PIP_FLASH_ATTENTION_AVAILABLE = False
    pip_flashattention_functional = None

# --- Import from local files ---
try:
    from flashattention_kernel import flash_attention as custom_kernel_mha
    from flashattention_kernel import self_attention_slow as naive_mha_pytorch
    from flashattention_kernel import self_attention_fast as sdpa_pytorch
    logger.info("Successfully imported MHA kernels from 'flashattention_kernel.py'.")
except ImportError as e:
    logger.error(f"Could not import from 'flashattention_kernel.py': {e}. Ensure the file is in PYTHONPATH.")
    # Define dummy functions if import fails to allow script to load, but benchmarks will fail
    custom_kernel_mha = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("custom_kernel_mha not imported"))
    naive_mha_pytorch = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("naive_mha_pytorch not imported"))
    sdpa_pytorch = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sdpa_pytorch not imported"))


try:
    from mqa_forward import mqa_flash_attention as custom_mqa_kernel
    from mqa_forward import naive_mqa_attention
    logger.info("Successfully imported MQA kernels from 'mqa_forward.py'.")
except ImportError as e:
    logger.error(f"Could not import from 'mqa_forward.py': {e}. Ensure the file is in PYTHONPATH.")
    custom_mqa_kernel = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("custom_mqa_kernel not imported"))
    naive_mqa_attention = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("naive_mqa_attention not imported"))


# --- Helper functions for accuracy comparison ---
def max_diff(a, b):
    if a is None or b is None: return float('inf')
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor): return float('inf')
    if a.shape != b.shape: return float('inf')
    return (a - b).abs().max().item()

def avg_diff(a, b):
    if a is None or b is None: return float('inf')
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor): return float('inf')
    if a.shape != b.shape: return float('inf')
    return (a - b).abs().mean().item()

def avg_pctg_error(a, b):
    if a is None or b is None: return float('inf')
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor): return float('inf')
    if a.shape != b.shape: return float('inf')
    epsilon = torch.finfo(a.dtype).eps if a.dtype.is_floating_point else 1e-6 # general epsilon
    denominator = torch.max(a.abs(), b.abs())
    safe_denominator = torch.where(denominator < epsilon, epsilon, denominator) # Avoid division by zero or too small numbers
    return ((a - b).abs() / safe_denominator).mean().item() * 100 # As percentage

# Global dictionary to store extra metrics
extra_metrics_store = {}

def generate_configs(mqa_eval_flag, current_n_heads, batch_size_val, head_dim_val, seq_len_options, benchmark_mode_val):
    configs = []
    BATCH, N_HEADS, D_HEAD = batch_size_val, current_n_heads, head_dim_val

    # Determine modes for benchmark_mode_val
    if benchmark_mode_val == "forward":
        modes = ["fwd"]
    elif benchmark_mode_val == "full":
        modes = ["fwd", "bwd"]
    else:
        raise ValueError(f"Invalid benchmark_mode: {benchmark_mode_val}. Choose 'forward' or 'full'.")

    if mqa_eval_flag:
        providers = {
            "custom_mqa_kernel": ("Custom MQA Kernel", ("purple", "-")),
            "naive_mqa_pytorch": ("Naive Pytorch MQA", ("cyan", "--")),
            "sdpa_expanded_for_mqa": ("Pytorch SDPA (K/V Expanded for MQA)", ("orange", "-.")),
            # "sdpa_broadcast_mqa": ("Pytorch SDPA MQA (Broadcast)", ("blue", "-")),
            "custom_kernel_mha_expanded_for_mqa": ("Custom Kernel MHA (K/V Expanded for MQA)", ("magenta", ":")),
            # "naive_mha_pytorch_expanded_for_mqa": ("Naive Pytorch MHA (K/V Expanded for MQA)", ("brown", "--")),
        }
        if PIP_FLASH_ATTENTION_AVAILABLE:
            providers["pip_flashattention_broadcast_mqa"] = ("Pip FlashAttention MQA (Broadcast)", ("lime", "-."))
        else:
            logger.warning("Pip FlashAttention MQA provider will be skipped as the package is not available.")
        plot_suffix = f"mqa-eval-b{BATCH}-h{N_HEADS}-d{D_HEAD}"
    else: # MHA Mode
        providers = {
            "custom_kernel_mha": ("Custom FlashAttention Kernel", ("red", "-")),
            "naive_mha_pytorch": ("Naive Pytorch MHA", ("green", "--")),
            "sdpa_pytorch_mha": ("Pytorch SDPA MHA", ("blue", "-.")),
        }
        if PIP_FLASH_ATTENTION_AVAILABLE:
             providers["pip_flashattention_mha"] = ("Pip FlashAttention MHA", ("orange", ":"))
        else:
            logger.warning("Pip FlashAttention MHA provider will be skipped as the package is not available.")
        plot_suffix = f"mha-eval-b{BATCH}-h{N_HEADS}-d{D_HEAD}"

    provider_vals = list(providers.keys())
    provider_names = [p[0] for p in providers.values()]
    provider_styles = [p[1] for p in providers.values()]

    for mode in modes:
        for causal in [True, False]:
            csv_filename_stem = f"{plot_suffix}-mode_{mode}-causal_{causal}"
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=seq_len_options,
                    line_arg="provider",
                    line_vals=provider_vals,
                    line_names=provider_names,
                    styles=provider_styles,
                    ylabel="TFLOP/s", # Primary metric for triton's internal reporting
                    plot_name=csv_filename_stem, # Used as stem for CSV filename
                    args={
                        "H_query": N_HEADS, # Number of query heads
                        "BATCH": BATCH,
                        "D_HEAD": D_HEAD,
                        "mode": mode,
                        "causal": causal,
                        "mqa_eval": mqa_eval_flag
                    },
                )
            )
    return configs


# Triton's perf_report decorator will be applied in the main loop
# core benchmark logic.
def bench_all_attentions_fn(BATCH, H_query, N_CTX, D_HEAD, mode, causal, provider, mqa_eval, dtype=torch.float16, device="cuda"):
    # Initialize accuracy metrics to NaN
    fwd_max_diff, fwd_avg_pctg = float('nan'), float('nan')
    dq_max_diff, dk_max_diff, dv_max_diff = float('nan'), float('nan'), float('nan')
    dq_avg_pctg, dk_avg_pctg, dv_avg_pctg = float('nan'), float('nan'), float('nan')
    time_s, peak_mem_gb = float('nan'), float('nan')

    requires_grad = (mode == 'bwd')
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    # --- Skip Logic ---
    if "naive" in provider and N_CTX > 2048: # Changed from 2048 to 4096
        logger.info(f"Skipping provider={provider} for N_CTX={N_CTX} > 4096 due to potential slowness/OOM.")
        # Store NaNs for all metrics for this skipped run
        run_key = (provider, N_CTX, mode, causal, mqa_eval, H_query, D_HEAD, BATCH)
        extra_metrics_store[run_key] = {
            "Time (s)": float('nan'), "Peak Memory (GB)": float('nan'), "TFLOP/s": float('nan'),
            "Fwd Max Abs Diff (vs SDPA)": float('nan'), "Fwd Avg % Err (vs SDPA)": float('nan'),
            "Bwd dQ Max Abs Diff (vs SDPA)": float('nan'), "Bwd dK Max Abs Diff (vs SDPA)": float('nan'),
            "Bwd dV Max Abs Diff (vs SDPA)": float('nan'),
            "Bwd dQ Avg % Err (vs SDPA)": float('nan'), "Bwd dK Avg % Err (vs SDPA)": float('nan'),
            "Bwd dV Avg % Err (vs SDPA)": float('nan'),
        }
        return float('nan') # Return NaN for TFLOPs, triton's primary metric

    if "pip_flashattention" in provider and not PIP_FLASH_ATTENTION_AVAILABLE:
        logger.warning(f"Skipping PIP FlashAttention provider={provider} as package is not available.")
        run_key = (provider, N_CTX, mode, causal, mqa_eval, H_query, D_HEAD, BATCH)
        extra_metrics_store[run_key] = {
            "Time (s)": float('nan'), "Peak Memory (GB)": float('nan'), "TFLOP/s": float('nan'),
            "Fwd Max Abs Diff (vs SDPA)": float('nan'), "Fwd Avg % Err (vs SDPA)": float('nan'),
            "Bwd dQ Max Abs Diff (vs SDPA)": float('nan'), "Bwd dK Max Abs Diff (vs SDPA)": float('nan'),
            "Bwd dV Max Abs Diff (vs SDPA)": float('nan'),
            "Bwd dQ Avg % Err (vs SDPA)": float('nan'), "Bwd dK Avg % Err (vs SDPA)": float('nan'),
            "Bwd dV Avg % Err (vs SDPA)": float('nan'),
        }
        return float('nan')


    torch.manual_seed(0)
    # Tensor shapes
    q_shape = (BATCH, H_query, N_CTX, D_HEAD)
    
    # K/V shapes depend on MQA and provider
    if mqa_eval:
        if provider in ["custom_mqa_kernel", "naive_mqa_pytorch", "sdpa_broadcast_mqa", "pip_flashattention_broadcast_mqa"]:
            # These MQA kernels expect K/V with 1 head
            kv_shape = (BATCH, 1, N_CTX, D_HEAD)
        else: # MHA kernels used for MQA (K/V expanded) or SDPA expecting full heads
            kv_shape = (BATCH, H_query, N_CTX, D_HEAD)
    else: # MHA mode
        kv_shape = (BATCH, H_query, N_CTX, D_HEAD)

    # Tensors for the timed benchmark
    q = torch.randn(q_shape, dtype=dtype, device=device, requires_grad=requires_grad)
    k = torch.randn(kv_shape, dtype=dtype, device=device, requires_grad=requires_grad)
    v = torch.randn(kv_shape, dtype=dtype, device=device, requires_grad=requires_grad)
    dout = torch.randn_like(q) if requires_grad else None

    # --- Prepare args for pip_flashattention_functional ---
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlen_q = N_CTX
    max_seqlen_k = N_CTX
    dropout_p = 0.0

    if "pip_flashattention" in provider:
        # For unpadded, cu_seqlens sums to total_tokens. (BATCH * N_CTX for q)
        cu_seqlens_q = torch.arange(0, (BATCH + 1) * N_CTX, step=N_CTX, dtype=torch.int32, device=device)
        if mqa_eval and provider == "pip_flashattention_broadcast_mqa":
            # K/V have dimension (BATCH, 1, N_CTX, D_HEAD) but are treated as BATCH sequences of N_CTX
            # The flash_attn_unpadded_func expects K/V to be packed qkv-style or similar if using different seq lens for k
            # For simple broadcasting with same seqlen N_CTX for K/V as Q, this should work.
            # The effective number of sequences for K/V that the function might expect in its most general form
            # is BATCH, even if head dim is 1. The total elements in K would be BATCH * N_CTX * D_HEAD.
             cu_seqlens_k = torch.arange(0, (BATCH + 1) * N_CTX, step=N_CTX, dtype=torch.int32, device=device)
        else: # MHA or MQA where K/V have same H_query as Q (already expanded)
             cu_seqlens_k = torch.arange(0, (BATCH + 1) * N_CTX, step=N_CTX, dtype=torch.int32, device=device)


    # --- Accuracy Check (Compare against SDPA) ---
    # We always compare against SDPA with K/V expanded to H_query if they aren't already
    # This provides a consistent MHA-style reference.
    q_ref_acc = q.clone().detach().requires_grad_(requires_grad)
    # K/V for reference (SDPA) always need H_query heads.
    if k.shape[1] == 1 and H_query > 1: # If original k is MQA-style
        k_ref_acc = k.clone().detach().expand(BATCH, H_query, N_CTX, D_HEAD).requires_grad_(requires_grad)
        v_ref_acc = v.clone().detach().expand(BATCH, H_query, N_CTX, D_HEAD).requires_grad_(requires_grad)
    else:
        k_ref_acc = k.clone().detach().requires_grad_(requires_grad)
        v_ref_acc = v.clone().detach().requires_grad_(requires_grad)

    out_ref_acc, _ = sdpa_pytorch(q_ref_acc, k_ref_acc, v_ref_acc, causal=causal) # sdpa_pytorch is self_attention_fast
    q_grad_ref, k_grad_ref, v_grad_ref = None, None, None
    if requires_grad:
        out_ref_acc.backward(dout.clone().detach(), retain_graph=False)
        q_grad_ref = q_ref_acc.grad.clone()
        k_grad_ref = k_ref_acc.grad.clone() # This will have H_query heads
        v_grad_ref = v_ref_acc.grad.clone() # This will have H_query heads
        q_ref_acc.grad, k_ref_acc.grad, v_ref_acc.grad = None, None, None # Clear grads for ref
    
    del q_ref_acc, k_ref_acc, v_ref_acc # Free memory for ref tensors before provider run

    # Provider tensors for accuracy
    q_prov_acc = q.clone().detach().requires_grad_(requires_grad)
    k_prov_acc = k.clone().detach().requires_grad_(requires_grad) # Use original kv_shape for provider
    v_prov_acc = v.clone().detach().requires_grad_(requires_grad)

    out_prov_acc = None
    q_grad_prov, k_grad_prov, v_grad_prov = None, None, None

    try:
        if provider == "custom_kernel_mha":
            out_prov_acc = custom_kernel_mha(q_prov_acc, k_prov_acc, v_prov_acc, sm_scale=sm_scale, causal=causal)
        elif provider == "naive_mha_pytorch":
            out_prov_acc, _ = naive_mha_pytorch(q_prov_acc, k_prov_acc, v_prov_acc, causal=causal)
        elif provider == "sdpa_pytorch_mha": # This is SDPA itself, effectively a self-check path
            out_prov_acc, _ = sdpa_pytorch(q_prov_acc, k_prov_acc, v_prov_acc, causal=causal)
        elif provider == "pip_flashattention_mha" and PIP_FLASH_ATTENTION_AVAILABLE:
            # Inputs for flash_attn_unpadded_func: q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal
            out_prov_acc = pip_flashattention_functional(
                q_prov_acc.reshape(-1, H_query, D_HEAD), k_prov_acc.reshape(-1, H_query, D_HEAD), v_prov_acc.reshape(-1, H_query, D_HEAD),
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
                softmax_scale=sm_scale, causal=causal
            ).reshape(BATCH, H_query, N_CTX, D_HEAD)
        elif provider == "custom_mqa_kernel":
            out_prov_acc = custom_mqa_kernel(q_prov_acc, k_prov_acc, v_prov_acc, sm_scale=sm_scale, causal=causal)
        elif provider == "naive_mqa_pytorch":
            out_prov_acc, _ = naive_mqa_attention(q_prov_acc, k_prov_acc, v_prov_acc, causal=causal)
        elif provider == "sdpa_expanded_for_mqa": # K/V are already H_query from shape setup
            out_prov_acc, _ = sdpa_pytorch(q_prov_acc, k_prov_acc, v_prov_acc, causal=causal)
        elif provider == "sdpa_broadcast_mqa": # K/V are (B,1,N,D)
            out_prov_acc, _ = sdpa_pytorch(q_prov_acc, k_prov_acc, v_prov_acc, causal=causal)
        elif provider == "custom_kernel_mha_expanded_for_mqa": # K/V are (B,H_query,N,D)
            out_prov_acc = custom_kernel_mha(q_prov_acc, k_prov_acc, v_prov_acc, sm_scale=sm_scale, causal=causal)
        elif provider == "naive_mha_pytorch_expanded_for_mqa": # K/V are (B,H_query,N,D)
            out_prov_acc, _ = naive_mha_pytorch(q_prov_acc, k_prov_acc, v_prov_acc, causal=causal)
        elif provider == "pip_flashattention_broadcast_mqa" and PIP_FLASH_ATTENTION_AVAILABLE: # K/V are (B,1,N,D)
            # Expand K/V to match Q's head dimension for this interface
            k_prov_acc_expanded = k_prov_acc.expand(BATCH, H_query, N_CTX, D_HEAD)
            v_prov_acc_expanded = v_prov_acc.expand(BATCH, H_query, N_CTX, D_HEAD)
            out_prov_acc = pip_flashattention_functional(
                q_prov_acc.reshape(-1, H_query, D_HEAD), 
                k_prov_acc_expanded.reshape(-1, H_query, D_HEAD), # Use expanded K
                v_prov_acc_expanded.reshape(-1, H_query, D_HEAD), # Use expanded V
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
                softmax_scale=sm_scale, causal=causal
            ).reshape(BATCH, H_query, N_CTX, D_HEAD)


        if out_prov_acc is not None:
            fwd_max_diff = max_diff(out_prov_acc, out_ref_acc)
            fwd_avg_pctg = avg_pctg_error(out_prov_acc, out_ref_acc)
            # logger.info(f"Acc Check [{provider} vs SDPA | FWD | N_CTX={N_CTX}] Max Abs Diff: {fwd_max_diff:.2e}, Avg % Err: {fwd_avg_pctg:.2f}%")

            if requires_grad:
                out_prov_acc.backward(dout.clone().detach(), retain_graph=False)
                q_grad_prov = q_prov_acc.grad.clone()
                k_grad_prov = k_prov_acc.grad.clone() # This will have provider's K head count
                v_grad_prov = v_prov_acc.grad.clone() # This will have provider's V head count

                dq_max_diff = max_diff(q_grad_prov, q_grad_ref)
                dq_avg_pctg = avg_pctg_error(q_grad_prov, q_grad_ref)

                # Special handling for K/V grads if provider is MQA (1 head) and ref is MHA (H_query heads)
                if k_grad_prov is not None and k_grad_ref is not None:
                    if k_grad_prov.shape[1] == 1 and k_grad_ref.shape[1] == H_query and H_query > 1:
                        # Compare MQA K grad to first head of ref K grad OR sum ref K grad
                        # memory_benchmark.py compared to first head, let's stick to that for consistency.
                        dk_max_diff = max_diff(k_grad_prov, k_grad_ref[:, 0:1, :, :])
                        dk_avg_pctg = avg_pctg_error(k_grad_prov, k_grad_ref[:, 0:1, :, :])
                    else: # Standard comparison (MHA vs MHA or MQA vs MQA with same head count)
                        dk_max_diff = max_diff(k_grad_prov, k_grad_ref)
                        dk_avg_pctg = avg_pctg_error(k_grad_prov, k_grad_ref)
                
                if v_grad_prov is not None and v_grad_ref is not None:
                    if v_grad_prov.shape[1] == 1 and v_grad_ref.shape[1] == H_query and H_query > 1:
                        dv_max_diff = max_diff(v_grad_prov, v_grad_ref[:, 0:1, :, :])
                        dv_avg_pctg = avg_pctg_error(v_grad_prov, v_grad_ref[:, 0:1, :, :])
                    else:
                        dv_max_diff = max_diff(v_grad_prov, v_grad_ref)
                        dv_avg_pctg = avg_pctg_error(v_grad_prov, v_grad_ref)
                # logger.info(f"Acc Check [{provider} vs SDPA | BWD | N_CTX={N_CTX}] Max Grad Diff: dQ={dq_max_diff:.2e}, dK={dk_max_diff:.2e}, dV={dv_max_diff:.2e}")
                # logger.info(f"Acc Check [{provider} vs SDPA | BWD | N_CTX={N_CTX}] Avg Grad %Err: dQ={dq_avg_pctg:.2f}%, dK={dk_avg_pctg:.2f}%, dV={dv_avg_pctg:.2f}%")
    except Exception as e_acc:
        logger.error(f"Error during accuracy check for provider={provider}, N_CTX={N_CTX}: {e_acc}")
    finally:
        del q_prov_acc, k_prov_acc, v_prov_acc, out_prov_acc
        del q_grad_prov, k_grad_prov, v_grad_prov
        del out_ref_acc # Already deleted q_ref_acc etc.
        if requires_grad: del q_grad_ref, k_grad_ref, v_grad_ref
        torch.cuda.empty_cache()

    # --- Timed Benchmark ---
    # Use original q, k, v, dout tensors. Clear gradients if any from accuracy check.
    if q.grad is not None: q.grad.zero_()
    if k.grad is not None: k.grad.zero_()
    if v.grad is not None: v.grad.zero_()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Define the forward function for timing
    fwd_fn_timed = None
    if provider == "custom_kernel_mha":
        fwd_fn_timed = lambda: custom_kernel_mha(q, k, v, sm_scale=sm_scale, causal=causal)
    elif provider == "naive_mha_pytorch":
        fwd_fn_timed = lambda: naive_mha_pytorch(q, k, v, causal=causal)[0] # Returns (out, mask)
    elif provider == "sdpa_pytorch_mha":
        fwd_fn_timed = lambda: sdpa_pytorch(q, k, v, causal=causal)[0]
    elif provider == "pip_flashattention_mha" and PIP_FLASH_ATTENTION_AVAILABLE:
        # k and v have shape (BATCH, 1, N_CTX, D_HEAD) for this provider
        # Expand K/V to match Q's head dimension for this interface
        k_timed_expanded = k.expand(BATCH, H_query, N_CTX, D_HEAD)
        v_timed_expanded = v.expand(BATCH, H_query, N_CTX, D_HEAD)
        fwd_fn_timed = lambda: pip_flashattention_functional(
            q.reshape(-1, H_query, D_HEAD), 
            k_timed_expanded.reshape(-1, H_query, D_HEAD), # Use expanded K
            v_timed_expanded.reshape(-1, H_query, D_HEAD), # Use expanded V
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
            softmax_scale=sm_scale, causal=causal
        ).reshape(BATCH, H_query, N_CTX, D_HEAD)
    elif provider == "custom_mqa_kernel":
        fwd_fn_timed = lambda: custom_mqa_kernel(q, k, v, sm_scale=sm_scale, causal=causal)
    elif provider == "naive_mqa_pytorch":
        fwd_fn_timed = lambda: naive_mqa_attention(q, k, v, causal=causal)[0]
    elif provider == "sdpa_expanded_for_mqa": # K/V are H_query here
        fwd_fn_timed = lambda: sdpa_pytorch(q, k, v, causal=causal)[0]
    elif provider == "sdpa_broadcast_mqa": # K/V are 1 head
        fwd_fn_timed = lambda: sdpa_pytorch(q, k, v, causal=causal)[0]
    elif provider == "custom_kernel_mha_expanded_for_mqa":
        fwd_fn_timed = lambda: custom_kernel_mha(q, k, v, sm_scale=sm_scale, causal=causal)
    elif provider == "naive_mha_pytorch_expanded_for_mqa":
        fwd_fn_timed = lambda: naive_mha_pytorch(q, k, v, causal=causal)[0]
    elif provider == "pip_flashattention_broadcast_mqa" and PIP_FLASH_ATTENTION_AVAILABLE:
        # k and v have shape (BATCH, 1, N_CTX, D_HEAD) for this provider
        # Expand K/V to match Q's head dimension for this interface
        k_timed_expanded = k.expand(BATCH, H_query, N_CTX, D_HEAD)
        v_timed_expanded = v.expand(BATCH, H_query, N_CTX, D_HEAD)
        fwd_fn_timed = lambda: pip_flashattention_functional(
            q.reshape(-1, H_query, D_HEAD), 
            k_timed_expanded.reshape(-1, H_query, D_HEAD), # Use expanded K
            v_timed_expanded.reshape(-1, H_query, D_HEAD), # Use expanded V
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
            softmax_scale=sm_scale, causal=causal
        ).reshape(BATCH, H_query, N_CTX, D_HEAD)
    else:
        logger.error(f"Unknown provider for timing or PIP FlashAttention not available: {provider}")
        # Store NaNs for all metrics for this failed run
        run_key = (provider, N_CTX, mode, causal, mqa_eval, H_query, D_HEAD, BATCH)
        extra_metrics_store[run_key] = {
            "Time (s)": float('nan'), "Peak Memory (GB)": float('nan'), "TFLOP/s": float('nan'),
            "Fwd Max Abs Diff (vs SDPA)": fwd_max_diff, "Fwd Avg % Err (vs SDPA)": fwd_avg_pctg, # Keep acc if available
            "Bwd dQ Max Abs Diff (vs SDPA)": dq_max_diff, "Bwd dK Max Abs Diff (vs SDPA)": dk_max_diff, "Bwd dV Max Abs Diff (vs SDPA)": dv_max_diff,
            "Bwd dQ Avg % Err (vs SDPA)": dq_avg_pctg, "Bwd dK Avg % Err (vs SDPA)": dk_avg_pctg, "Bwd dV Avg % Err (vs SDPA)": dv_avg_pctg,
        }
        return float('nan')


    # Prepare for timed benchmark run (forward and potentially backward)
    if requires_grad:
        # Run forward once to get 'o_timed' for backward pass, using the specific provider's fwd_fn_timed
        o_timed = fwd_fn_timed()
        # Define the backward function to benchmark. Retain graph for do_bench.
        bwd_fn_timed = lambda: o_timed.backward(dout, retain_graph=True)
        bench_fn_final = bwd_fn_timed
    else: # Forward only
        bench_fn_final = fwd_fn_timed
    
    # Run the timed benchmark
    # Warmup and rep counts can be adjusted.
    # Using quantiles=None because we want the mean time (ms).
    ms = triton.testing.do_bench(bench_fn_final, warmup=10, rep=20, quantiles=None)
    time_s = ms / 1000.0

    torch.cuda.synchronize()
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    # logger.info(f"Time/Mem [{provider} | {mode.upper()} | N_CTX={N_CTX}] Time: {time_s:.4f}s, Peak memory: {peak_mem_gb:.3f}GB")

    # FLOPs calculation (H_query is the number of query heads)
    # For MQA, K/V have 1 head but are broadcast H_query times for QK^T and P@V effectively.
    flops_per_matmul = 2.0 * BATCH * H_query * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul # One for QK^T, one for P@V
    if mode == "bwd": # Backward pass is roughly 2.5x the fwd FLOPs (1 for fwd, 1.5 for bwd of 2 matmuls)
        total_flops *= 2.5 # This accounts for dP, dS, dQ, dK, dV
    if causal: # Causal attention roughly halves the FLOPs
        total_flops *= 0.5
    
    tflops = total_flops * 1e-12 / time_s if time_s > 0 else float('nan')

    # Store all metrics
    run_key = (provider, N_CTX, mode, causal, mqa_eval, H_query, D_HEAD, BATCH) # Unique key
    extra_metrics_store[run_key] = {
        "Time (s)": time_s,
        "Peak Memory (GB)": peak_mem_gb,
        "TFLOP/s": tflops,
        "Fwd Max Abs Diff (vs SDPA)": fwd_max_diff,
        "Fwd Avg % Err (vs SDPA)": fwd_avg_pctg,
        "Bwd dQ Max Abs Diff (vs SDPA)": dq_max_diff,
        "Bwd dK Max Abs Diff (vs SDPA)": dk_max_diff,
        "Bwd dV Max Abs Diff (vs SDPA)": dv_max_diff,
        "Bwd dQ Avg % Err (vs SDPA)": dq_avg_pctg,
        "Bwd dK Avg % Err (vs SDPA)": dk_avg_pctg,
        "Bwd dV Avg % Err (vs SDPA)": dv_avg_pctg,
    }
    
    # Return the primary metric for triton.testing.perf_report
    return tflops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Attention Implementations: Time, TFLOPs, Memory")
    parser.add_argument("--mqa_eval", action="store_true", help="Run MQA forward pass evaluation.")
    parser.add_argument("--benchmark_mode", type=str, default="forward", choices=["forward", "full"],
                        help="Benchmark mode: 'forward' for fwd pass only, 'full' for fwd and bwd.")
    parser.add_argument("--num_heads_options", nargs='+', type=int, default=[32, 64],
                        help="List of number of query heads (N_HEADS) to iterate over.")
    parser.add_argument("--seq_len_options", nargs='+', type=int, default=[1024, 2048, 4096, 8192], # Reduced default max for faster runs
                        help="List of sequence lengths (N_CTX) to iterate over (e.g., 1024 2048 4096).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (BATCH).")
    parser.add_argument("--head_dim", type=int, default=64, help="Dimension of each attention head (D_HEAD).")
    parser.add_argument("--output_dir", type=str, default="benchmarks/benchmark_results",
                        help="Directory to save CSV results.")
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving CSV results to: {args.output_dir}")

    all_metric_keys = [
        "Time (s)", "Peak Memory (GB)", "TFLOP/s", 
        "Fwd Max Abs Diff (vs SDPA)", "Fwd Avg % Err (vs SDPA)",
        "Bwd dQ Max Abs Diff (vs SDPA)", "Bwd dK Max Abs Diff (vs SDPA)", "Bwd dV Max Abs Diff (vs SDPA)",
        "Bwd dQ Avg % Err (vs SDPA)", "Bwd dK Avg % Err (vs SDPA)", "Bwd dV Avg % Err (vs SDPA)"
    ]
    default_extra_metrics_template = {key: float('nan') for key in all_metric_keys}

    for n_heads_current in args.num_heads_options:
        logger.info(f"\nRunning benchmarks for N_HEADS (Query Heads): {n_heads_current}")
        
        current_configs_list = generate_configs(
            mqa_eval_flag=args.mqa_eval,
            current_n_heads=n_heads_current,
            batch_size_val=args.batch_size,
            head_dim_val=args.head_dim,
            seq_len_options=args.seq_len_options,
            benchmark_mode_val=args.benchmark_mode
        )
        
        if not current_configs_list:
            logger.warning(f"No configurations generated for N_HEADS={n_heads_current}. Skipping.")
            continue

        DecoratedBenchFn = triton.testing.perf_report(current_configs_list)(bench_all_attentions_fn)
        
        try:
            logger.info(f"Starting benchmark execution for N_HEADS={n_heads_current} (data will be stored in memory)...")
            # Run benchmarks but prevent Triton from saving any files automatically
            DecoratedBenchFn.run(save_path=None, print_data=True, show_plots=False)
            logger.info(f"Benchmark execution completed for N_HEADS={n_heads_current}.")
        except Exception as e_run:
            logger.error(f"Exception during DecoratedBenchFn.run for N_HEADS={n_heads_current}: {e_run}")
            import traceback
            traceback.print_exc()
            continue

        logger.info(f"Manually saving CSV files from collected metrics for N_HEADS={n_heads_current}...")
        for config_obj in current_configs_list: # config_obj is a triton.testing.Benchmark object
            data_for_this_csv = []
            # Map provider keys (line_vals) to provider names (line_names) for the 'Provider Name' column
            provider_name_map = dict(zip(config_obj.line_vals, config_obj.line_names))

            args_for_key = config_obj.args # Fixed arguments for this config (mode, causal, H_query, etc.)

            for n_ctx_val in config_obj.x_vals: # Iterate N_CTX
                for provider_key in config_obj.line_vals: # Iterate provider keys
                    run_key = (
                        provider_key,
                        n_ctx_val,
                        args_for_key['mode'],
                        args_for_key['causal'],
                        args_for_key['mqa_eval'],
                        args_for_key['H_query'],
                        args_for_key['D_HEAD'],
                        args_for_key['BATCH']
                    )
                    
                    metrics_dict = extra_metrics_store.get(run_key, default_extra_metrics_template.copy())
                    
                    row_data = {
                        config_obj.x_names[0]: n_ctx_val, # e.g., N_CTX
                        'Provider Name': provider_name_map.get(provider_key, provider_key),
                        'provider': provider_key,
                        'N_HEADS_query': args_for_key['H_query'],
                        'D_HEAD': args_for_key['D_HEAD'],
                        'BATCH': args_for_key['BATCH'],
                        'mode': args_for_key['mode'],
                        'causal': args_for_key['causal'],
                        'mqa_eval': args_for_key['mqa_eval'],
                    }
                    row_data.update(metrics_dict)
                    data_for_this_csv.append(row_data)
            
            if data_for_this_csv:
                final_df = pd.DataFrame(data_for_this_csv)
                
                # Reorder columns to be consistent
                ordered_cols = [config_obj.x_names[0], 'Provider Name', 'provider',
                                'N_HEADS_query', 'D_HEAD', 'BATCH', 'mode', 'causal', 'mqa_eval'] + \
                               all_metric_keys
                ordered_cols = [col for col in ordered_cols if col in final_df.columns] # Ensure all cols exist
                final_df = final_df[ordered_cols]

                csv_filename = config_obj.plot_name + ".csv"
                final_csv_path = os.path.join(args.output_dir, csv_filename)
                try:
                    final_df.to_csv(final_csv_path, index=False)
                    logger.info(f"Successfully saved: {final_csv_path}")
                except IOError as e_save:
                    logger.error(f"Error: Failed to save final CSV {final_csv_path}. Reason: {e_save}")
                except Exception as e_save_other:
                    logger.error(f"Error: An unexpected error occurred while saving {final_csv_path}. Reason: {e_save_other}")
            else:
                logger.warning(f"No data collected for CSV: {config_obj.plot_name}.csv. Skipping save.")

    logger.info("\nAll benchmark runs and CSV processing complete.")
    logger.info(f"Final CSV files are located in: {args.output_dir}")
