import triton.testing
import math  
import os 
import logging

import triton
import triton.language as tl
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# configs for A100 and H100 
# (dtype, head_dim): (BLOCK_Q, BLOCK_K, num_warps, num_stages)
_h100_config = {
    (torch.float32, 64): (128, 32, 4, 3),
    (torch.float32, 128): (32, 64, 4, 3),
    (torch.float32, 256): (32, 32, 4, 3),
    (torch.bfloat16, 64): (128, 128, 4, 3),
    (torch.bfloat16, 128): (128, 64, 8, 3),
    (torch.bfloat16, 256): (64, 32, 4, 3),
    (torch.float16, 64): (128, 128, 4, 3),
    (torch.float16, 128): (128, 128, 8, 3),
    (torch.float16, 256): (64, 32, 4, 3),
}

_a100_config = {
    (torch.float32, 64): (128, 32, 4, 3),
    (torch.float32, 128): (128, 32, 4, 3),
    (torch.float32, 256): (64, 16, 4, 3),
    (torch.bfloat16, 64): (128, 64, 4, 3),
    (torch.bfloat16, 128): (128, 64, 8, 3),
    (torch.bfloat16, 256): (32, 64, 4, 3),
    (torch.float16, 64): (128, 64, 4, 3),
    (torch.float16, 128): (128, 64, 8, 3),
    (torch.float16, 256): (32, 64, 4, 3),
}


def get_config(dtype, head_dim):
    # Normalize dtype if it's a string or numpy dtype
    if isinstance(dtype, str):
        if dtype in ("fp16", "float16", "torch.float16"):
            dtype = torch.float16
        elif dtype in ("bf16", "bfloat16", "torch.bfloat16"):
            dtype = torch.bfloat16
        elif dtype in ("fp32", "float32", "torch.float32"):
            dtype = torch.float32
        else:
            raise ValueError(f"Unknown dtype string: {dtype}")
    elif hasattr(dtype, 'name'):  # e.g. numpy dtype
        if dtype.name in ("float16", "fp16"):
            dtype = torch.float16
        elif dtype.name in ("bfloat16", "bf16"):
            dtype = torch.bfloat16
        elif dtype.name in ("float32", "fp32"):
            dtype = torch.float32
        else:
            raise ValueError(f"Unknown dtype: {dtype}")
    # Detect device
    compute_capability = torch.cuda.get_device_capability()
    is_h100 = compute_capability[0] >= 9  # H100 is compute capability 9.0+

    # Choose config dictionary
    config_table = _h100_config if is_h100 else _a100_config

    # Get config or raise error
    key = (dtype, head_dim)
    if key not in config_table:
        raise ValueError(f"No FlashAttention config found for dtype={dtype}, head_dim={head_dim} on {'H100' if is_h100 else 'A100'}.")

    return config_table[key]

def strides(t):
    assert t is not None
    return [t.stride(i) for i in range(t.ndim)]

def fwd_config_pruner(configs, nargs, HEAD_DIM, DTYPE, **kwargs):
    min_size, max_size = 32, 256
    min_pipeline, max_pipeline = 1, 3
    min_warps, max_warps = 1, 8

    if HEAD_DIM == 64:
        min_pipeline = 2
    elif HEAD_DIM == 128:
        max_size = 128
        min_size = 32
        max_pipeline = 3
        max_warps = 4
    elif HEAD_DIM == 256:
        max_size = 128
        min_size = 32
        max_pipeline = 2
        max_warps = 4
    
    # Filter configs based on criteria
    filtered_configs = [
        config for config in configs
        if (min_size <= config.kwargs["TILE_Q_SIZE"] <= max_size and
            min_size <= config.kwargs["TILE_K_SIZE"] <= max_size and
            min_pipeline <= config.num_stages <= max_pipeline and
            min_warps <= config.num_warps <= max_warps)
    ]

    # Add default configs
    default_config = get_config(DTYPE, HEAD_DIM)
    if default_config is not None:
        default_configs = [
            triton.Config(
                dict(
                    PIPELINING=default_config[3],
                    TILE_Q_SIZE=default_config[0],
                    TILE_K_SIZE=default_config[1],
                    V_PRELOAD=V_PRELOAD,
                ),
                num_warps=default_config[2],
                num_stages=default_config[3],
            )
            for V_PRELOAD in [True, False]
        ]
        filtered_configs.extend(default_configs)
        logger.warning(f"Added default configs for HEAD_DIM={HEAD_DIM}, DTYPE={DTYPE}")
    
    logger.warning(f"Start benchmarking forward streaming_attention {len(filtered_configs) = }")
    return filtered_configs

def torch_dtype_to_triton(dtype):
    if dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.float32:
        return tl.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

# --------------- FORWARD KERNEL (Identical to complete_flashattention.py) ---------------

@triton.jit
def self_attention_forward(
    Q_ptr: tl.tensor, K_ptr: tl.tensor, V_ptr: tl.tensor, L_ptr: tl.tensor, Out_ptr: tl.tensor, # pointers to input and output tensors
    Max_ptr: tl.tensor, SumExp_ptr: tl.tensor, # pointers for saving m_i and l_i
    stride_qb: int, stride_qh: int, stride_qn: int, stride_qd: int, # strides for Q
    stride_kb: int, stride_kh: int, stride_kn: int, stride_kd: int, # strides for K 
    stride_vb: int, stride_vh: int, stride_vn: int, stride_vd: int, # strides for V
    stride_ob: int, stride_oh: int, stride_on: int, stride_od: int, # strides for output
    stride_maxb: int, stride_maxh: int, stride_maxn: int, # strides for Max
    stride_sumexpb: int, stride_sumexph: int, stride_sumexpn: int, # strides for SumExp
    lens_stride: int, # stride for lens, 
    SEQ_LEN: int, #
    HEAD_DIM: tl.constexpr, # 
    LEN_PRESENT: tl.constexpr, # if true, mask is applied
    PRESCALE: tl.constexpr, # if true, scale Q by 1/sqrt(head_dim) before matmul with K^T
    INPUT_PRECISION: tl.constexpr, # precision of matrix multiplications
    DTYPE: tl.constexpr, # output datatype
    SOFTMAX_SCALE: tl.constexpr, # 1/sqrt(head_dim) for softmax
    TILE_Q_SIZE: tl.constexpr, # size of tile for Q
    TILE_K_SIZE: tl.constexpr, # size of tile for K
    V_PRELOAD: tl.constexpr, # if true, preload V into memory
    PIPELINING: tl.constexpr, # number of tiles to process in parallel
    TIME_BUCKET: tl.constexpr, # number of tokens to process at once 
    RCP_LN2: tl.constexpr, # 1/ln(2), trick for scaling softmax
    CAUSAL: tl.constexpr, # if true, apply causal mask
):
    # get tile we'll calculate for
    batch = tl.program_id(0)
    head_num = tl.program_id(1) # This corresponds to the query head index
    q_tile_idx = tl.program_id(2)
    q_token_idx = q_tile_idx * TILE_Q_SIZE

    # get mask if needed
    if LEN_PRESENT:
        seq_len = tl.load(L_ptr + batch * lens_stride)
        seq_len = tl.minimum(seq_len, SEQ_LEN)
        q_need_mask = q_token_idx + TILE_Q_SIZE >= seq_len
    else: 
        seq_len = SEQ_LEN
        q_need_mask = 0

    # return if out of bounds
    if q_token_idx >= seq_len: 
        return
    
    q_start = q_token_idx

    # load Q for current tile, using the query head index
    q_offset = batch * stride_qb + head_num * stride_qh
    q_tile_ptr = tl.make_block_ptr(
        base= Q_ptr + q_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(q_start, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1, 0), # row major layout 
    )

    # load K transpose for current tile
    # In MQA, K has only one head dimension, so stride_kh should be 0.
    # The kernel logic remains the same, it just uses the provided stride.
    # The head_num from program_id(1) is *not* used to index K's head dimension.
    k_offset = batch * stride_kb # + head_num * stride_kh <- stride_kh is 0 here
    k_tile_ptr = tl.make_block_ptr(
        base= K_ptr + k_offset,
        shape=(HEAD_DIM, SEQ_LEN), # K shape is (B, 1, N, D), so K_ptr is (B*N*D)
        strides=(stride_kd, stride_kn), # Stride for dim D, stride for dim N
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_K_SIZE),
        order=(0, 1), 
    )

    # load V for current tile
    # Similar to K, V has only one head dimension, stride_vh should be 0.
    v_offset = batch * stride_vb # + head_num * stride_vh <- stride_vh is 0 here
    v_tile_ptr = tl.make_block_ptr(
        base= V_ptr + v_offset,
        shape=(SEQ_LEN, HEAD_DIM), # V shape is (B, 1, N, D)
        strides=(stride_vn, stride_vd), # Stride for dim N, stride for dim D
        offsets=(0, 0),
        block_shape=(TILE_K_SIZE, HEAD_DIM),
        order=(1, 0), 
    )

    m_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32) - 1e6 # running max of qkt
    l_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32) # running sum of exp(qkt)
    o_i = tl.zeros([TILE_Q_SIZE, HEAD_DIM], dtype=DTYPE) # running sum of v * exp(qkt)

    q_tile_indices = tl.arange(0, TILE_Q_SIZE) + q_token_idx

    q_curr_tile = tl.load(q_tile_ptr, boundary_check=(0,))
    softmax_scale: tl.constexpr = tl.cast(SOFTMAX_SCALE * RCP_LN2, q_curr_tile.dtype)


    if PRESCALE:
        q_curr_tile = q_curr_tile * softmax_scale

    max_tile = tl.cdiv(seq_len, TILE_K_SIZE)
    if CAUSAL:
        max_tile = tl.minimum(max_tile, tl.cdiv(q_token_idx + TILE_Q_SIZE, TILE_K_SIZE))

    for tile_idx in tl.range(0, max_tile, num_stages=PIPELINING):
        last_iteration = tile_idx == max_tile - 1
        kv_token_idx = tile_idx * TILE_K_SIZE

        if last_iteration:
            k_curr_tile = tl.load(tl.advance(k_tile_ptr, (0, kv_token_idx)), boundary_check=(1,))
        else: 
            k_curr_tile = tl.load(tl.advance(k_tile_ptr, (0, kv_token_idx)))

        if V_PRELOAD:
            if last_iteration:
                v_curr_tile = tl.load(tl.advance(v_tile_ptr, (kv_token_idx, 0)), boundary_check=(0,))
            else: 
                v_curr_tile = tl.load(tl.advance(v_tile_ptr, (kv_token_idx, 0)))

        qk = tl.dot(q_curr_tile, k_curr_tile, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

        if not PRESCALE:
            qk = qk * softmax_scale

        # Causal mask: prevent attention to future tokens
        if CAUSAL:
            q_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)
            k_indices = kv_token_idx + tl.arange(0, TILE_K_SIZE)
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            qk = tl.where(causal_mask, qk, -1e6)

        if last_iteration:
            kv_indices = kv_token_idx + tl.arange(0, TILE_K_SIZE)
            mask = (kv_indices < seq_len)[None, :]  # shape [1, TILE_K_SIZE]
            qk = tl.where(mask, qk, -1e6)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)

        l_i = l_i * alpha + l_ij
        o_i = o_i * alpha[:, None]

        if not V_PRELOAD:
            if last_iteration:
                v_curr_tile = tl.load(tl.advance(v_tile_ptr, (kv_token_idx, 0)), boundary_check=(0,))
            else: 
                v_curr_tile = tl.load(tl.advance(v_tile_ptr, (kv_token_idx, 0)))   
            
        p_cast = p.to(DTYPE)
        v_cast = v_curr_tile.to(DTYPE)
        o_cast = o_i.to(DTYPE)
        o_i = tl.dot(p_cast, v_cast, acc=o_cast, input_precision=INPUT_PRECISION, out_dtype=DTYPE)

        m_i = m_ij

    o_i = o_i / l_i[:, None]
    
    # Use query head index for storing Max and SumExp
    max_offset = batch * stride_maxb + head_num * stride_maxh + q_start * stride_maxn
    sumexp_offset = batch * stride_sumexpb + head_num * stride_sumexph + q_start * stride_sumexpn

    max_ptrs = Max_ptr + max_offset + tl.arange(0, TILE_Q_SIZE) * stride_maxn
    sumexp_ptrs = SumExp_ptr + sumexp_offset + tl.arange(0, TILE_Q_SIZE) * stride_sumexpn

    if q_need_mask:
        # Mask for valid query indices within the tile based on actual sequence length
        q_lens_mask = (q_tile_indices < seq_len)
        # Apply output mask before storing O
        o_i = tl.where(q_lens_mask[:, None], o_i, 0.0) 
        # Store m_i and l_i with masking 
        tl.store(max_ptrs, m_i, mask=q_lens_mask)
        tl.store(sumexp_ptrs, l_i, mask=q_lens_mask)
    else:
        # Store m_i and l_i without masking 
        tl.store(max_ptrs, m_i)
        tl.store(sumexp_ptrs, l_i)
    # --- Store Output --- 
    # Use query head index for storing Output
    o_offset = batch * stride_ob + head_num * stride_oh
    o_tile_ptr = tl.make_block_ptr(
        base= Out_ptr + o_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(q_start, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    tl.store(o_tile_ptr, o_i.to(DTYPE), boundary_check=(0,))


def autotune_prehook(kwargs, exception=None, **_):
    if "lens" in kwargs and kwargs["lens"] is not None: 
        kwargs["lens"].add_(kwargs["q"].shape[2]) # lens += time

def autotune_posthook(kwargs, exception=None, **_):
    if "lens" in kwargs and kwargs["lens"] is not None: 
        kwargs["lens"].add_(-kwargs["q"].shape[2]) # lens -= time

streaming_forward = triton.heuristics(
    dict(
        PIPELINING=lambda _: 1,
        TILE_Q_SIZE=lambda _: 64, 
        TILE_K_SIZE=lambda _: 64,
    )
)(self_attention_forward)

streaming_forward_autotuned = triton.autotune(
    configs = [
        triton.Config(
            dict(
                PIPELINING=pipe, 
                TILE_Q_SIZE=tile_q, 
                TILE_K_SIZE=tile_k,
                V_PRELOAD=V_PRELOAD,
            ),
            num_warps=num_warps,
            num_stages=pipe,
        )
        for num_warps in [4,8]
        for pipe in [1,2]
        for tile_q in [
            2**i
            for i in range(
                int(math.log2(32) + 0.1), 
                int(math.log2(256) + 0.1) + 1
            )
        ]
        for tile_k in [
            2**i
            for i in range(
                int(math.log2(32) + 0.1), 
                int(math.log2(256) + 0.1) + 1,
            )
        ]
        for V_PRELOAD in [True, False]
    ],
    key = ["HEAD_DIM", "INPUT_PRECISION", "TIME_BUCKET", "DTYPE"], 
    prune_configs_by = dict(early_config_prune=fwd_config_pruner),
    pre_hook = autotune_prehook,
    post_hook = autotune_posthook,
)(self_attention_forward)


def mqa_flash_attention(q, k, v, lens=None, sm_scale=None, autotune=True, prescale=False, causal=False):
    """
    Forward pass for Multi-Query Attention (MQA) using Triton kernel.
    K and V are assumed to have a single head dimension shared across query heads.

    Args:
        q: Query tensor of shape (batch_size, head_num, seq_len, head_dim)
        k: Key tensor of shape (batch_size, 1, seq_len, head_dim)
        v: Value tensor of shape (batch_size, 1, seq_len, head_dim)
        lens: Optional lens tensor of shape (batch_size)
        sm_scale: Optional scale factor (1/sqrt(head_dim)). Defaults if None.
        autotune: Whether to use autotuning.
        prescale: Whether to scale Q before QK^T matmul.
        causal: Whether to apply causal masking.
    Returns:
        Output tensor of shape (batch_size, head_num, seq_len, head_dim)
    """
    # Input validation
    batch, q_heads, seq_len, HEAD_DIM = q.shape
    assert HEAD_DIM in {16, 32, 64, 128, 256}
    assert k.shape == (batch, 1, seq_len, HEAD_DIM), f"K shape mismatch: expected {(batch, 1, seq_len, HEAD_DIM)}, got {k.shape}"
    assert v.shape == (batch, 1, seq_len, HEAD_DIM), f"V shape mismatch: expected {(batch, 1, seq_len, HEAD_DIM)}, got {v.shape}"
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    assert lens is None or (
        lens.dtype == torch.int32 and batch == len(lens) and lens.ndim == 1
    )

    O = torch.zeros_like(q, memory_format=torch.contiguous_format)

    # Note: Max and SumExp dimensions match the Output/Query dimensions
    Max = torch.empty((batch, q_heads, seq_len), dtype=torch.float32, device=q.device)
    SumExp = torch.empty((batch, q_heads, seq_len), dtype=torch.float32, device=q.device)

    INPUT_PRECISION = (
        "tf32" if torch.get_float32_matmul_precision() != "highest" else "ieee"
    )

    # Calculate strides
    q_strides = strides(q)
    k_strides = strides(k)
    v_strides = strides(v)
    o_strides = strides(O)
    max_strides = strides(Max)
    sumexp_strides = strides(SumExp)
    lens_stride = strides(lens)[0] if lens is not None else 0

    # --- MQA Specific: Set K and V head strides to 0 ---
    # Tensor shape is (B, H, N, D), strides are typically (H*N*D, N*D, D, 1)
    # We want to set the stride for the H dimension (index 1) to 0 for K and V
    k_strides[1] = 0 
    v_strides[1] = 0

    # Grid calculation based on Query heads
    grid = lambda args: (
        batch, 
        q_heads, # Use query heads for grid dimension 1
        triton.cdiv(seq_len, args["TILE_Q_SIZE"])
    )

    fwd_function = streaming_forward_autotuned if autotune else streaming_forward

    extra_kwargs = dict(RCP_LN2=1.0 / math.log(2.0))
    if not autotune:
        extra_kwargs['PIPELINING'] = 1

    # Note: Pass the modified k_strides and v_strides to the kernel
    fwd_function[grid](
        q, 
        k,
        v, 
        lens, 
        O, 
        Max,       # Pass Max tensor
        SumExp,    # Pass SumExp tensor
        *q_strides, 
        *k_strides, # Pass modified K strides
        *v_strides, # Pass modified V strides
        *o_strides, 
        *max_strides,     # Pass Max strides (matches Q/O)
        *sumexp_strides,  # Pass SumExp strides (matches Q/O)
        lens_stride, 
        SEQ_LEN=seq_len, 
        HEAD_DIM=HEAD_DIM, 
        PRESCALE=prescale, 
        INPUT_PRECISION=INPUT_PRECISION, 
        DTYPE=torch_dtype_to_triton(q.dtype), 
        SOFTMAX_SCALE=sm_scale, 
        LEN_PRESENT=lens is not None, 
        TIME_BUCKET=triton.next_power_of_2(seq_len), 
        CAUSAL=causal,
        **extra_kwargs
    )

    # No backward pass needed, so we don't save context or deal with Max conversion
    
    return O

def naive_mqa_attention(q, k, v, lens=None, causal=False):
    """
    Naive Multi-Query Attention forward pass using PyTorch matmuls.

    Args:
        q: Query tensor of shape (batch_size, q_head_num, seq_len, head_dim)
        k: Key tensor of shape (batch_size, 1, seq_len, head_dim)
        v: Value tensor of shape (batch_size, 1, seq_len, head_dim)
        lens: Optional lens tensor of shape (batch_size)
        causal: If True, apply causal mask.
    Returns:
        Output tensor of shape (batch_size, q_head_num, seq_len, head_dim),
        Combined attention mask tensor of shape (batch_size, q_head_num, seq_len, seq_len)
    """
    batch_size, q_head_num, seq_len, head_dim = q.shape
    kv_head_num = k.shape[1]
    assert kv_head_num == 1, "K and V must have exactly one head for MQA."
    assert k.shape == (batch_size, 1, seq_len, head_dim)
    assert v.shape == (batch_size, 1, seq_len, head_dim)

    # Scale Q
    q_scaled = q / math.sqrt(head_dim)

    # Q K^T
    # q_scaled: (B, Hq, N, D)
    # k.transpose(-2, -1): (B, 1, D, N)
    # Result: (B, Hq, N, N) due to broadcasting on Hq dim for k
    qkt = q_scaled @ k.transpose(-2, -1)

    # --- Construct the combined attention mask ---
    # Start with all True (allow all attention)
    # Mask shape should be (B, Hq, N, N)
    final_attn_mask = torch.ones(batch_size, q_head_num, seq_len, seq_len, device=q.device, dtype=torch.bool)

    # Optional Causal mask
    if causal:
        causal_mask_base = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
        # Expand for q_head_num: (1, 1, N, N) -> (1, Hq, N, N) if Hq > 1, or just use as is
        causal_mask = causal_mask_base.unsqueeze(0).unsqueeze(0) # Shape (1,1,N,N)
        # It will broadcast correctly with final_attn_mask (B, Hq, N, N)
        final_attn_mask &= causal_mask

    # Length masking (if lens provided)
    lens_padding_mask_1d = None # For output masking later
    if lens is not None:
        positions = torch.arange(seq_len, device=q.device).view(1, 1, seq_len)
        # Expand lens for q_head_num: (B, 1, 1) -> (B, Hq, 1) or let it broadcast
        lens_exp = lens.view(batch_size, 1, 1).expand(-1, q_head_num, -1) # Shape (B, Hq, N)
        
        # Create 1D mask (B, Hq, N, 1) for rows
        lens_padding_mask_1d_q = (positions < lens_exp.unsqueeze(-1)) # (B, Hq, N, 1)
        # Create 1D mask (B, 1, N, 1) for columns (keys/values are shared)
        # We need to compare positions with lens_exp for the single K/V head and then broadcast
        lens_exp_kv = lens.view(batch_size, 1, 1).expand(-1, 1, -1) # (B, 1, N)
        lens_padding_mask_1d_kv = (positions < lens_exp_kv.unsqueeze(-1)) # (B, 1, N, 1)

        # Create the [B, Hq, S, S] lens mask
        # lens_padding_mask_1d_q: (B, Hq, N, 1)
        # lens_padding_mask_1d_kv.transpose(-2,-1): (B, 1, 1, N)
        # Broadcasting will result in (B, Hq, N, N)
        attn_lens_mask = lens_padding_mask_1d_q & lens_padding_mask_1d_kv.transpose(-2,-1)
        final_attn_mask &= attn_lens_mask
        
        # For output masking, we only care about the query sequence length validity
        lens_padding_mask_1d = lens_padding_mask_1d_q

    # --- Apply the combined mask to attention scores ---
    mask_value = torch.finfo(qkt.dtype).min 
    qkt = qkt.masked_fill(~final_attn_mask, mask_value)

    # Attention weights
    attention_weights = F.softmax(qkt, dim=-1) # Shape (B, Hq, N, N)

    # Attention output
    # attention_weights: (B, Hq, N, N)
    # v: (B, 1, N, D)
    # Result: (B, Hq, N, D) due to broadcasting on Hq dim for v
    output = attention_weights @ v 
    
    # --- Apply output masking based *only* on lens for query tokens ---
    if lens_padding_mask_1d is not None:
        output = torch.where(lens_padding_mask_1d, output, 0) 

    return output, final_attn_mask


