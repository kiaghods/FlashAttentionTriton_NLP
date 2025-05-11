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


# --------------- FORWARD KERNEL ---------------

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
    head_num = tl.program_id(1)
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

    # load Q, K, V for current tile
    q_offset = batch * stride_qb + head_num * stride_qh
    q_tile_ptr = tl.make_block_ptr(
        base= Q_ptr + q_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(q_start, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1, 0), # row major layout 
    )

    # we load K transpose for current tile
    k_offset = batch * stride_kb + head_num * stride_kh
    k_tile_ptr = tl.make_block_ptr(
        base= K_ptr + k_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_K_SIZE),
        order=(0, 1), 
    )

    v_offset = batch * stride_vb + head_num * stride_vh
    v_tile_ptr = tl.make_block_ptr(
        base= V_ptr + v_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_vn, stride_vd),
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

def torch_dtype_to_triton(dtype):
    if dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.float32:
        return tl.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def self_attention_slow(q, k, v, lens=None, causal=False):
    """
    Self-attention using manual matrix multiplication implementation.
    
    Args:
        q: Query tensor of shape (batch_size, head_num, seq_len, head_dim)
        k: Key tensor of shape (batch_size, head_num, seq_len, head_dim)
        v: Value tensor of shape (batch_size, head_num, seq_len, head_dim)
        lens: Optional lens tensor of shape (batch_size)
        causal: If True, apply causal mask.
    Returns:
        Output tensor of shape (batch_size, head_num, seq_len, head_dim),
        Combined attention mask tensor of shape (batch_size, head_num, seq_len, seq_len)
    """
    batch_size, head_num, seq_len, head_dim = q.shape

    # attention scores
    qkt = (q / math.sqrt(head_dim)) @ k.transpose(-2, -1)

    # --- Construct the combined attention mask ---
    # Start with all True (allow all attention)
    final_attn_mask = torch.ones(batch_size, head_num, seq_len, seq_len, device=q.device, dtype=torch.bool)

    # Optional Causal mask
    if causal:
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
        # Apply to final mask (AND operation)
        final_attn_mask &= causal_mask.unsqueeze(0).unsqueeze(0) # Expand and apply

    # Length masking (if lens provided)
    lens_padding_mask_1d = None # For output masking later
    if lens is not None:
        positions = torch.arange(seq_len, device=q.device).view(1, 1, seq_len).repeat(batch_size, head_num, 1)
        lens_exp = lens.view(batch_size, 1, 1).repeat(1, head_num, seq_len)
        lens_padding_mask_1d = (positions < lens_exp).unsqueeze(-1) # Shape [B, H, S, 1]
        # Create the [B, H, S, S] lens mask
        attn_lens_mask = lens_padding_mask_1d & lens_padding_mask_1d.transpose(-2, -1)
        # Apply to final mask (AND operation)
        final_attn_mask &= attn_lens_mask

    # --- Apply the combined mask to attention scores ---
    # Use the smallest representable value for the dtype instead of -1e6 or -inf
    mask_value = torch.finfo(qkt.dtype).min 
    qkt = qkt.masked_fill(~final_attn_mask, mask_value)

    # attention weights
    attention_weights = F.softmax(qkt, dim=-1)

    # attention output
    output = attention_weights @ v
    
    # --- Apply output masking based *only* on lens ---
    if lens_padding_mask_1d is not None:
        output = torch.where(lens_padding_mask_1d, output, 0) # Use the [B, H, S, 1] mask here

    return output, final_attn_mask
    


def self_attention_fast(q, k, v, lens=None, causal=False):
    """
    Self-attention using Pytorch scaled-dot product attention.

    Args:
        q: Query tensor of shape (batch_size, head_num, seq_len, head_dim)
        k: Key tensor of shape (batch_size, head_num, seq_len, head_dim)
        v: Value tensor of shape (batch_size, head_num, seq_len, head_dim)
        lens: Optional lens tensor of shape (batch_size)
        causal: If True, apply causal mask.
    Returns:
        Output tensor of shape (batch_size, head_num, seq_len, head_dim),
        Mask identifying padded output rows (shape [B, H, S, 1]) or 1 if no padding.
    """
    batch_size, head_num, seq_len, head_dim = q.shape

    # --- Call SDPA ---
    # Let SDPA handle internal masking based on the causal flag
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=causal)

    # --- Apply output masking based *only* on lens ---
    output_mask = 1 # Default if no lens
    if lens is not None:
        positions = torch.arange(seq_len, device=q.device).view(1, 1, seq_len).repeat(batch_size, head_num, 1)
        lens_exp = lens.view(batch_size, 1, 1).repeat(1, head_num, seq_len)
        output_mask = (positions < lens_exp).unsqueeze(-1) # Shape [B, H, S, 1]
        output = torch.where(output_mask, output, 0) # Use the [B, H, S, 1] mask here

    return output, output_mask # Return output and the [B, H, S, 1] mask or 1

# --------------- BACKWARD KERNEL ---------------
@triton.jit
def self_attention_backward_preprocess(
    out,    # [B, H, N, D] tensor -- output of attention
    dO,     # [B, H, N, D] tensor -- gradient of loss wrt output
    SumExp, # [B, H, N] tensor -- sumexp 
    new_dO, # [B, H, N, D] tensor -- normalized dO / SumExp
    delta,  # [B, H, N] tensor -- delta = sum_k o_k * dO_k (where dO is normalized)
    # Strides
    stride_ob, stride_oh, stride_on, stride_od, # Strides for out
    stride_dob, stride_doh, stride_don, stride_dod, # Strides for dO
    stride_lb, stride_lh, stride_ln,            # Strides for SumExp (l)
    stride_ndob, stride_ndoh, stride_ndon, stride_ndod, # Strides for new_dO
    stride_deltab, stride_deltah, stride_deltan, # Strides for delta
    # Meta parameters
    HEAD_NUM: tl.constexpr,
    TILE_Q_SIZE: tl.constexpr, # tile size along seq dim (N)
    HEAD_DIM: tl.constexpr,    # head dim (D)
):
    block_idx = tl.program_id(0) # Index along sequence length blocks
    bh_idx = tl.program_id(1)    # Combined batch/head index

    batch_idx = bh_idx // HEAD_NUM
    head_idx = bh_idx % HEAD_NUM

    q_start = block_idx * TILE_Q_SIZE
    offs_m = q_start + tl.arange(0, TILE_Q_SIZE) # Indices along seq dim [TILE_Q_SIZE]
    offs_n = tl.arange(0, HEAD_DIM)              # Indices along head dim [HEAD_DIM]

    # --- Pointers for the current batch, head, and block --- 
    # Pointers for 2D slices [TILE_Q_SIZE, HEAD_DIM]
    o_ptr = out + batch_idx * stride_ob + head_idx * stride_oh + \
            (offs_m[:, None] * stride_on + offs_n[None, :] * stride_od)
    do_ptr = dO + batch_idx * stride_dob + head_idx * stride_doh + \
             (offs_m[:, None] * stride_don + offs_n[None, :] * stride_dod)
    new_do_ptr = new_dO + batch_idx * stride_ndob + head_idx * stride_ndoh + \
                 (offs_m[:, None] * stride_ndon + offs_n[None, :] * stride_ndod)

    # Pointers for 1D slices [TILE_Q_SIZE]
    sumexp_ptr = SumExp + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_ln
    delta_ptr = delta + batch_idx * stride_deltab + head_idx * stride_deltah + offs_m * stride_deltan

    # --- Load --- (Assume tensors are large enough, add boundary checks if needed)
    o = tl.load(o_ptr).to(tl.float32)
    do = tl.load(do_ptr).to(tl.float32)
    denom = tl.load(sumexp_ptr).to(tl.float32) # Denominator (SumExp)

    # --- Compute --- 
    # Normalize dO: dO_scaled = dO / SumExp
    do_scaled = do / denom[:, None] # Shape [TILE_Q_SIZE, HEAD_DIM]
    # Compute delta: delta = rowsum(O * dO_scaled)
    delta_computed = tl.sum(o * do_scaled, axis=1) # Shape [TILE_Q_SIZE]

    # --- Write-back ---
    tl.store(new_do_ptr, do_scaled)
    tl.store(delta_ptr, delta_computed) # Store computed delta


@triton.jit
def self_attention_backward(
    # Inputs
    Q, K, V,        # [B, H, N, D] tensors
    SOFTMAX_SCALE,  # float
    out,            # [B, H, N, D] tensor -- output of attention
    dO,             # [B, H, N, D] tensor -- gradient of loss wrt output (possibly scaled)
    # Outputs
    dQ, dK, dV,     # [B, H, N, D] tensor -- gradient of loss wrt query, key, value
    # Intermediates
    L, M,           # [B, H, N] tensor -- sumexp and max from forward pass
    D,              # [B, H, N] tensor -- delta = rowsum(O * dO_scaled)
    # Strides
    stride_qb, stride_qh, stride_qm, stride_qd, # strides for Q
    stride_kb, stride_kh, stride_kn, stride_kd, # strides for K
    stride_vb, stride_vh, stride_vn, stride_vd, # strides for V
    stride_ob, stride_oh, stride_om, stride_od, # strides for out (O)
    stride_dob, stride_doh, stride_dom, stride_dod, # strides for dO
    stride_dqb, stride_dqh, stride_dqm, stride_dqd, # strides for dQ
    stride_dkb, stride_dkh, stride_dkn, stride_dkd, # strides for dK
    stride_dvb, stride_dvh, stride_dvn, stride_dvd, # strides for dV
    stride_lb, stride_lh, stride_ln,            # strides for L (SumExp)
    stride_mb, stride_mh, stride_mn,            # strides for M (Max)
    stride_deltab, stride_deltah, stride_deltan, # strides for D (Delta)
    # Meta parameters
    BATCH_NUM, HEAD_NUM, N_CTX,                 # B, H, N
    num_block,      # number of blocks along sequence dim N
    # Block sizes
    TILE_Q_SIZE: tl.constexpr, # Block size for Q (M dimension of tiles)
    BLOCK_DMODEL: tl.constexpr,# Head dimension (D)
    TILE_K_SIZE: tl.constexpr, # Block size for K/V (N dimension of tiles)
    PIPELINING: tl.constexpr,  # number of tiles to process in parallel (pipelining stages)
    CAUSAL: tl.constexpr,      # if true, apply causal mask
):
    off_hz = tl.program_id(0) # Combined Batch*Head index
    off_z = off_hz // HEAD_NUM # Batch index
    off_h = off_hz % HEAD_NUM  # Head index

    # offset pointers for batch/head using respective strides
    Q += off_z * stride_qb + off_h * stride_qh
    K += off_z * stride_kb + off_h * stride_kh
    V += off_z * stride_vb + off_h * stride_vh
    out += off_z * stride_ob + off_h * stride_oh   # Add offset for 'out' tensor
    dO += off_z * stride_dob + off_h * stride_doh
    dQ += off_z * stride_dqb + off_h * stride_dqh
    dK += off_z * stride_dkb + off_h * stride_dkh
    dV += off_z * stride_dvb + off_h * stride_dvh
    # Offsets for 3D intermediate tensors (L, M, D)
    L += off_z * stride_lb + off_h * stride_lh
    M += off_z * stride_mb + off_h * stride_mh
    D += off_z * stride_deltab + off_h * stride_deltah

    # TILE_Q_SIZE is the block size for the Q dimension (M)
    # TILE_K_SIZE is the block size for the K dimension (N)

    offs_k = tl.arange(0, BLOCK_DMODEL)
    num_q_blocks = num_block
    num_k_blocks = num_block

    # Outer loop over K blocks (K tiles)
    for n_block in range(num_k_blocks):
        start_n = n_block * TILE_K_SIZE
        offs_n = start_n + tl.arange(0, TILE_K_SIZE)
        # Accumulators for dK and dV for this K block
        dk_accum = tl.zeros([TILE_K_SIZE, BLOCK_DMODEL], dtype=tl.float32)
        dv_accum = tl.zeros([TILE_K_SIZE, BLOCK_DMODEL], dtype=tl.float32)
        # Inner loop over Q blocks (Q tiles)
        for m_block in range(num_q_blocks):
            start_m = m_block * TILE_Q_SIZE
            offs_m = start_m + tl.arange(0, TILE_Q_SIZE)
            # Load q, do, l, m, d for this Q block
            q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd)
            q = tl.load(q_ptrs)
            do_ptrs = dO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dod)
            do = tl.load(do_ptrs)
            l_ptrs = L + offs_m * stride_ln
            m_ptrs = M + offs_m * stride_mn
            d_ptrs = D + offs_m * stride_deltan
            l_i = tl.load(l_ptrs)
            m_i = tl.load(m_ptrs)
            Di = tl.load(d_ptrs)
            # Load k and v for this K block
            k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd)
            v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vd)
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            # Recompute attention scores for this Q/K block pair
            qk = tl.dot(q, tl.trans(k))  # [TILE_Q_SIZE, TILE_K_SIZE]
            # Apply causal mask if needed
            if CAUSAL:
                q_indices = start_m + tl.arange(0, TILE_Q_SIZE)
                k_indices = start_n + tl.arange(0, TILE_K_SIZE)
                causal_mask = q_indices[:, None] >= k_indices[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))
            qk = qk * SOFTMAX_SCALE
            p = tl.exp(qk - m_i[:, None])
            # Use correct Triton dtype for casting, e.g. tl.float16, tl.bfloat16, or tl.float32 as appropriate
            p_cast = p.to(tl.float16)
            # dv = p.T @ do
            p_trans = tl.trans(p_cast)
            do_cast = do.to(p_cast.dtype)  # or explicitly to fp16 if that’s your intent
            dv = tl.dot(p_trans, do_cast)
            # dp = do @ v.T
            v_trans = tl.trans(v)
            dp = tl.dot(do, v_trans)
            ds = p_cast * (dp - Di[:, None])
            ds = ds * SOFTMAX_SCALE
            ds_cast = ds.to(Q.dtype.element_ty)
            # dk = ds.T @ q
            ds_trans = tl.trans(ds_cast)
            dk = tl.dot(ds_trans, q)
            # Accumulate dK/dV for this K block
            dk_accum += dk
            dv_accum += dv
            # Compute dq_delta and write immediately
            dq_delta = tl.dot(ds_cast, k)
            dq_ptrs = dQ + (offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqd)
            dq_prev = tl.load(dq_ptrs, eviction_policy="evict_last")
            dq_prev += dq_delta
            tl.store(dq_ptrs, dq_prev, eviction_policy="evict_last")
        # After inner Q loop, store dK and dV for this K block
        dk_ptrs = dK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkd)
        dv_ptrs = dV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvd)
        tl.store(dk_ptrs, dk_accum)
        tl.store(dv_ptrs, dv_accum)



def self_attention_backward_slow(q, k, v, do, p, res_mask=None, sm_scale=None):
    """
    Backward pass for self-attention.
    
    Args:
        q: Query tensor of shape (batch_size, n_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, n_heads, seq_len, head_dim)
        v: Value tensor of shape (batch_size, n_heads, seq_len, head_dim)
        do: Gradient of output tensor of shape (batch_size, n_heads, seq_len, head_dim)
        p: Attention weights tensor of shape (batch_size, n_heads, seq_len, seq_len)
        res_mask: Optional mask tensor of shape (batch_size, n_heads, seq_len, 1) from forward pass
        sm_scale: Optional scaling factor for softmax (1/sqrt(head_dim))
    """
    dv = p.transpose(-2,-1) @ do
    dp = do @ v.transpose(-1,-2)
    d_scores = p * (dp - (dp * p).sum(dim=-1, keepdim=True))

    if sm_scale is not None:
        d_scores *= sm_scale
    
    dq = d_scores @ k
    dk = d_scores.transpose(-2, -1) @ q

    return dq, dk, dv


# --------------- class FlashAttentionFunction ---------------
class FlashAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, lens, sm_scale, autotune=True, prescale=False, causal=False):
        # Call Triton forward kernel
        batch, heads, seq_len, HEAD_DIM = q.shape
        assert HEAD_DIM in {16, 32, 64, 128, 256}
        assert HEAD_DIM == k.shape[-1] and HEAD_DIM == v.shape[-1]
        assert sm_scale is not None
        assert lens is None or (
            lens.dtype == torch.int32 and batch == len(lens) and lens.ndim == 1
        )

        O = torch.zeros_like(q, memory_format=torch.contiguous_format)

        Max = torch.empty((batch, heads, seq_len), dtype=torch.float32, device=q.device)
        SumExp = torch.empty((batch, heads, seq_len), dtype=torch.float32, device=q.device)

        INPUT_PRECISION = (
            "tf32" if torch.get_float32_matmul_precision() != "highest" else "ieee"
        )

        grid = lambda args: (
            batch, 
            heads, 
            triton.cdiv(seq_len, args["TILE_Q_SIZE"])
        )

        fwd_function = streaming_forward_autotuned if autotune else streaming_forward

        extra_kwargs = dict(RCP_LN2=1.0 / math.log(2.0))
        if not autotune:
            extra_kwargs['PIPELINING'] = 1

        fwd_function[grid](
            q, 
            k,
            v, 
            lens, 
            O, 
            Max,       # Pass Max tensor
            SumExp,    # Pass SumExp tensor
            *strides(q), 
            *strides(k), 
            *strides(v), 
            *strides(O), 
            *strides(Max),     # Pass Max strides
            *strides(SumExp),  # Pass SumExp strides
            *(strides(lens) if lens is not None else [0]), 
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

        # <<< Conversion Step >>>
        # The forward kernel used exp2 and saved m' = max(qkT * S * RCP_LN2).
        # The backward kernel uses exp and expects M = max(qkT * S).
        # Conversion: M = m' / RCP_LN2 = m' * ln(2).
        Max.mul_(math.log(2.0)) # Multiply Max tensor by ln(2) in-place

        # Determine the TILE_Q_SIZE used by the launched kernel
        best_config = getattr(fwd_function, 'best_config', None)
        if best_config: # Autotuner ran and found a best config
            TILE_Q_SIZE = best_config.kwargs['TILE_Q_SIZE']
        elif not autotune: # No autotuning, get the default
            default_cfg_vals = get_config(q.dtype, HEAD_DIM)
            TILE_Q_SIZE = default_cfg_vals[0]
        else: # Autotuner ran but somehow best_config is not set (should not happen)
            raise RuntimeError("Could not determine TILE_Q_SIZE after autotuning.")

        # Calculate the actual grid dimensions used based on the determined TILE_Q_SIZE
        launch_grid = (batch, heads, triton.cdiv(seq_len, TILE_Q_SIZE))

        # Save tensors needed for backward pass
        ctx.save_for_backward(q, k, v, O, lens, Max, SumExp)
        ctx.grid = launch_grid     # Save the calculated grid
        ctx.sm_scale = sm_scale
        ctx.autotune = autotune
        ctx.prescale = prescale
        ctx.causal = causal
        ctx.BLOCK_M = TILE_Q_SIZE     # Save the determined block size M (TILE_Q_SIZE)
        ctx.BLOCK_DMODEL = HEAD_DIM   # Save block size D (HEAD_DIM)
        ctx.PIPELINING = best_config.kwargs.get('PIPELINING', best_config.num_stages) if best_config else 1 # Retrieve PIPELINING used
        return O

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lens, m, l = ctx.saved_tensors # Corrected order based on save_for_backward
        # Retrieve saved values
        grid = ctx.grid
        BLOCK_M = ctx.BLOCK_M
        BLOCK_DMODEL = ctx.BLOCK_DMODEL
        sm_scale = ctx.sm_scale
        PIPELINING = ctx.PIPELINING
        # Ensure do is contiguous
        do = do.contiguous()

        # Allocate gradient tensors
        dq = torch.zeros_like(q, dtype=torch.float32) # Ensure dq is float32 for accumulation
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # Preprocess dO and calculate delta
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)

        # Grid for preprocess: Needs to cover all elements, launched in blocks of BLOCK_M
        batch, heads, seq_len, _ = q.shape
        n_blocks = triton.cdiv(seq_len, BLOCK_M)
        grid_preprocess = (n_blocks, batch * heads) # num_q_blocks, B * H

        self_attention_backward_preprocess[grid_preprocess](
            o, do, l,           # Inputs: output, grad_output, sumexp
            do_scaled, delta,   # Outputs: scaled_grad_output, delta
            *strides(o),        # Strides for o
            *strides(do),       # Strides for dO
            *strides(l),        # Strides for SumExp (l)
            *strides(do_scaled),# Strides for new_dO
            *strides(delta),    # Strides for delta
            HEAD_NUM=heads,     # Pass number of heads
            TILE_Q_SIZE=BLOCK_M,# Use the saved block size
            HEAD_DIM=BLOCK_DMODEL,
        )

        # Grid for main backward kernel: One program per batch * head
        grid_backward = (grid[0] * grid[1],)
        num_block = grid[2] # Number of blocks along sequence length

        # Determine num_warps based on head dimension
        num_warps = 4 if BLOCK_DMODEL <= 64 else 8

        self_attention_backward[grid_backward](
            q, k, v, sm_scale,      # Inputs: q, k, v, sm_scale
            o, do_scaled,           # Inputs: output, scaled_grad_output (dO/L)
            dq, dk, dv,             # Outputs: grad_q, grad_k, grad_v
            l, m,                   # Inputs: sumexp (L), max (M)
            delta,                  # Input: delta (D)
            # Strides for all tensors
            *strides(q),            # q strides
            *strides(k),            # k strides
            *strides(v),            # v strides
            *strides(o),            # o strides
            *strides(do_scaled),    # do_scaled strides
            *strides(dq),           # dq strides
            *strides(dk),           # dk strides
            *strides(dv),           # dv strides
            *strides(l),            # l (sumexp) strides
            *strides(m),            # m (max) strides
            *strides(delta),        # delta (D) strides
            # Meta params
            BATCH_NUM=batch, HEAD_NUM=heads, N_CTX=seq_len, # Dimensions
            num_block=num_block,    # Number of blocks
            # Block sizes
            TILE_Q_SIZE=BLOCK_M,    # Block size for Q (dim M)
            BLOCK_DMODEL=BLOCK_DMODEL,# Head dim
            TILE_K_SIZE=BLOCK_M,    # Block size for K (dim N), assume square
            num_warps=num_warps,
            num_stages=PIPELINING,  # Use PIPELINING parameter for pipelining stages
            PIPELINING=PIPELINING,  # Pass to kernel as constexpr
            CAUSAL=ctx.causal,      # Pass causal flag as tl.constexpr
        )
        # The backward function expects gradients for q, k, v, lens, sm_scale, autotune, prescale, causal
        # Only q, k, v require gradients. Others are None.
        return dq.to(q.dtype), dk, dv, None, None, None, None, None

def flash_attention(q, k, v, lens=None, sm_scale=None, autotune=True, prescale=False, causal=False):
    # Add default sm_scale if not provided
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return FlashAttentionFunction.apply(q, k, v, lens, sm_scale, autotune, prescale, causal)
