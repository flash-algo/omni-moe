import triton
from triton import language as tl


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.heuristics(
    {
        "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_scores_tail_kernel(
    X,
    W,
    S,
    sorted_token_ids,
    expert_offsets,
    stride_xm,
    stride_xn,
    stride_wk,
    stride_wn,
    stride_im,
    num_tokens,
    hidden_size: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_nb = tl.arange(0, BLOCK_N)

    # Load segment boundaries
    seg_start = tl.load(expert_offsets + pid_k)
    seg_end = tl.load(expert_offsets + pid_k + 1)

    # Compute pair ids
    pair_ids = seg_start + offs_m
    mask_m = pair_ids < seg_end

    # Load token ids
    token_ids = tl.load(
        sorted_token_ids + pair_ids * stride_im,
        mask=mask_m,
        other=0,
    )
    mask_m &= token_ids < num_tokens

    # Initialize pointers
    x_ptrs = X + token_ids[:, None] * stride_xm
    w_ptrs = W + pid_k * stride_wk

    # Initialize accumulator for s
    acc_s = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over hidden dimension
    for start_n in range(0, hidden_size, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_nb

        # Load x
        if EVEN_N:
            x = tl.load(
                x_ptrs + offs_n[None, :] * stride_xn,
                mask=mask_m[:, None],
                other=0.0,
            )
        else:
            x = tl.load(
                x_ptrs + offs_n[None, :] * stride_xn,
                mask=mask_m[:, None] & (offs_n[None, :] < hidden_size),
                other=0.0,
            )

        # Load w
        if EVEN_N:
            w = tl.load(w_ptrs + offs_n * stride_wn)
        else:
            w = tl.load(
                w_ptrs + offs_n * stride_wn,
                mask=offs_n < hidden_size,
                other=0.0,
            )

        # Compute s
        acc_s += tl.sum(x * w[None, :], axis=1)

    # Write back s
    tl.store(
        S + pair_ids,
        acc_s,
        mask=mask_m,
    )


@triton.autotune(
    reset_to_zero=["Out"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_states_tail_kernel(
    S,
    V,
    G,
    Out,
    sorted_token_ids,
    expert_offsets,
    stride_sm,
    stride_vk,
    stride_vn,
    stride_gm,
    stride_im,
    stride_om,
    stride_on,
    num_tokens,
    hidden_size: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load segment boundaries
    seg_start = tl.load(expert_offsets + pid_k)
    seg_end = tl.load(expert_offsets + pid_k + 1)

    # Compute pair ids
    pair_ids = seg_start + offs_m
    mask_m = pair_ids < seg_end

    # Load token ids
    token_ids = tl.load(
        sorted_token_ids + pair_ids * stride_im,
        mask=mask_m,
        other=0,
    )
    mask_m &= token_ids < num_tokens

    # Initialize pointers
    s_ptrs = S + pair_ids * stride_sm
    g_ptrs = G + pair_ids * stride_gm
    v_ptrs = V + pid_k * stride_vk + offs_n * stride_vn
    o_ptrs = Out + token_ids[:, None] * stride_om + offs_n[None, :] * stride_on

    # Load s
    s = tl.load(s_ptrs, mask=mask_m, other=0.0)
    # Load g
    g = tl.load(g_ptrs, mask=mask_m, other=0.0)

    # Compute gated s
    gs = silu(s).cast(g.dtype) * g

    # Load v
    if EVEN_N:
        v = tl.load(v_ptrs)
    else:
        v = tl.load(v_ptrs, mask=offs_n < hidden_size, other=0.0)
    # Compute o
    o = gs[:, None] * v[None, :]

    # Write back o
    if EVEN_N:
        tl.atomic_add(o_ptrs, o, mask=mask_m[:, None])
    else:
        tl.atomic_add(
            o_ptrs,
            o,
            mask=mask_m[:, None] & (offs_n[None, :] < hidden_size),
        )


@triton.heuristics(
    {
        "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_scores_group_kernel(
    X,
    W,
    S,
    token_rows,
    group_offsets,
    active_expert_ids,
    stride_xm,
    stride_xn,
    stride_wk,
    stride_wn,
    stride_sm,
    stride_sk,
    stride_rm,
    stride_off,
    stride_am,
    num_tokens,
    num_experts,
    num_active,
    hidden_size: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_g = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, GROUP_SIZE)
    offs_nb = tl.arange(0, BLOCK_N)

    seg_start = tl.load(group_offsets + pid_g * stride_off)
    seg_end = tl.load(group_offsets + (pid_g + 1) * stride_off)

    row_ids = seg_start + offs_m
    mask_m = row_ids < seg_end

    token_ids = tl.load(token_rows + row_ids * stride_rm, mask=mask_m, other=0)
    mask_m &= token_ids < num_tokens

    expert_comp_ids = pid_g * GROUP_SIZE + offs_k
    mask_k = expert_comp_ids < num_active

    expert_orig = tl.load(
        active_expert_ids + expert_comp_ids * stride_am,
        mask=mask_k,
        other=0,
    ).to(tl.int32)
    mask_k &= expert_orig < num_experts

    acc_s = tl.zeros((BLOCK_M, GROUP_SIZE), dtype=tl.float32)
    x_ptrs = X + token_ids[:, None] * stride_xm

    for start_n in range(0, hidden_size, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_nb

        if EVEN_N:
            x = tl.load(
                x_ptrs + offs_n[None, :] * stride_xn,
                mask=mask_m[:, None],
                other=0.0,
            )
        else:
            x = tl.load(
                x_ptrs + offs_n[None, :] * stride_xn,
                mask=mask_m[:, None] & (offs_n[None, :] < hidden_size),
                other=0.0,
            )

        w_tile = tl.load(
            W + expert_orig[None, :] * stride_wk + offs_n[:, None] * stride_wn,
            mask=mask_k[None, :]
            if EVEN_N
            else (mask_k[None, :] & (offs_n[:, None] < hidden_size)),
            other=0.0,
        )

        acc_s += tl.dot(x, w_tile)

    tl.store(
        S + row_ids[:, None] * stride_sm + offs_k[None, :] * stride_sk,
        acc_s,
        mask=mask_m[:, None] & mask_k[None, :],
    )


@triton.heuristics(
    {
        "EVEN_N": lambda args: args["hidden_size"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_states_group_kernel(
    S,
    V,
    g_dense,
    Out,
    token_rows,
    group_offsets,
    active_expert_ids,
    stride_sm,
    stride_sk,
    stride_vk,
    stride_vn,
    stride_gm,
    stride_gk,
    stride_om,
    stride_on,
    stride_rm,
    stride_off,
    stride_am,
    num_tokens,
    num_experts,
    num_active,
    hidden_size: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_g = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, GROUP_SIZE)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    seg_start = tl.load(group_offsets + pid_g * stride_off)
    seg_end = tl.load(group_offsets + (pid_g + 1) * stride_off)

    row_ids = seg_start + offs_m
    mask_m = row_ids < seg_end

    token_ids = tl.load(token_rows + row_ids * stride_rm, mask=mask_m, other=0)
    mask_m &= token_ids < num_tokens

    expert_comp_ids = pid_g * GROUP_SIZE + offs_k
    mask_k = expert_comp_ids < num_active

    expert_orig = tl.load(
        active_expert_ids + expert_comp_ids * stride_am,
        mask=mask_k,
        other=0,
    ).to(tl.int32)
    mask_k &= expert_orig < num_experts

    s = tl.load(
        S + row_ids[:, None] * stride_sm + offs_k[None, :] * stride_sk,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    )

    g = tl.load(
        g_dense + row_ids[:, None] * stride_gm + offs_k[None, :] * stride_gk,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    )

    gs = (silu(s) * g).to(g.dtype)

    v_tile = tl.load(
        V + expert_orig[:, None] * stride_vk + offs_n[None, :] * stride_vn,
        mask=mask_k[:, None]
        if EVEN_N
        else (mask_k[:, None] & (offs_n[None, :] < hidden_size)),
        other=0.0,
    )

    o = tl.dot(gs, v_tile)
    o_ptrs = Out + token_ids[:, None] * stride_om

    if EVEN_N:
        tl.atomic_add(
            o_ptrs + offs_n[None, :] * stride_on,
            o.to(tl.float32),
            mask=mask_m[:, None],
        )
    else:
        tl.atomic_add(
            o_ptrs + offs_n[None, :] * stride_on,
            o.to(tl.float32),
            mask=mask_m[:, None] & (offs_n[None, :] < hidden_size),
        )
