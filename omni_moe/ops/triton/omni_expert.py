import torch
import triton
from triton import language as tl

from omni_moe.ops.triton import utils, activations, omni_scheduler


@triton.autotune(
    configs=utils.get_expert_fwd_scores_tail_autotune_configs(),
    key=utils.EXPERT_FWD_SCORES_TAIL_AUTOTUNE_KEYS,
)
@triton.jit
def _fwd_scores_tail_kernel(
    X,
    W,
    S,
    tail_token_ids,
    tail_expert_ids,
    tail_offsets,
    stride_xm,
    stride_xn,
    stride_wk,
    stride_wn,
    stride_im,
    num_tokens,
    hidden_size: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    m_block = tl.program_id(0)
    k_idx = tl.program_id(1)

    # Initialize offsets
    offs_m = m_block * TILE_M + tl.arange(0, TILE_M)
    offs_nb = tl.arange(0, TILE_N)

    # Load segment boundaries
    seg_start = tl.load(tail_offsets + k_idx)
    seg_end = tl.load(tail_offsets + k_idx + 1)

    # Map compressed expert id to original expert id
    expert_ids = tl.load(tail_expert_ids + k_idx)

    # Compute pair ids
    pair_ids = seg_start + offs_m
    mask_m = pair_ids < seg_end

    # Load token ids
    token_ids = tl.load(
        tail_token_ids + pair_ids * stride_im,
        mask=mask_m,
        other=-1,
    )
    mask_m &= token_ids < num_tokens

    # Initialize pointers
    x_ptrs = X + token_ids[:, None] * stride_xm
    w_ptrs = W + expert_ids * stride_wk

    # Initialize accumulator for s
    acc_s = tl.zeros((TILE_M,), dtype=tl.float32)

    # Loop over hidden dimension
    for n_block in range(0, hidden_size, TILE_N):
        n_block = tl.multiple_of(n_block, TILE_N)
        offs_n = n_block + offs_nb
        mask_n = offs_n < hidden_size

        # Load x
        x = tl.load(
            x_ptrs + offs_n[None, :] * stride_xn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )

        # Load w
        w = tl.load(
            w_ptrs + offs_n * stride_wn,
            mask=mask_n,
            other=0.0,
        )

        # Compute s
        acc_s += tl.sum(x * w[None, :], axis=1)

    # Write back s
    tl.store(S + pair_ids, acc_s, mask=mask_m)


@triton.autotune(
    configs=utils.get_expert_fwd_states_tail_autotune_configs(),
    key=utils.EXPERT_FWD_STATES_TAIL_AUTOTUNE_KEYS,
    reset_to_zero=["Out"],
)
@triton.jit
def _fwd_states_tail_kernel(
    S,
    V,
    G,
    Out,
    tail_token_ids,
    tail_expert_ids,
    tail_offsets,
    stride_sm,
    stride_vk,
    stride_vn,
    stride_gm,
    stride_im,
    stride_om,
    stride_on,
    num_tokens,
    hidden_size: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)
    k_idx = tl.program_id(2)

    # Initialize offsets
    offs_m = m_block * TILE_M + tl.arange(0, TILE_M)
    offs_n = n_block * TILE_N + tl.arange(0, TILE_N)
    mask_n = offs_n < hidden_size

    # Load segment boundaries
    seg_start = tl.load(tail_offsets + k_idx)
    seg_end = tl.load(tail_offsets + k_idx + 1)

    # Map compressed expert id to original expert id
    expert_ids = tl.load(tail_expert_ids + k_idx)

    # Compute pair ids
    pair_ids = seg_start + offs_m
    mask_m = pair_ids < seg_end

    # Load token ids
    tail_token_ids = tl.load(
        tail_token_ids + pair_ids * stride_im,
        mask=mask_m,
        other=0,
    )
    mask_m &= tail_token_ids < num_tokens

    # Initialize pointers
    s_ptrs = S + pair_ids * stride_sm
    g_ptrs = G + pair_ids * stride_gm
    v_ptrs = V + expert_ids * stride_vk + offs_n * stride_vn
    o_ptrs = Out + tail_token_ids[:, None] * stride_om + offs_n[None, :] * stride_on

    # Load s
    s = tl.load(s_ptrs, mask=mask_m, other=0.0)
    # Load g
    g = tl.load(g_ptrs, mask=mask_m, other=0.0)

    # Compute gated s
    gs = activations.silu(s).cast(g.dtype) * g

    # Load v
    v = tl.load(v_ptrs, mask=mask_n, other=0.0)
    # Compute o
    o = gs[:, None] * v[None, :]

    # Write back o
    tl.atomic_add(
        o_ptrs,
        o,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.autotune(
    configs=utils.get_expert_fwd_scores_group_autotune_configs(),
    key=utils.EXPERT_FWD_SCORES_GROUP_AUTOTUNE_KEYS,
)
@triton.jit
def _fwd_scores_group_kernel(
    X,
    W,
    S,
    group_token_ids,
    group_expert_ids,
    group_offsets,
    stride_xm,
    stride_xn,
    stride_wk,
    stride_wn,
    stride_sm,
    stride_sk,
    num_tokens,
    group_size: tl.constexpr,
    hidden_size: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    m_block = tl.program_id(0)
    k_block = tl.program_id(1)
    g_idx = tl.program_id(2)

    # Initialize offsets
    offs_m = m_block * TILE_M + tl.arange(0, TILE_M)
    offs_k = k_block * TILE_K + tl.arange(0, TILE_K)
    offs_nb = tl.arange(0, TILE_N)

    # Load segment boundaries
    seg_start = tl.load(group_offsets + g_idx)
    seg_end = tl.load(group_offsets + g_idx + 1)

    # Compute row ids
    row_ids = seg_start + offs_m
    mask_m = row_ids < seg_end
    mask_k = offs_k < group_size

    # Load token ids
    token_ids = tl.load(
        group_token_ids + row_ids,
        mask=mask_m,
        other=-1,
    )
    mask_m &= token_ids < num_tokens

    # Load expert ids
    expert_ids = tl.load(
        group_expert_ids + g_idx * group_size + offs_k,
        mask=mask_k,
        other=0,
    )

    # Initialize pointers
    x_ptrs = X + token_ids[:, None] * stride_xm
    w_ptrs = W + expert_ids[None, :] * stride_wk
    s_ptrs = S + row_ids[:, None] * stride_sm + offs_k[None, :] * stride_sk

    # Initialize accumulator for s
    acc_s = tl.zeros((TILE_M, TILE_K), dtype=tl.float32)

    # Loop over hidden dimension
    for n_block in range(0, hidden_size, TILE_N):
        n_block = tl.multiple_of(n_block, TILE_N)
        offs_n = n_block + offs_nb
        mask_n = offs_n < hidden_size

        # Load x
        x = tl.load(
            x_ptrs + offs_n[None, :] * stride_xn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )

        # Load w
        w = tl.load(
            w_ptrs + offs_n[:, None] * stride_wn,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Compute s
        acc_s += tl.dot(x, w)

    # Write back s
    tl.store(s_ptrs, acc_s, mask=mask_m[:, None] & mask_k[None, :])


@triton.autotune(
    configs=utils.get_expert_fwd_states_group_autotune_configs(),
    key=utils.EXPERT_FWD_STATES_GROUP_AUTOTUNE_KEYS,
    reset_to_zero=["Out"],
)
@triton.jit
def _fwd_states_group_kernel(
    S,
    V,
    G,
    Out,
    group_token_ids,
    group_expert_ids,
    group_offsets,
    stride_sm,
    stride_sk,
    stride_vk,
    stride_vn,
    stride_gm,
    stride_gk,
    stride_im,
    stride_om,
    stride_on,
    num_tokens,
    group_size: tl.constexpr,
    hidden_size: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)
    g_idx = tl.program_id(2)

    # Initialize offsets
    offs_m = m_block * TILE_M + tl.arange(0, TILE_M)
    offs_n = n_block * TILE_N + tl.arange(0, TILE_N)
    mask_n = offs_n < hidden_size

    # Load segment boundaries
    seg_start = tl.load(group_offsets + g_idx)
    seg_end = tl.load(group_offsets + g_idx + 1)

    # Compute row ids
    row_ids = seg_start + offs_m
    mask_m = row_ids < seg_end

    # Load token ids
    token_ids = tl.load(
        group_token_ids + row_ids * stride_im,
        mask=mask_m,
        other=0,
    )
    mask_m &= token_ids < num_tokens

    # Initialize pointers
    s_ptrs = S + row_ids[:, None] * stride_sm
    g_ptrs = G + row_ids[:, None] * stride_gm
    v_ptrs = V + offs_n[None, :] * stride_vn
    o_ptrs = Out + token_ids[:, None] * stride_om + offs_n[None, :] * stride_on

    # Initialize accumulator for o
    acc_o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Loop over expert dimension
    for k_block in range(0, group_size, TILE_K):
        offs_k = k_block + tl.arange(0, TILE_K)
        mask_k = offs_k < group_size

        # Load expert ids
        expert_ids = tl.load(
            group_expert_ids + g_idx * group_size + offs_k,
            mask=mask_k,
            other=0,
        )

        # Load s
        s = tl.load(
            s_ptrs + offs_k[None, :] * stride_sk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Load g
        g = tl.load(
            g_ptrs + offs_k[None, :] * stride_gk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Compute gated s
        gs = activations.silu(s).cast(g.dtype) * g

        # Load v
        v = tl.load(
            v_ptrs + expert_ids[:, None] * stride_vk,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )

        # Compute o
        acc_o += tl.dot(gs, v)

    # Write back o
    tl.atomic_add(
        o_ptrs,
        acc_o,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.autotune(
    configs=utils.get_expert_bwd_states_tail_autotune_configs(),
    key=utils.EXPERT_BWD_STATES_TAIL_AUTOTUNE_KEYS,
)
@triton.jit
def _bwd_states_tail_kernel(
    V,
    G,
    S,
    dO,
    dG,
    dS,
    tail_token_ids,
    tail_expert_ids,
    tail_offsets,
    tail_pair_ids,
    stride_vk,
    stride_vn,
    stride_gm,
    stride_sm,
    stride_dom,
    stride_don,
    stride_dgm,
    stride_dsm,
    stride_im,
    stride_pm,
    num_tokens,
    hidden_size: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    m_block = tl.program_id(0)
    k_idx = tl.program_id(1)

    # Initialize offsets
    offs_m = m_block * TILE_M + tl.arange(0, TILE_M)
    offs_nb = tl.arange(0, TILE_N)

    # Load segment boundaries
    seg_start = tl.load(tail_offsets + k_idx)
    seg_end = tl.load(tail_offsets + k_idx + 1)

    # Map compressed expert id to original expert id
    expert_ids = tl.load(tail_expert_ids + k_idx)

    # Compute pair ids
    pair_ids = seg_start + offs_m
    mask_m = pair_ids < seg_end

    # Load token ids
    token_ids = tl.load(
        tail_token_ids + pair_ids * stride_im,
        mask=mask_m,
        other=0,
    )
    mask_m &= token_ids < num_tokens

    # Load p
    p = tl.load(tail_pair_ids + pair_ids * stride_pm, mask=mask_m, other=0)

    # Initialize pointers
    v_ptrs = V + expert_ids * stride_vk
    g_ptrs = G + pair_ids * stride_gm
    s_ptrs = S + pair_ids * stride_sm
    do_ptrs = dO + token_ids[:, None] * stride_dom
    dg_ptrs = dG + p * stride_dgm
    ds_ptrs = dS + pair_ids * stride_dsm

    # Load g
    g = tl.load(g_ptrs, mask=mask_m, other=0.0)

    # Load s
    s = tl.load(s_ptrs, mask=mask_m, other=0.0).to(tl.float32)

    # Initialize accumulator for dg
    acc_dg = tl.zeros((TILE_M,), dtype=tl.float32)

    # Loop over hidden dimension
    for start_n in range(0, hidden_size, TILE_N):
        start_n = tl.multiple_of(start_n, TILE_N)
        offs_n = start_n + offs_nb
        mask_n = offs_n < hidden_size

        do = tl.load(
            do_ptrs + offs_n[None, :] * stride_don,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )

        v = tl.load(
            v_ptrs + offs_n * stride_vn,
            mask=mask_n,
            other=0.0,
        )

        acc_dg += tl.sum(do * v[None, :], axis=1)

    # recomputation to save memory
    sig_s = tl.sigmoid(s)
    dsilu = sig_s + s * sig_s * (1.0 - sig_s)

    # Compute dg
    dg = (acc_dg * sig_s * s).to(g.dtype)

    # Store dg
    tl.store(dg_ptrs, dg, mask=mask_m)

    # Compute ds
    ds = acc_dg * g.to(tl.float32) * dsilu

    # Store ds
    tl.store(ds_ptrs, ds, mask=mask_m)


@triton.autotune(
    configs=utils.get_expert_bwd_scores_tail_autotune_configs(),
    key=utils.EXPERT_BWD_SCORES_TAIL_AUTOTUNE_KEYS,
    reset_to_zero=["dX", "dW", "dV"],
)
@triton.jit
def _bwd_scores_tail_kernel(
    X,
    W,
    G,
    S,
    dO,
    dX,
    dW,
    dV,
    dS,
    tail_token_ids,
    tail_expert_ids,
    tail_offsets,
    stride_xm,
    stride_xn,
    stride_wk,
    stride_wn,
    stride_gm,
    stride_sm,
    stride_dom,
    stride_don,
    stride_dxm,
    stride_dxn,
    stride_dwk,
    stride_dwn,
    stride_dvk,
    stride_dvn,
    stride_dsm,
    stride_im,
    num_tokens,
    hidden_size: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)
    k_idx = tl.program_id(2)

    # Initialize offsets
    offs_m = m_block * TILE_M + tl.arange(0, TILE_M)
    offs_n = n_block * TILE_N + tl.arange(0, TILE_N)
    mask_n = offs_n < hidden_size

    # Load segment boundaries
    seg_start = tl.load(tail_offsets + k_idx)
    seg_end = tl.load(tail_offsets + k_idx + 1)

    # Map compressed expert ids to original expert ids
    expert_ids = tl.load(tail_expert_ids + k_idx)

    # Compute pair ids
    pair_ids = seg_start + offs_m
    mask_m = pair_ids < seg_end

    # Load token ids
    token_ids = tl.load(
        tail_token_ids + pair_ids * stride_im,
        mask=mask_m,
        other=0,
    )
    mask_m &= token_ids < num_tokens

    # Initialize pointers
    x_ptrs = X + token_ids[:, None] * stride_xm + offs_n[None, :] * stride_xn
    w_ptrs = W + expert_ids * stride_wk + offs_n * stride_wn
    g_ptrs = G + pair_ids * stride_gm
    s_ptrs = S + pair_ids * stride_sm
    do_ptrs = dO + token_ids[:, None] * stride_dom + offs_n[None, :] * stride_don
    dx_ptrs = dX + token_ids[:, None] * stride_dxm + offs_n[None, :] * stride_dxn
    dw_ptrs = dW + expert_ids * stride_dwk + offs_n * stride_dwn
    dv_ptrs = dV + expert_ids * stride_dvk + offs_n * stride_dvn
    ds_ptrs = dS + pair_ids * stride_dsm

    # Load ds
    ds = tl.load(ds_ptrs, mask=mask_m, other=0.0)

    # Load w
    w = tl.load(w_ptrs, mask=mask_n, other=0.0)

    # Compute dx
    dx = ds[:, None] * w[None, :]

    # Store dx
    tl.atomic_add(
        dx_ptrs,
        dx.to(w.dtype),
        mask=mask_m[:, None] & mask_n[None, :],
    )

    # Load x
    x = tl.load(
        x_ptrs,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    # Compute dw
    dw = tl.sum(ds[:, None] * x, axis=0)

    # Store dw
    tl.atomic_add(dw_ptrs, dw.to(x.dtype), mask=mask_n)

    # Load g
    g = tl.load(g_ptrs, mask=mask_m, other=0.0)

    # Load s
    s = tl.load(s_ptrs, mask=mask_m, other=0.0)

    gate = activations.silu(s) * g

    # Load do
    do = tl.load(
        do_ptrs,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    # Compute dv
    dv = tl.sum(gate[:, None] * do, axis=0)

    # Store dv
    tl.atomic_add(dv_ptrs, dv.to(x.dtype), mask=mask_n)


def omni_expert_forward(
    hidden_states: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    routing_weights: torch.Tensor,
    indices: torch.Tensor,
):
    num_tokens, hidden_size = hidden_states.shape
    num_experts = up_weights.shape[0]

    # Get scheduling info
    scheduling_info = omni_scheduler.get_scheduling_info(
        routing_weights,
        indices,
        num_experts,
        group_size=16,  # minimum group size for matmul efficiency
    )

    # Allocate outputs
    tail_expert_weights = torch.empty(
        scheduling_info.tail_token_ids.numel(),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    out = torch.zeros_like(hidden_states)

    def tail_scores_grid(META):
        return (
            triton.cdiv(scheduling_info.max_tail_pairs_per_expert, META["TILE_M"]),
            scheduling_info.num_tail_experts,
        )

    _fwd_scores_tail_kernel[tail_scores_grid](
        hidden_states,
        up_weights,
        tail_expert_weights,
        scheduling_info.tail_token_ids,
        scheduling_info.tail_expert_ids,
        scheduling_info.tail_offsets,
        hidden_states.stride(0),
        hidden_states.stride(1),
        up_weights.stride(0),
        up_weights.stride(1),
        scheduling_info.tail_token_ids.stride(0),
        num_tokens,
        hidden_size=hidden_size,
    )

    def tail_states_grid(META):
        return (
            triton.cdiv(scheduling_info.max_tail_pairs_per_expert, META["TILE_M"]),
            triton.cdiv(hidden_size, META["TILE_N"]),
            scheduling_info.num_tail_experts,
        )

    _fwd_states_tail_kernel[tail_states_grid](
        tail_expert_weights,
        down_weights,
        scheduling_info.tail_routing_weights,
        out,
        scheduling_info.tail_token_ids,
        scheduling_info.tail_expert_ids,
        scheduling_info.tail_offsets,
        tail_expert_weights.stride(0),
        down_weights.stride(0),
        down_weights.stride(1),
        scheduling_info.tail_routing_weights.stride(0),
        scheduling_info.tail_token_ids.stride(0),
        out.stride(0),
        out.stride(1),
        num_tokens,
        hidden_size=hidden_size,
    )

    return out, tail_expert_weights, scheduling_info


def omni_expert_backward(
    do: torch.Tensor,
    hidden_states: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    routing_weights: torch.Tensor,
    tail_expert_weights: torch.Tensor,
    tail_routing_weights: torch.Tensor,
    tail_token_ids: torch.Tensor,
    tail_expert_ids: torch.Tensor,
    tail_offsets: torch.Tensor,
    tail_sorted_pair_ids: torch.Tensor,
):
    num_tokens, hidden_size = hidden_states.shape
    num_tail_experts = tail_expert_ids.numel()

    # recompute max_tail_pairs_per_expert because it is not saved in ctx
    max_tail_pairs_per_expert = int(
        torch.max(tail_offsets[1:] - tail_offsets[:-1]).item()
    )

    # Allocate outputs
    dx = torch.zeros_like(hidden_states)
    dw = torch.zeros_like(up_weights)
    dv = torch.zeros_like(down_weights)
    dg = torch.empty_like(routing_weights).view(-1)
    ds = torch.empty_like(tail_expert_weights)

    def tail_states_grid(META):
        return (
            triton.cdiv(max_tail_pairs_per_expert, META["TILE_M"]),
            num_tail_experts,
        )

    _bwd_states_tail_kernel[tail_states_grid](
        down_weights,
        tail_routing_weights,
        tail_expert_weights,
        do,
        dg,
        ds,
        tail_token_ids,
        tail_expert_ids,
        tail_offsets,
        tail_sorted_pair_ids,
        down_weights.stride(0),
        down_weights.stride(1),
        tail_routing_weights.stride(0),
        tail_expert_weights.stride(0),
        do.stride(0),
        do.stride(1),
        dg.stride(0),
        ds.stride(0),
        tail_token_ids.stride(0),
        tail_sorted_pair_ids.stride(0),
        num_tokens,
        hidden_size=hidden_size,
    )

    def tail_scores_grid(META):
        return (
            triton.cdiv(max_tail_pairs_per_expert, META["TILE_M"]),
            triton.cdiv(hidden_size, META["TILE_N"]),
            num_tail_experts,
        )

    _bwd_scores_tail_kernel[tail_scores_grid](
        hidden_states,
        up_weights,
        tail_routing_weights,
        tail_expert_weights,
        do,
        dx,
        dw,
        dv,
        ds,
        tail_token_ids,
        tail_expert_ids,
        tail_offsets,
        hidden_states.stride(0),
        hidden_states.stride(1),
        up_weights.stride(0),
        up_weights.stride(1),
        tail_routing_weights.stride(0),
        tail_expert_weights.stride(0),
        do.stride(0),
        do.stride(1),
        dx.stride(0),
        dx.stride(1),
        dw.stride(0),
        dw.stride(1),
        dv.stride(0),
        dv.stride(1),
        ds.stride(0),
        tail_token_ids.stride(0),
        num_tokens,
        hidden_size=hidden_size,
    )

    dg = dg.view_as(routing_weights)

    return dx, dw, dv, dg


class OmniExpertFunc(torch.autograd.Function):
    @staticmethod
    @utils.ensure_contiguous
    def forward(ctx, hidden_states, up_weights, down_weights, routing_weights, indices):
        experts_states, tail_expert_weights, scheduling_info = omni_expert_forward(
            hidden_states,
            up_weights,
            down_weights,
            routing_weights,
            indices,
        )

        ctx.save_for_backward(
            hidden_states,
            up_weights,
            down_weights,
            routing_weights,
            tail_expert_weights,
            scheduling_info.tail_routing_weights,
            scheduling_info.tail_token_ids,
            scheduling_info.tail_expert_ids,
            scheduling_info.tail_offsets,
            scheduling_info.tail_sorted_pair_ids,
        )

        return experts_states

    @staticmethod
    @utils.ensure_contiguous
    def backward(ctx, do):
        (
            hidden_states,
            up_weights,
            down_weights,
            routing_weights,
            tail_expert_weights,
            tail_routing_weights,
            tail_token_ids,
            tail_expert_ids,
            tail_offsets,
            tail_sorted_pair_ids,
        ) = ctx.saved_tensors

        dx, dw, dv, dg = omni_expert_backward(
            do,
            hidden_states,
            up_weights,
            down_weights,
            routing_weights,
            tail_expert_weights,
            tail_routing_weights,
            tail_token_ids,
            tail_expert_ids,
            tail_offsets,
            tail_sorted_pair_ids,
        )

        return dx, dw, dv, dg, None


def triton_omni_expert_func(
    hidden_states, up_weights, down_weights, routing_weights, indices
):
    return OmniExpertFunc.apply(
        hidden_states, up_weights, down_weights, routing_weights, indices
    )
