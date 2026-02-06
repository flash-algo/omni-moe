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
    token_ids,
    expert_ids,
    expert_offsets,
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
    seg_start = tl.load(expert_offsets + k_idx)
    seg_end = tl.load(expert_offsets + k_idx + 1)

    # Map compressed expert id to original expert id
    expert_id = tl.load(expert_ids + k_idx)

    # Compute pair ids
    pair_ids = seg_start + offs_m
    mask_m = pair_ids < seg_end

    # Load token ids
    token_ids = tl.load(
        token_ids + pair_ids * stride_im,
        mask=mask_m,
        other=-1,
    )
    mask_m &= token_ids < num_tokens

    # Initialize pointers
    x_ptrs = X + token_ids[:, None] * stride_xm
    w_ptrs = W + expert_id * stride_wk

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
    token_ids,
    expert_ids,
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
    seg_start = tl.load(expert_offsets + k_idx)
    seg_end = tl.load(expert_offsets + k_idx + 1)

    # Map compressed expert id to original expert id
    expert_id = tl.load(expert_ids + k_idx)

    # Compute pair ids
    pair_ids = seg_start + offs_m
    mask_m = pair_ids < seg_end

    # Load token ids
    token_ids = tl.load(
        token_ids + pair_ids * stride_im,
        mask=mask_m,
        other=0,
    )
    mask_m &= token_ids < num_tokens

    # Initialize pointers
    s_ptrs = S + pair_ids * stride_sm
    g_ptrs = G + pair_ids * stride_gm
    v_ptrs = V + expert_id * stride_vk + offs_n * stride_vn
    o_ptrs = Out + token_ids[:, None] * stride_om + offs_n[None, :] * stride_on

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


def _omni_expert_forward(
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

    return out, scheduling_info


class OmniExpertFunc(torch.autograd.Function):
    @staticmethod
    @utils.ensure_contiguous
    def forward(ctx, hidden_states, up_weights, down_weights, routing_weights, indices):
        experts_states, scheduling_info = _omni_expert_forward(
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
        )

        return experts_states

    # @staticmethod
    # @utils.ensure_contiguous
    # def backward(ctx, dexperts_states):
    #     (
    #         hidden_states,
    #         down_weights,
    #         up_weights,
    #         routing_weights,
    #         expert_scores,
    #         sorted_routing_weights,
    #         sorted_token_ids,
    #         expert_offsets,
    #         sorted_pair_ids,
    #     ) = ctx.saved_tensors

    #     dhidden_states, ddown_weights, dup_weights, drouting_weights = (
    #         _flash_expert_backward(
    #             dexperts_states,
    #             hidden_states,
    #             down_weights,
    #             up_weights,
    #             routing_weights,
    #             expert_scores,
    #             sorted_routing_weights,
    #             sorted_token_ids,
    #             expert_offsets,
    #             sorted_pair_ids,
    #         )
    #     )

    #     return dhidden_states, ddown_weights, dup_weights, None, drouting_weights


def triton_omni_expert_func(
    hidden_states, up_weights, down_weights, routing_weights, indices
):
    return OmniExpertFunc.apply(
        hidden_states, up_weights, down_weights, routing_weights, indices
    )
