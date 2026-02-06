from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SchedulingInfo:
    """
    Scheduling information for omni expert kernels.

    Attributes:
        tail_token_ids (torch.Tensor): token ids for tail pairs
        tail_expert_ids (torch.Tensor): expert ids for tail pairs
        tail_routing_weights (torch.Tensor): routing weights for tail pairs
        tail_offsets (torch.Tensor): pair segment offsets for tail experts
        tail_sorted_pair_ids (torch.Tensor): sorted indices into original tail pairs
        num_tail_experts (int): number of unique tail experts
        max_tail_pairs_per_expert (int): maximum number of tail pairs per expert
    """

    # Tail info
    tail_token_ids: torch.Tensor = None
    tail_expert_ids: torch.Tensor = None
    tail_routing_weights: torch.Tensor = None
    tail_offsets: torch.Tensor = None
    tail_sorted_pair_ids: torch.Tensor = None
    num_tail_experts: int = 0
    max_tail_pairs_per_expert: int = 0

    # # Group info
    # group_token_ids: torch.Tensor = None
    # group_expert_ids: torch.Tensor = None
    # group_routing_weights: torch.Tensor = None
    # group_offsets: torch.Tensor = None
    # max_rows_per_group: int = 0
    # num_groups: int = 0


def get_scheduling_info(
    G: torch.Tensor,
    Indices: torch.Tensor,
    num_experts: int,
    group_size: int,
):
    num_tokens, num_experts_per_token = G.shape

    # Determine if grouping is possible
    is_group = num_tokens * num_experts_per_token / num_experts >= group_size
    is_group = False  # Disable grouping for now

    # Initialize basic info
    token_ids = (
        torch.arange(
            num_tokens * num_experts_per_token, device=Indices.device, dtype=torch.int32
        )
        // num_experts_per_token
    )
    expert_ids = Indices.flatten().to(torch.int32)
    G = G.flatten()

    # All pairs are tail
    if not is_group:
        # Sort by expert ids
        sorted_expert_ids, sorted_pair_ids = torch.sort(expert_ids)
        sorted_pair_ids = sorted_pair_ids.to(torch.int32)

        # Get tail info
        tail_token_ids = token_ids[sorted_pair_ids]
        tail_routing_weights = G[sorted_pair_ids]

        # Build compressed expert list and offsets from sorted pairs
        tail_expert_ids, expert_counts = torch.unique_consecutive(
            sorted_expert_ids, return_counts=True
        )
        num_tail_experts = tail_expert_ids.numel()

        tail_offsets = torch.zeros(
            num_tail_experts + 1, dtype=torch.int32, device=Indices.device
        )
        tail_offsets[1:] = torch.cumsum(expert_counts, dim=0)

        # Get max tail pairs per expert
        max_tail_pairs_per_expert = torch.max(expert_counts).item()

        return SchedulingInfo(
            tail_token_ids=tail_token_ids,
            tail_expert_ids=tail_expert_ids,
            tail_routing_weights=tail_routing_weights,
            tail_offsets=tail_offsets,
            tail_sorted_pair_ids=sorted_pair_ids,
            num_tail_experts=num_tail_experts,
            max_tail_pairs_per_expert=max_tail_pairs_per_expert,
        )
