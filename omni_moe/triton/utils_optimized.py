"""
Optimized utility functions with extended autotune configurations.
"""
import triton
from omni_moe.triton.utils import *


def get_router_fwd_autotune_configs_optimized():
    """
    Get optimized autotuning configurations for the router forward kernel.
    
    Improvements:
    - More TILE_M options for better granularity
    - More NUM_WARPS options for different parallelism levels
    - More NUM_STAGES options for better instruction-level parallelism
    
    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    configs = []
    # Expanded options for better coverage
    BLOCK_M_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    NUM_WARPS_OPTIONS = [2, 4, 8]
    NUM_STAGES_OPTION = [1, 2, 3]

    for bm in BLOCK_M_OPTIONS:
        for nw in NUM_WARPS_OPTIONS:
            for ns in NUM_STAGES_OPTION:
                configs.append(
                    triton.Config(
                        {
                            "TILE_M": bm,
                        },
                        num_warps=nw,
                        num_stages=ns,
                    )
                )
    return configs


def get_router_fwd_split_experts_autotune_configs_optimized():
    """
    Get optimized autotuning configurations for the router forward split experts kernel.
    
    Improvements:
    - More TILE_M options (not fixed to 1)
    - Extended TILE_N options
    - More NUM_WARPS and NUM_STAGES options
    
    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    configs = []
    # Expanded options
    BLOCK_M_OPTIONS = [1, 2, 4]  # Allow some batching
    BLOCK_N_OPTIONS = [512, 1024, 2048, 4096, 8192, 16384]
    NUM_WARPS_OPTIONS = [4, 8]
    NUM_STAGES_OPTION = [1, 2]

    for bm in BLOCK_M_OPTIONS:
        for bn in BLOCK_N_OPTIONS:
            for nw in NUM_WARPS_OPTIONS:
                for ns in NUM_STAGES_OPTION:
                    configs.append(
                        triton.Config(
                            {
                                "TILE_M": bm,
                                "TILE_N": bn,
                            },
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


def get_router_bwd_autotune_configs_optimized():
    """
    Get optimized autotuning configurations for the router backward kernel.
    
    Improvements:
    - More granular TILE_M and TILE_K options
    - Extended NUM_WARPS options
    - More NUM_STAGES options
    
    :return configs: List of triton.Config objects
    """
    device = get_device()
    arch = get_arch(device)

    if arch == "N/A":
        raise ValueError("Your device architecture is not supported for now.")

    configs = []
    # Expanded options
    BLOCK_M_OPTIONS = [32, 64, 128, 256]
    BLOCK_K_OPTIONS = [8, 16, 32, 64]
    NUM_WARPS_OPTIONS = [2, 4, 8]
    NUM_STAGES_OPTION = [1, 2, 3]

    for bm in BLOCK_M_OPTIONS:
        for bk in BLOCK_K_OPTIONS:
            for nw in NUM_WARPS_OPTIONS:
                for ns in NUM_STAGES_OPTION:
                    configs.append(
                        triton.Config(
                            {
                                "TILE_M": bm,
                                "TILE_K": bk,
                            },
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


def get_arch_specific_configs(arch: str, kernel_type: str):
    """
    Get architecture-specific optimized configurations.
    
    :param arch: Architecture string (e.g., "cuda:sm80", "cuda:sm86")
    :param kernel_type: Type of kernel ("fwd", "fwd_split", "bwd")
    
    :return configs: List of triton.Config objects optimized for the architecture
    """
    configs = []
    
    if "sm_80" in arch or "sm_86" in arch:  # A100, RTX 30xx
        if kernel_type == "fwd":
            # A100 has high memory bandwidth, prefer larger tiles
            BLOCK_M_OPTIONS = [64, 128, 256]
            NUM_WARPS_OPTIONS = [4, 8]
            NUM_STAGES_OPTION = [2, 3]
        elif kernel_type == "fwd_split":
            BLOCK_M_OPTIONS = [1, 2]
            BLOCK_N_OPTIONS = [4096, 8192, 16384]
            NUM_WARPS_OPTIONS = [8]
            NUM_STAGES_OPTION = [2]
        else:  # bwd
            BLOCK_M_OPTIONS = [128, 256]
            BLOCK_K_OPTIONS = [32, 64]
            NUM_WARPS_OPTIONS = [4, 8]
            NUM_STAGES_OPTION = [2, 3]
    
    elif "sm_89" in arch or "sm_90" in arch:  # H100, L40S
        if kernel_type == "fwd":
            # H100 has even higher bandwidth and more SMs
            BLOCK_M_OPTIONS = [128, 256]
            NUM_WARPS_OPTIONS = [8, 16]
            NUM_STAGES_OPTION = [3, 4]
        elif kernel_type == "fwd_split":
            BLOCK_M_OPTIONS = [2, 4]
            BLOCK_N_OPTIONS = [8192, 16384]
            NUM_WARPS_OPTIONS = [8, 16]
            NUM_STAGES_OPTION = [2, 3]
        else:  # bwd
            BLOCK_M_OPTIONS = [256]
            BLOCK_K_OPTIONS = [64]
            NUM_WARPS_OPTIONS = [8, 16]
            NUM_STAGES_OPTION = [3, 4]
    
    else:  # Fallback to general configs
        return None
    
    # Build configs based on kernel type
    if kernel_type == "fwd":
        for bm in BLOCK_M_OPTIONS:
            for nw in NUM_WARPS_OPTIONS:
                for ns in NUM_STAGES_OPTION:
                    configs.append(
                        triton.Config(
                            {"TILE_M": bm},
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    elif kernel_type == "fwd_split":
        for bm in BLOCK_M_OPTIONS:
            for bn in BLOCK_N_OPTIONS:
                for nw in NUM_WARPS_OPTIONS:
                    for ns in NUM_STAGES_OPTION:
                        configs.append(
                            triton.Config(
                                {"TILE_M": bm, "TILE_N": bn},
                                num_warps=nw,
                                num_stages=ns,
                            )
                        )
    else:  # bwd
        for bm in BLOCK_M_OPTIONS:
            for bk in BLOCK_K_OPTIONS:
                for nw in NUM_WARPS_OPTIONS:
                    for ns in NUM_STAGES_OPTION:
                        configs.append(
                            triton.Config(
                                {"TILE_M": bm, "TILE_K": bk},
                                num_warps=nw,
                                num_stages=ns,
                            )
                        )
    
    return configs
