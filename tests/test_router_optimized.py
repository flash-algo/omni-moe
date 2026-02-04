import pytest
import torch

import testing
from omni_moe.triton.omni_router import triton_omni_router_func
from omni_moe.triton.omni_router_optimized import triton_omni_router_func_optimized


def pytorch_router_forward(
    router_logits_x: torch.Tensor,
    router_logits_y: torch.Tensor,
    num_keys: int,
    top_k: int,
):
    scores_x, indices_x = router_logits_x.topk(num_keys, dim=-1)
    scores_y, indices_y = router_logits_y.topk(num_keys, dim=-1)
    all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
    all_indices = indices_x.unsqueeze(-1) * num_keys + indices_y.unsqueeze(-2)
    all_scores = all_scores.view(*all_scores.shape[:-2], -1)
    all_indices = all_indices.view(*all_indices.shape[:-2], -1)
    scores, pos_idx = all_scores.topk(top_k, dim=-1)
    _ = all_indices.gather(-1, pos_idx)
    return scores


def triton_router_forward_original(
    router_logits_x: torch.Tensor,
    router_logits_y: torch.Tensor,
    num_keys: int,
    top_k: int,
):
    scores, _ = triton_omni_router_func(
        router_logits_x, router_logits_y, num_keys, top_k
    )
    return scores


def triton_router_forward_optimized(
    router_logits_x: torch.Tensor,
    router_logits_y: torch.Tensor,
    num_keys: int,
    top_k: int,
):
    scores, _ = triton_omni_router_func_optimized(
        router_logits_x, router_logits_y, num_keys, top_k
    )
    return scores


def pytorch_router_backward(
    loss: torch.Tensor,
    logits_x: torch.Tensor,
    logits_y: torch.Tensor,
):
    loss.backward()
    return logits_x.grad, logits_y.grad


def triton_router_backward_original(
    loss: torch.Tensor,
    logits_x: torch.Tensor,
    logits_y: torch.Tensor,
):
    loss.backward()
    return logits_x.grad, logits_y.grad


def triton_router_backward_optimized(
    loss: torch.Tensor,
    logits_x: torch.Tensor,
    logits_y: torch.Tensor,
):
    loss.backward()
    return logits_x.grad, logits_y.grad


def make_forward_factory(
    num_tokens: int, num_keys: int, top_k: int, device: torch.device, dtype: torch.dtype
):
    gen = torch.Generator(device=device).manual_seed(0)
    base_x = torch.randn(
        num_tokens, num_keys, device=device, dtype=dtype, generator=gen
    )
    base_y = torch.randn(
        num_tokens, num_keys, device=device, dtype=dtype, generator=gen
    )

    def _factory(_impl: testing.Implementation):
        logits_x = base_x.clone()
        logits_y = base_y.clone()
        return (logits_x, logits_y, num_keys, top_k), {}

    return _factory


def make_backward_factory(
    num_tokens: int, num_keys: int, top_k: int, device: torch.device, dtype: torch.dtype, use_optimized: bool = False
):
    gen = torch.Generator(device=device).manual_seed(0)
    base_x = torch.randn(
        num_tokens, num_keys, device=device, dtype=dtype, generator=gen
    )
    base_y = torch.randn(
        num_tokens, num_keys, device=device, dtype=dtype, generator=gen
    )

    def _factory(impl: testing.Implementation):
        logits_x = base_x.clone().detach().requires_grad_(True)
        logits_y = base_y.clone().detach().requires_grad_(True)
        if impl.backend == testing.Backend.PYTORCH:
            loss = pytorch_router_forward(logits_x, logits_y, num_keys, top_k).sum()
        elif impl.backend == testing.Backend.TRITON:
            if use_optimized:
                loss = triton_router_forward_optimized(logits_x, logits_y, num_keys, top_k).sum()
            else:
                loss = triton_router_forward_original(logits_x, logits_y, num_keys, top_k).sum()
        return (loss, logits_x, logits_y), {}

    return _factory


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        (1024, 64, 16),
        (1024, 64, 32),
        (1024, 128, 16),
        (1024, 128, 32),
        (1024, 256, 16),
        (1024, 256, 32),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_router_forward_comparison(
    dtype: torch.dtype, case: tuple[int, int, int]
) -> None:
    """Compare original vs optimized forward implementation."""
    num_tokens, num_keys, top_k = case
    device = torch.device("cuda")

    num_expert = num_keys * num_keys
    print(
        f"\n[omni router forward comparison] num_tokens={num_tokens}, "
        f"num_expert={num_expert}, select_expert={top_k}"
    )

    # Test original implementation
    impls_original = testing.get_impls(
        pytorch_impl=pytorch_router_forward,
        triton_impl=triton_router_forward_original,
    )
    flops = 4.0 * num_tokens * num_keys * num_keys
    config = testing.BenchmarkConfig(warmup=5, repeat=10)
    
    print("\n--- Original Implementation ---")
    results_original = testing.run_benchmarks(
        impls_original,
        make_forward_factory(num_tokens, num_keys, top_k, device, dtype),
        flops=flops,
        config=config,
        validate=True,
    )
    testing.show_benchmarks(results_original)

    # Test optimized implementation
    impls_optimized = testing.get_impls(
        pytorch_impl=pytorch_router_forward,
        triton_impl=triton_router_forward_optimized,
    )
    
    print("\n--- Optimized Implementation ---")
    results_optimized = testing.run_benchmarks(
        impls_optimized,
        make_forward_factory(num_tokens, num_keys, top_k, device, dtype),
        flops=flops,
        config=config,
        validate=True,
    )
    testing.show_benchmarks(results_optimized)

    # Calculate speedup
    original_time = results_original[1].mean_time_ms  # Triton result
    optimized_time = results_optimized[1].mean_time_ms  # Triton result
    speedup = original_time / optimized_time
    print(f"\n🚀 Speedup: {speedup:.2f}x ({original_time:.3f}ms -> {optimized_time:.3f}ms)")


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        (1024, 64, 16),
        (1024, 64, 32),
        (1024, 128, 16),
        (1024, 128, 32),
        (1024, 256, 16),
        (1024, 256, 32),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_router_backward_comparison(
    dtype: torch.dtype, case: tuple[int, int, int]
) -> None:
    """Compare original vs optimized backward implementation."""
    num_tokens, num_keys, top_k = case
    device = torch.device("cuda")

    num_expert = num_keys * num_keys
    print(
        f"\n[omni router backward comparison] num_tokens={num_tokens}, "
        f"num_expert={num_expert}, select_expert={top_k}"
    )

    # Test original implementation
    impls_original = testing.get_impls(
        pytorch_impl=pytorch_router_backward,
        triton_impl=triton_router_backward_original,
    )
    flops = 2.0 * 4.0 * num_tokens * num_keys * num_keys
    config = testing.BenchmarkConfig(warmup=5, repeat=10)
    
    print("\n--- Original Implementation ---")
    results_original = testing.run_benchmarks(
        impls_original,
        make_backward_factory(num_tokens, num_keys, top_k, device, dtype, use_optimized=False),
        flops=flops,
        config=config,
        validate=True,
    )
    testing.show_benchmarks(results_original)

    # Test optimized implementation
    impls_optimized = testing.get_impls(
        pytorch_impl=pytorch_router_backward,
        triton_impl=triton_router_backward_optimized,
    )
    
    print("\n--- Optimized Implementation ---")
    results_optimized = testing.run_benchmarks(
        impls_optimized,
        make_backward_factory(num_tokens, num_keys, top_k, device, dtype, use_optimized=True),
        flops=flops,
        config=config,
        validate=True,
    )
    testing.show_benchmarks(results_optimized)

    # Calculate speedup
    original_time = results_original[1].mean_time_ms  # Triton result
    optimized_time = results_optimized[1].mean_time_ms  # Triton result
    speedup = original_time / optimized_time
    print(f"\n🚀 Speedup: {speedup:.2f}x ({original_time:.3f}ms -> {optimized_time:.3f}ms)")


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16],
)
@pytest.mark.parametrize(
    "case",
    [
        (1024, 64, 16),
        (1024, 128, 32),
        (1024, 256, 32),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_router_correctness(
    dtype: torch.dtype, case: tuple[int, int, int]
) -> None:
    """Verify that optimized implementation produces correct results."""
    num_tokens, num_keys, top_k = case
    device = torch.device("cuda")

    print(
        f"\n[correctness test] num_tokens={num_tokens}, "
        f"num_keys={num_keys}, top_k={top_k}"
    )

    # Generate test data
    gen = torch.Generator(device=device).manual_seed(42)
    logits_x = torch.randn(num_tokens, num_keys, device=device, dtype=dtype, generator=gen)
    logits_y = torch.randn(num_tokens, num_keys, device=device, dtype=dtype, generator=gen)

    # Run original implementation
    scores_orig, indices_orig = triton_omni_router_func(
        logits_x.clone(), logits_y.clone(), num_keys, top_k
    )

    # Run optimized implementation
    scores_opt, indices_opt = triton_omni_router_func_optimized(
        logits_x.clone(), logits_y.clone(), num_keys, top_k
    )

    # Compare results
    scores_match = torch.allclose(scores_orig, scores_opt, rtol=1e-3, atol=1e-3)
    indices_match = torch.equal(indices_orig, indices_opt)

    print(f"Scores match: {scores_match}")
    print(f"Indices match: {indices_match}")
    
    if not scores_match:
        max_diff = (scores_orig - scores_opt).abs().max().item()
        print(f"Max score difference: {max_diff}")
    
    assert scores_match, "Scores do not match between original and optimized implementations"
    assert indices_match, "Indices do not match between original and optimized implementations"
    
    print("✅ Correctness test passed!")
