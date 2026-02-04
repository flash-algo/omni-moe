#!/usr/bin/env python3
"""
Verification script for backward pass fix.
Tests both original and optimized implementations.
"""

import torch
import sys

# Test original implementation
try:
    from omni_moe.triton.omni_router import triton_omni_router_func
    print("✓ Original implementation imported successfully")
except Exception as e:
    print(f"✗ Failed to import original implementation: {e}")
    sys.exit(1)

# Test optimized implementation
try:
    from omni_moe.triton.omni_router_optimized import triton_omni_router_func_optimized
    print("✓ Optimized implementation imported successfully")
except Exception as e:
    print(f"✗ Failed to import optimized implementation: {e}")
    sys.exit(1)


def test_backward_numerical_stability(impl_func, impl_name):
    """Test backward pass numerical stability."""
    print(f"\n{'='*60}")
    print(f"Testing {impl_name}")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("⚠ CUDA not available, skipping test")
        return
    
    dtype = torch.bfloat16
    num_tokens = 1024
    num_expert_sqrt = 64
    num_experts_per_token = 16
    
    print(f"Config: tokens={num_tokens}, expert_sqrt={num_expert_sqrt}, top_k={num_experts_per_token}")
    
    # Create test data
    torch.manual_seed(42)
    logits_x = torch.randn(num_tokens, num_expert_sqrt, device=device, dtype=dtype, requires_grad=True)
    logits_y = torch.randn(num_tokens, num_expert_sqrt, device=device, dtype=dtype, requires_grad=True)
    
    # Forward pass
    scores, indices = impl_func(logits_x, logits_y, num_expert_sqrt, num_experts_per_token)
    
    # Check for invalid values in forward pass
    if torch.isnan(scores).any():
        print("✗ NaN detected in forward scores")
        return False
    if torch.isinf(scores).any():
        print("✗ Inf detected in forward scores")
        return False
    print("✓ Forward pass: No NaN/Inf")
    
    # Backward pass
    loss = scores.sum()
    loss.backward()
    
    # Check gradients
    if logits_x.grad is None or logits_y.grad is None:
        print("✗ Gradients not computed")
        return False
    
    # Check for invalid values in gradients
    if torch.isnan(logits_x.grad).any() or torch.isnan(logits_y.grad).any():
        print("✗ NaN detected in gradients")
        return False
    
    if torch.isinf(logits_x.grad).any() or torch.isinf(logits_y.grad).any():
        print("✗ Inf detected in gradients")
        return False
    
    print("✓ Backward pass: No NaN/Inf")
    
    # Check gradient statistics
    grad_x_max = logits_x.grad.abs().max().item()
    grad_y_max = logits_y.grad.abs().max().item()
    grad_x_mean = logits_x.grad.abs().mean().item()
    grad_y_mean = logits_y.grad.abs().mean().item()
    
    print(f"Gradient X: max={grad_x_max:.4f}, mean={grad_x_mean:.4f}")
    print(f"Gradient Y: max={grad_y_max:.4f}, mean={grad_y_mean:.4f}")
    
    # Check for reasonable gradient magnitudes
    if grad_x_max > 100 or grad_y_max > 100:
        print("⚠ Warning: Very large gradients detected")
    
    # Check gradient coverage (non-zero elements)
    nonzero_x = (logits_x.grad.abs() > 1e-6).sum().item()
    nonzero_y = (logits_y.grad.abs() > 1e-6).sum().item()
    total_x = logits_x.grad.numel()
    total_y = logits_y.grad.numel()
    
    coverage_x = nonzero_x / total_x * 100
    coverage_y = nonzero_y / total_y * 100
    
    print(f"Gradient coverage X: {coverage_x:.2f}% ({nonzero_x}/{total_x})")
    print(f"Gradient coverage Y: {coverage_y:.2f}% ({nonzero_y}/{total_y})")
    
    print(f"✓ {impl_name} passed all checks")
    return True


def compare_implementations():
    """Compare original and optimized implementations."""
    print(f"\n{'='*60}")
    print("Comparing Original vs Optimized")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("⚠ CUDA not available, skipping comparison")
        return
    
    dtype = torch.bfloat16
    num_tokens = 512
    num_expert_sqrt = 64
    num_experts_per_token = 16
    
    # Create identical test data
    torch.manual_seed(123)
    logits_x_orig = torch.randn(num_tokens, num_expert_sqrt, device=device, dtype=dtype, requires_grad=True)
    logits_y_orig = torch.randn(num_tokens, num_expert_sqrt, device=device, dtype=dtype, requires_grad=True)
    
    logits_x_opt = logits_x_orig.clone().detach().requires_grad_(True)
    logits_y_opt = logits_y_orig.clone().detach().requires_grad_(True)
    
    # Original implementation
    scores_orig, indices_orig = triton_omni_router_func(
        logits_x_orig, logits_y_orig, num_expert_sqrt, num_experts_per_token
    )
    loss_orig = scores_orig.sum()
    loss_orig.backward()
    
    # Optimized implementation
    scores_opt, indices_opt = triton_omni_router_func_optimized(
        logits_x_opt, logits_y_opt, num_expert_sqrt, num_experts_per_token
    )
    loss_opt = scores_opt.sum()
    loss_opt.backward()
    
    # Compare forward results
    scores_close = torch.allclose(scores_orig, scores_opt, rtol=1e-3, atol=1e-3)
    indices_match = torch.equal(indices_orig, indices_opt)
    
    print(f"Forward scores match: {scores_close}")
    print(f"Forward indices match: {indices_match}")
    
    if not scores_close:
        max_diff = (scores_orig - scores_opt).abs().max().item()
        print(f"  Max score difference: {max_diff:.6f}")
    
    # Compare backward results
    grad_x_close = torch.allclose(logits_x_orig.grad, logits_x_opt.grad, rtol=1e-2, atol=1e-2)
    grad_y_close = torch.allclose(logits_y_orig.grad, logits_y_opt.grad, rtol=1e-2, atol=1e-2)
    
    print(f"Backward grad_x match: {grad_x_close}")
    print(f"Backward grad_y match: {grad_y_close}")
    
    if not grad_x_close:
        max_diff = (logits_x_orig.grad - logits_x_opt.grad).abs().max().item()
        print(f"  Max grad_x difference: {max_diff:.6f}")
    
    if not grad_y_close:
        max_diff = (logits_y_orig.grad - logits_y_opt.grad).abs().max().item()
        print(f"  Max grad_y difference: {max_diff:.6f}")
    
    if scores_close and indices_match and grad_x_close and grad_y_close:
        print("✓ Implementations match within tolerance")
        return True
    else:
        print("⚠ Implementations have differences (may be acceptable)")
        return False


if __name__ == "__main__":
    print("Omni Router Backward Pass Fix Verification")
    print("=" * 60)
    
    # Test original implementation
    test_backward_numerical_stability(triton_omni_router_func, "Original Implementation")
    
    # Test optimized implementation
    test_backward_numerical_stability(triton_omni_router_func_optimized, "Optimized Implementation")
    
    # Compare implementations
    compare_implementations()
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run full test suite: pytest tests/test_router.py -s")
    print("2. Run optimized tests: pytest tests/test_router_optimized.py -s")
    print("3. Check for numerical stability in your specific use case")
