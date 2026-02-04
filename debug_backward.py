#!/usr/bin/env python3
"""
Deep debugging script for backward pass issues.
"""

import torch
import numpy as np
from omni_moe.triton.omni_router import triton_omni_router_func

def pytorch_reference_backward(
    router_logits_x: torch.Tensor,
    router_logits_y: torch.Tensor,
    num_expert_sqrt: int,
    num_experts_per_token: int,
):
    """Reference PyTorch implementation for comparison."""
    # Forward
    scores_x, indices_x = router_logits_x.topk(num_expert_sqrt, dim=-1)
    scores_y, indices_y = router_logits_y.topk(num_expert_sqrt, dim=-1)
    all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
    all_indices = indices_x.unsqueeze(-1) * num_expert_sqrt + indices_y.unsqueeze(-2)
    all_scores = all_scores.view(*all_scores.shape[:-2], -1)
    all_indices = all_indices.view(*all_indices.shape[:-2], -1)
    scores, pos_idx = all_scores.topk(num_experts_per_token, dim=-1)
    indices = all_indices.gather(-1, pos_idx)
    
    return scores, indices


def debug_backward():
    """Debug backward pass in detail."""
    print("="*80)
    print("Deep Debugging Backward Pass")
    print("="*80)
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # Small test case for detailed inspection
    num_tokens = 4
    num_expert_sqrt = 8
    num_experts_per_token = 4
    
    print(f"\nTest config: tokens={num_tokens}, expert_sqrt={num_expert_sqrt}, top_k={num_experts_per_token}")
    
    # Create deterministic test data
    torch.manual_seed(42)
    logits_x_pt = torch.randn(num_tokens, num_expert_sqrt, device=device, dtype=dtype, requires_grad=True)
    logits_y_pt = torch.randn(num_tokens, num_expert_sqrt, device=device, dtype=dtype, requires_grad=True)
    
    logits_x_tr = logits_x_pt.clone().detach().requires_grad_(True)
    logits_y_tr = logits_y_pt.clone().detach().requires_grad_(True)
    
    # PyTorch reference
    print("\n" + "-"*80)
    print("PyTorch Reference")
    print("-"*80)
    scores_pt, indices_pt = pytorch_reference_backward(
        logits_x_pt, logits_y_pt, num_expert_sqrt, num_experts_per_token
    )
    loss_pt = scores_pt.sum()
    loss_pt.backward()
    
    print(f"Scores shape: {scores_pt.shape}")
    print(f"Indices shape: {indices_pt.shape}")
    print(f"Indices:\n{indices_pt}")
    
    print(f"\nGrad X shape: {logits_x_pt.grad.shape}")
    print(f"Grad X nonzero: {(logits_x_pt.grad.abs() > 1e-6).sum().item()}/{logits_x_pt.grad.numel()}")
    print(f"Grad X range: [{logits_x_pt.grad.min().item():.4f}, {logits_x_pt.grad.max().item():.4f}]")
    print(f"Grad X:\n{logits_x_pt.grad}")
    
    print(f"\nGrad Y shape: {logits_y_pt.grad.shape}")
    print(f"Grad Y nonzero: {(logits_y_pt.grad.abs() > 1e-6).sum().item()}/{logits_y_pt.grad.numel()}")
    print(f"Grad Y range: [{logits_y_pt.grad.min().item():.4f}, {logits_y_pt.grad.max().item():.4f}]")
    print(f"Grad Y:\n{logits_y_pt.grad}")
    
    # Triton implementation
    print("\n" + "-"*80)
    print("Triton Implementation")
    print("-"*80)
    scores_tr, indices_tr = triton_omni_router_func(
        logits_x_tr, logits_y_tr, num_expert_sqrt, num_experts_per_token
    )
    loss_tr = scores_tr.sum()
    loss_tr.backward()
    
    print(f"Scores shape: {scores_tr.shape}")
    print(f"Indices shape: {indices_tr.shape}")
    print(f"Indices:\n{indices_tr}")
    
    print(f"\nGrad X shape: {logits_x_tr.grad.shape}")
    print(f"Grad X nonzero: {(logits_x_tr.grad.abs() > 1e-6).sum().item()}/{logits_x_tr.grad.numel()}")
    print(f"Grad X range: [{logits_x_tr.grad.min().item():.4f}, {logits_x_tr.grad.max().item():.4f}]")
    print(f"Grad X:\n{logits_x_tr.grad}")
    
    print(f"\nGrad Y shape: {logits_y_tr.grad.shape}")
    print(f"Grad Y nonzero: {(logits_y_tr.grad.abs() > 1e-6).sum().item()}/{logits_y_tr.grad.numel()}")
    print(f"Grad Y range: [{logits_y_tr.grad.min().item():.4f}, {logits_y_tr.grad.max().item():.4f}]")
    print(f"Grad Y:\n{logits_y_tr.grad}")
    
    # Compare
    print("\n" + "-"*80)
    print("Comparison")
    print("-"*80)
    
    # Forward comparison
    scores_match = torch.allclose(scores_pt, scores_tr, rtol=1e-3, atol=1e-3)
    indices_match = torch.equal(indices_pt, indices_tr)
    print(f"Forward scores match: {scores_match}")
    print(f"Forward indices match: {indices_match}")
    
    if not scores_match:
        diff = (scores_pt - scores_tr).abs()
        print(f"  Max score diff: {diff.max().item():.6f}")
        print(f"  Score diff:\n{diff}")
    
    if not indices_match:
        print(f"  Indices diff:\n{indices_pt - indices_tr}")
    
    # Backward comparison
    grad_x_match = torch.allclose(logits_x_pt.grad, logits_x_tr.grad, rtol=1e-2, atol=1e-2)
    grad_y_match = torch.allclose(logits_y_pt.grad, logits_y_tr.grad, rtol=1e-2, atol=1e-2)
    
    print(f"\nBackward grad_x match: {grad_x_match}")
    print(f"Backward grad_y match: {grad_y_match}")
    
    if not grad_x_match:
        diff = (logits_x_pt.grad - logits_x_tr.grad).abs()
        print(f"  Max grad_x diff: {diff.max().item():.6f}")
        print(f"  Grad X diff:\n{diff}")
        
        # Find positions with large differences
        large_diff = diff > 0.1
        if large_diff.any():
            print(f"  Positions with large diff (>0.1):")
            for i in range(num_tokens):
                for j in range(num_expert_sqrt):
                    if large_diff[i, j]:
                        print(f"    [{i},{j}]: PT={logits_x_pt.grad[i,j].item():.4f}, TR={logits_x_tr.grad[i,j].item():.4f}, diff={diff[i,j].item():.4f}")
    
    if not grad_y_match:
        diff = (logits_y_pt.grad - logits_y_tr.grad).abs()
        print(f"  Max grad_y diff: {diff.max().item():.6f}")
        print(f"  Grad Y diff:\n{diff}")
        
        # Find positions with large differences
        large_diff = diff > 0.1
        if large_diff.any():
            print(f"  Positions with large diff (>0.1):")
            for i in range(num_tokens):
                for j in range(num_expert_sqrt):
                    if large_diff[i, j]:
                        print(f"    [{i},{j}]: PT={logits_y_pt.grad[i,j].item():.4f}, TR={logits_y_tr.grad[i,j].item():.4f}, diff={diff[i,j].item():.4f}")
    
    # Analyze gradient distribution
    print("\n" + "-"*80)
    print("Gradient Distribution Analysis")
    print("-"*80)
    
    # Check for zero gradients in PyTorch that are non-zero in Triton
    pt_zero_x = (logits_x_pt.grad.abs() < 1e-6)
    tr_nonzero_x = (logits_x_tr.grad.abs() > 1e-6)
    spurious_x = pt_zero_x & tr_nonzero_x
    
    pt_zero_y = (logits_y_pt.grad.abs() < 1e-6)
    tr_nonzero_y = (logits_y_tr.grad.abs() > 1e-6)
    spurious_y = pt_zero_y & tr_nonzero_y
    
    print(f"Spurious gradients in X (PT=0, TR≠0): {spurious_x.sum().item()}")
    if spurious_x.any():
        print("  Positions:")
        for i in range(num_tokens):
            for j in range(num_expert_sqrt):
                if spurious_x[i, j]:
                    print(f"    [{i},{j}]: TR={logits_x_tr.grad[i,j].item():.4f}")
    
    print(f"Spurious gradients in Y (PT=0, TR≠0): {spurious_y.sum().item()}")
    if spurious_y.any():
        print("  Positions:")
        for i in range(num_tokens):
            for j in range(num_expert_sqrt):
                if spurious_y[i, j]:
                    print(f"    [{i},{j}]: TR={logits_y_tr.grad[i,j].item():.4f}")
    
    # Check which experts were selected
    print("\n" + "-"*80)
    print("Expert Selection Analysis")
    print("-"*80)
    
    for token_idx in range(num_tokens):
        print(f"\nToken {token_idx}:")
        selected_experts = indices_tr[token_idx].cpu().numpy()
        print(f"  Selected experts (flat): {selected_experts}")
        
        # Convert flat indices to (ix, iy)
        for expert_flat in selected_experts:
            ix = expert_flat // num_expert_sqrt
            iy = expert_flat % num_expert_sqrt
            print(f"    Expert {expert_flat}: ix={ix}, iy={iy}")


if __name__ == "__main__":
    debug_backward()
