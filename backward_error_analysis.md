# 反向传播数值误差分析

## 问题现象

从测试结果看到：
- **相对误差**: inf (说明 PyTorch 结果某些位置为 0，但 Triton 非 0)
- **绝对误差**: 最大 3.0 (远超允许的 0.001)
- **误差率**: 0.2%-0.8% 的元素不匹配

## 根本原因分析

### 1. Atomic Add 的数值精度问题

**问题代码** (omni_router.py, line 272-281):
```python
# Atomic accumulation to dscores_x and dscores_y
tl.atomic_add(
    dscores_x_ptr + ix * stride_dsxn,
    dscores,
    mask=mask,
)
tl.atomic_add(
    dscores_y_ptr + iy * stride_dsyn,
    dscores,
    mask=mask,
)
```

**问题点**:
1. **无效索引处理不当**: 当 `indices = -1` (无效索引) 时，`ix` 和 `iy` 计算会得到负数
   - `ix = -1 // num_expert_sqrt = -1`
   - `iy = -1 - (-1) * num_expert_sqrt = -1`
   - 负索引会导致访问错误的内存位置

2. **Mask 无效**: 虽然有 mask，但 atomic_add 在 mask=False 时仍然会执行地址计算
   - 负索引 + stride 可能导致越界或访问到其他位置
   - 累积到错误位置导致数值错误

3. **Float32 转换时机**: dscores 转换为 float32，但 atomic_add 可能在硬件层面仍有精度损失

### 2. 为什么会出现 inf 相对误差

**场景**:
1. PyTorch 实现中，某个专家从未被选中，梯度为 0
2. Triton 实现中，由于索引错误，该位置被错误累积了梯度
3. 相对误差 = |triton - pytorch| / |pytorch| = |非零值| / 0 = inf

### 3. 为什么绝对误差达到 3.0

**原因**:
- 多个 token 的梯度被错误累积到同一个位置
- bfloat16 精度下，多次 atomic_add 累积误差
- 索引错误导致某些位置累积了本不该有的梯度

## 解决方案

### 方案 1: 修复无效索引处理 ✅ (推荐)

在 atomic_add 前显式过滤无效索引:

```python
# 创建有效索引的 mask
valid_mask = (indices >= 0) & mask

# 只对有效索引执行 atomic_add
tl.atomic_add(
    dscores_x_ptr + ix * stride_dsxn,
    dscores,
    mask=valid_mask,
)
```

### 方案 2: 使用条件 where 避免负索引

```python
# 将无效索引替换为 0 (安全位置)
ix_safe = tl.where(indices >= 0, ix, 0)
iy_safe = tl.where(indices >= 0, iy, 0)
dscores_safe = tl.where(indices >= 0, dscores, 0.0)

tl.atomic_add(
    dscores_x_ptr + ix_safe * stride_dsxn,
    dscores_safe,
    mask=mask,
)
```

### 方案 3: 增加数值稳定性检查

```python
# 在 atomic_add 前进行边界检查
ix_valid = (ix >= 0) & (ix < num_expert_sqrt)
iy_valid = (iy >= 0) & (iy < num_expert_sqrt)
valid_mask = mask & ix_valid & iy_valid

tl.atomic_add(
    dscores_x_ptr + ix * stride_dsxn,
    dscores,
    mask=valid_mask,
)
```

## 推荐修复

综合考虑性能和正确性，推荐**方案 1 + 方案 3**的组合:

```python
# 1. 过滤无效索引 (indices = -1)
valid_indices = indices >= 0

# 2. 计算专家坐标
ix = indices // num_expert_sqrt
iy = indices - ix * num_expert_sqrt

# 3. 确保坐标在有效范围内
ix_in_range = (ix >= 0) & (ix < num_expert_sqrt)
iy_in_range = (iy >= 0) & (iy < num_expert_sqrt)

# 4. 组合所有条件
valid_mask_x = mask & valid_indices & ix_in_range
valid_mask_y = mask & valid_indices & iy_in_range

# 5. 安全的 atomic_add
tl.atomic_add(
    dscores_x_ptr + ix * stride_dsxn,
    dscores,
    mask=valid_mask_x,
)
tl.atomic_add(
    dscores_y_ptr + iy * stride_dsyn,
    dscores,
    mask=valid_mask_y,
)
```

## 预期效果

修复后应该:
- ✅ 消除 inf 相对误差
- ✅ 绝对误差降低到 < 0.01 (bfloat16 精度范围内)
- ✅ 误差元素比例降低到 < 0.01%
