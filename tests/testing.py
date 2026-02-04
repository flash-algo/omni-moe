from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import mean
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "Backend",
    "Implementation",
    "BenchmarkConfig",
    "BenchmarkResult",
    "get_impls",
    "run_benchmarks",
    "show_benchmarks",
]


class Backend(str, Enum):
    PYTORCH = "pytorch"
    TRITON = "triton"
    CUTILE = "cutile"

    def __str__(self) -> str:
        return self.value


def _torch_synchronize_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


_SYNCHRONIZERS = {
    Backend.PYTORCH: _torch_synchronize_if_available,
    Backend.TRITON: _torch_synchronize_if_available,
    Backend.CUTILE: _torch_synchronize_if_available,
}


def _flatten_tensors(output: Any) -> List[torch.Tensor]:
    """Collect all tensors from nested tuple/list or a single tensor."""
    if torch.is_tensor(output):
        return [output]
    if isinstance(output, (list, tuple)):
        tensors: List[torch.Tensor] = []
        for item in output:
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


def _get_primary_dtype(output: Any) -> Optional[torch.dtype]:
    tensors = _flatten_tensors(output)
    return tensors[0].dtype if tensors else None


def _assert_allclose(out: Any, ref: Any, *, rtol: float, atol: float) -> None:
    """Recursively assert closeness for tensors or sequences of tensors."""
    if torch.is_tensor(ref):
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        return

    if isinstance(ref, (list, tuple)):
        if not isinstance(out, (list, tuple)) or len(out) != len(ref):
            raise AssertionError("Output structure mismatch vs baseline")
        for o_item, r_item in zip(out, ref):
            _assert_allclose(o_item, r_item, rtol=rtol, atol=atol)
        return

    # Fallback to exact equality for non-tensor scalars
    if out != ref:
        raise AssertionError("Non-tensor outputs differ from baseline")


@dataclass
class Implementation:
    """Wrapper describing a concrete kernel implementation."""

    name: str
    fn: Callable[..., Any]
    backend: Backend
    description: Optional[str] = None
    synchronizer: Optional[Callable[[], None]] = None

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError(f"Implementation '{self.name}' must wrap a callable.")
        if self.synchronizer is None:
            self.synchronizer = _SYNCHRONIZERS.get(self.backend, lambda: None)

    def synchronize(self) -> None:
        assert self.synchronizer is not None
        self.synchronizer()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    warmup: int = 5
    repeat: int = 20

    def __post_init__(self) -> None:
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.repeat <= 0:
            raise ValueError("repeat must be greater than zero")


@dataclass
class BenchmarkResult:
    """Outcome of benchmarking a single implementation."""

    impl: Implementation
    timings_ms: Sequence[float]
    flops: float
    output: Any
    peak_mem_allocated_bytes: int
    peak_mem_reserved_bytes: int

    @property
    def mean_ms(self) -> float:
        return mean(self.timings_ms)

    @property
    def best_ms(self) -> float:
        return min(self.timings_ms)

    @property
    def worst_ms(self) -> float:
        return max(self.timings_ms)

    @property
    def tflops(self) -> float:
        seconds = self.mean_ms / 1e3
        return self.flops / seconds / 1e12

    @property
    def peak_mem(self) -> float:
        return self.peak_mem_allocated_bytes / (1024**2)

    def speedup_vs(self, baseline: "BenchmarkResult") -> float:
        return baseline.mean_ms / self.mean_ms


def _run_benchmark(
    impl: Implementation,
    factory: Callable[[Implementation], Tuple[Tuple[Any, ...], Dict[str, Any]]],
    *,
    flops: float,
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkResult:
    """
    Run benchmark for a single implementation.

    Args:
        impl: The implementation to benchmark.
        factory: Callable producing ``(args, kwargs)`` per invocation. It should
            return a tuple where the first element is a tuple of positional
            arguments and the second element is a dict of keyword arguments.
        flops: Total floating-point operations performed by one invocation.
        config: Benchmark configuration. Defaults to ``BenchmarkConfig()``.

    Returns:
        BenchmarkResult: Aggregated timings, last output, and FLOPs metadata.
    """
    if config is None:
        config = BenchmarkConfig()

    # Warmup (not timed)
    for _ in range(config.warmup):
        args, kwargs = factory(impl)
        impl.synchronize()
        _ = impl(*args, **kwargs)
        impl.synchronize()

    # Timed repeats
    timings_ms: List[float] = []
    # Memory tracking
    peak_allocs: List[int] = []
    peak_resvs: List[int] = []
    output: Any = None
    for _ in range(config.repeat):
        args, kwargs = factory(impl)
        impl.synchronize()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = perf_counter()
        output = impl(*args, **kwargs)
        impl.synchronize()
        end = perf_counter()
        timings_ms.append((end - start) * 1e3)

        peak_allocs.append(torch.cuda.max_memory_allocated())
        peak_resvs.append(torch.cuda.max_memory_reserved())

    return BenchmarkResult(
        impl=impl,
        timings_ms=timings_ms,
        flops=flops,
        output=output,
        peak_mem_allocated_bytes=max(peak_allocs),
        peak_mem_reserved_bytes=max(peak_resvs),
    )


def get_impls(
    *,
    pytorch_impl: Optional[Callable[..., Any]] = None,
    triton_impl: Optional[Callable[..., Any]] = None,
    cutile_impl: Optional[Callable[..., Any]] = None,
) -> List[Implementation]:
    """
    Construct a default list of implementations for benchmarking.

    The PyTorch implementation is always used as the numerical baseline.
    Backend-specific implementations are included only if a callable
    is provided.
    """

    impls: List[Implementation] = []

    if pytorch_impl is not None:
        impls.append(Implementation("pytorch", pytorch_impl, Backend.PYTORCH))

    if triton_impl is not None:
        impls.append(Implementation("triton", triton_impl, Backend.TRITON))

    if cutile_impl is not None:
        impls.append(Implementation("cutile", cutile_impl, Backend.CUTILE))

    return impls


def run_benchmarks(
    impls: Iterable[Implementation],
    factory: Callable[[Implementation], Tuple[Tuple[Any, ...], Dict[str, Any]]],
    *,
    flops: float,
    config: Optional[BenchmarkConfig] = None,
    validate: bool = False,
) -> List[BenchmarkResult]:
    """
    Run benchmarks for multiple implementations, validating outputs.

    The first implementation is treated as the numerical baseline.
    Every other implementation's single sample output is compared against the
    baseline output produced from its own fresh factory invocation.
    """
    impl_list = list(impls)
    if not impl_list:
        return []

    # Establish baseline output
    baseline_impl = impl_list[0]
    base_args, base_kwargs = factory(baseline_impl)
    baseline_impl.synchronize()
    baseline_output = baseline_impl(*base_args, **base_kwargs)
    baseline_impl.synchronize()

    dtype = _get_primary_dtype(baseline_output)
    if dtype == torch.float32:
        rtol = 1e-5
        atol = 1e-8
    elif dtype == torch.float16:
        rtol = 1e-2
        atol = 1e-2
    elif dtype == torch.bfloat16:
        rtol = 1e-3
        atol = 1e-3
    else:
        rtol = 1e-5
        atol = 1e-8

    results: List[BenchmarkResult] = []
    for impl in impl_list:
        args, kwargs = factory(impl)
        impl.synchronize()
        out = impl(*args, **kwargs)
        impl.synchronize()

        if validate:
            try:
                _assert_allclose(out, baseline_output, rtol=rtol, atol=atol)
            except Exception as e:
                print(
                    "\n[validate] output mismatch"
                    f"impl={impl.name} backend={impl.backend} rtol={rtol} atol={atol}\n{e}\n"
                )
        else:
            print(
                f"[validate] skipping output validation"
                f"impl={impl.name} backend={impl.backend}\n"
            )

        res = _run_benchmark(impl, factory, flops=flops, config=config)
        results.append(res)
    return results


def show_benchmarks(results: Sequence[BenchmarkResult]) -> None:
    """
    Pretty-print benchmark results.

    If multiple results are provided, the first is treated as the baseline for
    speedup computation.
    """
    if not results:
        print("No results to display.")
        return

    baseline = results[0]

    # Header
    headers = ("backend", "dtype", "speed", "speedup", "tflops", "peak_mem")
    print(
        f"\n{headers[0]:<10} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10} {headers[4]:>10} {headers[5]:>10}"
    )
    print("-" * 66)

    for r in results:
        speed = r.speedup_vs(baseline)
        tflops = r.tflops if r.flops > 0 else 0.0

        primary_dtype = _get_primary_dtype(r.output)
        dtype_str = str(primary_dtype).replace("torch.", "") if primary_dtype else "-"

        print(
            f"{str(r.impl.backend):<10} "
            f"{dtype_str:>10} "
            f"{r.mean_ms:>10.3f} "
            f"{speed:>10.2f} {tflops:>10.3f} "
            f"{r.peak_mem:>10.2f} "
        )
