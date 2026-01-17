# Memory Investigation Report: Cognitive Flow

**Date:** 2026-01-16
**Investigator:** Claude (Opus 4.5)
**App Version:** 1.9.1

---

## Summary

Investigated whether Cognitive Flow (pythonw.exe running Parakeet/ONNX backend) is causing system memory pressure. **Conclusion: App is not the primary cause of physical RAM pressure, but has excessive virtual memory commit that should be optimized.**

---

## System State Observed

| Metric | Value |
|--------|-------|
| Total RAM | 15.7 GB |
| In Use | 14.9 GB (95%) |
| Available | 790 MB |
| Committed | 53.7 / 63.7 GB |
| Non-paged Pool | 2.2 GB (elevated) |
| Paged Pool | 1.0 GB |

---

## Cognitive Flow Process Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Working Set | ~60-100 MB | Normal - minimal physical RAM |
| Commit Size | 5.2 GB | Excessive - should be ~500 MB |
| GPU Memory | 3.3 GB | Expected for Parakeet model |

---

## Key Findings

### 1. Physical RAM Usage (Working Set): NOT GUILTY

The app's working set is only 60-100 MB, meaning it's using minimal physical RAM. The 14.9 GB physical RAM pressure is coming from other sources:

- Google Chrome (38 tabs): 1,476 MB
- Claude Code instances (8x): ~1,700 MB combined
- Various system processes

### 2. Virtual Memory Commit: EXCESSIVE

The app reserves 5.2 GB of virtual address space (commit) despite only using ~100 MB physically. This is caused by:

- **ONNX Runtime memory arenas**: Pre-allocates large memory pools for potential use
- **CUDA driver context**: Reserves address space for GPU operations
- **Default arena_extend_strategy**: Uses `kNextPowerOfTwo` which over-allocates

While commit != physical RAM, excessive commit:
- Counts against system commit limit (page file)
- Can cause issues when system is under memory pressure
- Reserves address space other apps could use

### 3. Unaccounted Memory: ~10 GB DISCREPANCY

User processes visible in Task Manager: ~4.5 GB
System reports in use: 14.9 GB
**Discrepancy: ~10 GB**

This memory is NOT from Cognitive Flow. Likely sources:
- Non-paged pool (2.2 GB - elevated, suggests driver issue)
- Kernel/driver allocations (not visible in Task Manager)
- Memory-mapped files
- Superfetch/prefetch caching

---

## Optimizations Applied (Pending Commit)

Added to `backends.py` for Parakeet/ONNX backend:

```python
# Session options
sess_options.enable_mem_pattern = False  # Disable memory pattern pre-computation
sess_options.enable_mem_reuse = True     # Keep reuse for efficiency

# CUDA provider options
cuda_options = {
    'arena_extend_strategy': 'kSameAsRequested',  # Don't over-allocate arenas
    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,      # Cap at 4GB GPU memory
    'cudnn_conv_algo_search': 'DEFAULT',          # Don't cache all algorithms
}
```

**Expected improvement:** Commit size should drop from 5.2 GB to ~1-2 GB.

---

## Limitations

ONNX Runtime has a known behavior where memory arenas are **never returned to the system** once allocated. From the [official documentation](https://onnxruntime.ai/docs/performance/tune-performance/memory.html):

> "The memory allocated by the arena is never returned to the system; once allocated it always remains allocated."

There is a `memory.enable_memory_arena_shrinkage` run option, but it requires access to `run_options` which the onnx-asr wrapper doesn't expose.

---

## Recommendations

### For Cognitive Flow:
1. Commit the memory optimizations (pending)
2. Test if commit size drops with new settings
3. Consider adding a "low memory mode" that uses CPU-only inference

### For System Memory Investigation:
1. Use [RAMMap](https://docs.microsoft.com/en-us/sysinternals/downloads/rammap) to identify where the ~10 GB discrepancy is going
2. Investigate the elevated 2.2 GB non-paged pool - may indicate a driver leak
3. Check for driver updates (especially GPU, network, storage drivers)

---

## Files Modified

- `cognitive_flow/backends.py` - Added ONNX memory optimization options
- `cognitive_flow/__init__.py` - Version 1.9.1
- `cognitive_flow/app.py` - Added GPU warmup on wake detection

---

## Status

Memory optimizations are coded but **not yet committed**. Awaiting confirmation to commit and test.
