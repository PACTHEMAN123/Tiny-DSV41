"""Independent CPU dKV oracle for the native attention backward's BF16 casts.

Uses actual saved BF16 output and FP32 LSE. It deliberately does not reproduce
Triton head/slot tiles, inverse adjacency, atomics or the ordered reduction tree.
CPU FP32 dot/exp/delta can still round differently from native Tensor Cores.
"""
import torch


@torch.no_grad()
def quantized_kv_reference(saved_tensors, grad_output):
    """Return FP64 key sums AND sums cast to original KV dtype; never mutate inputs."""
    if len(saved_tensors) != 9:
        raise ValueError('expected the native nine saved tensors')
    query, kv, indices, mask, compressed, picked, sinks, output, lse = saved_tensors
    values = (*saved_tensors, grad_output)
    if any(t.device.type != 'cpu' for t in values):
        raise ValueError('CPU tensors required; GPU copies belong to the diagnostic caller')
    if any(t.requires_grad for t in values):
        raise ValueError('detached diagnostic tensors required')
    batch, heads, length, dim = query.shape
    if output.shape != query.shape or grad_output.shape != query.shape or lse.shape != (batch, heads, length):
        raise ValueError('saved output/upstream/LSE shape differs')
    if query.dtype != torch.bfloat16 or output.dtype != query.dtype or grad_output.dtype != query.dtype or lse.dtype != torch.float32:
        raise ValueError('expected BF16 Q/O/dO and FP32 LSE')
    if mask.shape != indices.shape or indices.shape[:2] != (batch, length) or picked.shape[:2] != (batch, length):
        raise ValueError('index/mask layout differs')
    if any(t.ndim != 3 or t.shape[0] != batch or t.shape[2] != dim or t.dtype != query.dtype for t in (kv, compressed)):
        raise ValueError('KV layout/dtype differs')
    if not all(bool(torch.isfinite(t).all()) for t in (query, kv, compressed, output, lse, grad_output)):
        raise ValueError('nonfinite diagnostic input')
    sums = [torch.zeros(t.shape, dtype=torch.float64) for t in (kv, compressed)]
    counts = [0, 0]; scale = dim ** -.5
    for b in range(batch):
        for qpos in range(length):
            q = query[b, :, qpos].float()
            upstream = grad_output[b, :, qpos].float()
            # Actual rounded O is intentional: native backward uses this value,
            # not a recomputed ideal attention output.
            delta = (output[b, :, qpos].float() * upstream).sum(dim=-1)
            for branch, (keys, ids, valid) in enumerate((
                (kv[b], indices[b, qpos], mask[b, qpos].bool()),
                (compressed[b], picked[b, qpos], torch.ones_like(picked[b, qpos], dtype=torch.bool)))):
                valid = valid & (ids >= 0) & (ids < keys.shape[0])
                ids = ids[valid].long()
                if ids.numel() == 0:
                    continue
                selected = keys.index_select(0, ids).float()
                scores = (q @ selected.T) * scale
                probabilities = torch.exp(scores - lse[b, :, qpos, None])
                grad_probabilities = upstream @ selected.T
                grad_scores = probabilities * (grad_probabilities - delta[:, None])
                # These two BF16 conversions are part of the production formula.
                p = probabilities.bfloat16().double()
                ds = grad_scores.bfloat16().double()
                contributions = p.T @ upstream.double() + (ds.T @ q.double()) * scale
                # Explicit query/slot order and duplicate occurrences. No sparse
                # inverse/reducer/native backward code is used by this oracle.
                for slot, key in enumerate(ids.tolist()):
                    sums[branch][b, key].add_(contributions[slot])
                counts[branch] += ids.numel()
    return dict(fp64_sums=dict(grad_kv=sums[0], grad_compressed=sums[1]),
                cast_to_kv_dtype=dict(grad_kv=sums[0].to(kv.dtype), grad_compressed=sums[1].to(compressed.dtype)),
                valid_slot_occurrences=dict(window=counts[0], compressed=counts[1]),
                scope='Independent CPU FP32 local dot/exp/delta; BF16 P/dScore; FP64 head and fixed query/slot key sums; actual native rounded O/LSE. Not a bitwise TensorCore oracle.')
