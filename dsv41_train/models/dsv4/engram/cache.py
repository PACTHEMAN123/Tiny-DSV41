"""Bounded owner-local row cache. Table parameters remain the training authority."""

import torch


class _CachedRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, ids, cache, device, copy_rows):
        ctx.save_for_backward(ids)
        ctx.shape = weight.shape
        ctx.source_device = weight.device
        ctx.metrics = cache.metrics
        return cache.read(weight, ids, device, copy_rows)

    @staticmethod
    def backward(ctx, gradient):
        (ids,) = ctx.saved_tensors
        if gradient.is_cuda and ctx.source_device.type == "cpu":
            ctx.metrics["d2h_bytes"] += gradient.numel() * gradient.element_size()
        # Preserve every occurrence and its order, exactly as sparse embedding
        # backward. Only the optimizer coalesces accumulated low-precision rows.
        values = gradient.to(ctx.source_device)
        result = torch.sparse_coo_tensor(ids.to(ctx.source_device).unsqueeze(0), values, ctx.shape)
        return result, None, None, None, None


class RowCache:
    """Direct-mapped slots keep both GPU values and CPU metadata bounded."""

    def __init__(self, budget, metrics):
        self.budget = budget
        self.metrics = metrics
        self.values = None
        self.tags = None
        self.weight = None
        self.version = None
        self.storage = None

    @property
    def allocated_bytes(self):
        return 0 if self.values is None else self.values.numel() * self.values.element_size()

    def clear(self):
        self.values = self.tags = self.weight = self.version = self.storage = None

    def _validate(self, weight, device):
        identity = (weight.data_ptr(), weight.shape, weight.dtype, device)
        if self.weight is not weight or self.storage != identity:
            capacity = min(weight.shape[0], self.budget // (weight.shape[1] * weight.element_size()))
            if capacity < 1:
                raise ValueError("Engram cache budget must hold at least one row")
            self.values = torch.empty((capacity, weight.shape[1]), dtype=weight.dtype, device=device)
            self.tags = torch.full((capacity,), -1, dtype=torch.long)
            self.weight, self.storage = weight, identity
            self.version = weight._version
        elif weight._version != self.version:
            self.metrics["cache_invalidated_rows"] += int((self.tags >= 0).sum())
            self.tags.fill_(-1)
            self.version = weight._version

    def before_step(self, weight):
        if self.values is not None:
            self._validate(weight, self.values.device)

    def after_step(self, weight):
        if self.values is None or weight._version == self.version:
            return
        gradient = weight.grad
        if (self.weight is weight and weight._version == self.version + 1
                and gradient is not None and gradient.is_sparse):
            ids = gradient._indices()[0].cpu()
            slots = ids.remainder(self.tags.numel())
            changed = slots[self.tags[slots] == ids].unique()
            self.metrics["cache_invalidated_rows"] += changed.numel()
            self.tags[changed] = -1
            self.version = weight._version
        else:
            self._validate(weight, self.values.device)

    def lookup(self, weight, ids, device, copy_rows):
        return _CachedRows.apply(weight, ids, self, device, copy_rows)

    def read(self, weight, ids, device, copy_rows):
        self._validate(weight, device)
        count, width = ids.numel(), weight.shape[1]
        if not count:
            return torch.empty((0, width), dtype=weight.dtype, device=device)
        slots = ids.remainder(self.tags.numel())
        hit = self.tags[slots] == ids
        hits = hit.nonzero().flatten()
        misses = (~hit).nonzero().flatten()
        self.metrics["cache_hits"] += hits.numel()
        self.metrics["cache_misses"] += misses.numel()
        if not misses.numel():
            if device.type == "cuda":
                self.metrics["cache_metadata_h2d_bytes"] += slots.numel() * slots.element_size()
            return self.values.index_select(0, slots.to(device, non_blocking=True))
        output = torch.empty((count, width), dtype=weight.dtype, device=device)
        missing, inverse = ids[misses].unique(sorted=True, return_inverse=True)
        new_slots, collisions = missing.remainder(self.tags.numel()).unique(sorted=True, return_inverse=True)
        chosen = torch.full((new_slots.numel(),), -1, dtype=torch.long)
        chosen.scatter_reduce_(0, collisions, torch.arange(missing.numel()), reduce="amax")
        self.metrics["cache_fill_rows"] += missing.numel()
        self.metrics["cache_evictions"] += int((self.tags[new_slots] >= 0).sum())
        # One metadata transfer avoids a CUDA synchronization per index array.
        arrays = (hits, slots[hits], misses, inverse, new_slots, chosen)
        packed = torch.cat(arrays)
        if device.type == "cuda":
            self.metrics["cache_metadata_h2d_bytes"] += packed.numel() * packed.element_size()
        packed = packed.to(device, non_blocking=True)
        hit_positions, hit_slots, miss_positions, inverse_gpu, slots_gpu, chosen_gpu = packed.split(
            [value.numel() for value in arrays])
        # Materialize hits before eviction; returned rows never alias mutable slots.
        if hits.numel():
            output.index_copy_(0, hit_positions, self.values.index_select(0, hit_slots))
        if missing.numel():
            rows = copy_rows(weight.detach().index_select(0, missing.to(weight.device)), device)
            output.index_copy_(0, miss_positions, rows.index_select(0, inverse_gpu))
            self.values.index_copy_(0, slots_gpu, rows.index_select(0, chosen_gpu))
            self.tags[new_slots] = missing[chosen]
        return output
