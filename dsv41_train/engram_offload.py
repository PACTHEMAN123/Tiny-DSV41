"""Synchronous CPU Engram lookup and sparse SGD, without a master table copy."""

import weakref

import torch
import torch.nn.functional as F


class _CachedRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, indices, cache):
        ids = indices.reshape(-1).long()
        ctx.save_for_backward(ids)
        ctx.shape = weight.shape
        rows = cache.read(weight, ids)
        return rows.view(*indices.shape, weight.shape[1])

    @staticmethod
    def backward(ctx, gradient):
        (ids,) = ctx.saved_tensors
        # Keep all occurrences, including hits and duplicates, in original
        # request order. CPUEmbeddingLookup handles microbatch accumulation.
        values = gradient.reshape(-1, ctx.shape[1]).to("cpu")
        gradient = torch.sparse_coo_tensor(
            ids.unsqueeze(0), values, ctx.shape, is_coalesced=False, check_invariants=False)
        return gradient, None, None


class _RowCache:
    """Owner-local direct mapping; the byte budget includes CPU int64 tags."""

    def __init__(self, budget, device):
        self.budget, self.device = budget, device
        self.values = self.tags = self.signature = self.version = self._weight = None
        self._event = None
        self.metrics = dict(hits=0, misses=0, transferred_rows=0, resident_bytes=0)

    def wait(self):
        if self._event is not None:
            self._event.synchronize()
            self._event = None

    def clear(self):
        self.wait()
        if self.tags is not None:
            self.tags.fill_(-1)
        self.version = None

    def _validate(self, weight):
        signature = (weight.data_ptr(), weight.untyped_storage().data_ptr(), tuple(weight.shape),
                     tuple(weight.stride()), weight.dtype, weight.device, self.device)
        previous = self._weight() if self._weight is not None else None
        if previous is not weight or signature != self.signature:
            capacity = min(weight.shape[0], self.budget // (weight.shape[1] * weight.element_size() + 8))
            if capacity < 1:
                raise ValueError("Engram cache budget must hold one complete row and its int64 tag")
            # Drop previous storage before allocating a new dtype/shape; never
            # retain two complete cache generations inside this byte budget.
            self.values = self.tags = None
            self.metrics["resident_bytes"] = 0
            # Persistent buffers must remain writable when a caller switches
            # from inference_mode back to ordinary evaluation or training.
            with torch.inference_mode(False), torch.no_grad():
                self.values = torch.empty((capacity, weight.shape[1]), dtype=weight.dtype, device=self.device)
                self.tags = torch.full((capacity,), -1, dtype=torch.long, device="cpu")
            self.metrics["resident_bytes"] = self.values.numel() * self.values.element_size() + self.tags.numel() * 8
            self._weight, self.signature = weakref.ref(weight), signature
            self.version = weight._version
        elif weight._version != self.version:
            self.tags.fill_(-1)
            self.version = weight._version

    def read(self, weight, ids):
        # A prior access may have used another CUDA stream. Wait only for its
        # cache operations before reading or overwriting slots on this stream.
        self.wait()
        self._validate(weight)
        if ids.numel() and (ids.min() < 0 or ids.max() >= weight.shape[0]):
            raise IndexError("Engram cache row is outside its owner's range")
        try:
            output = torch.empty((ids.numel(), weight.shape[1]), dtype=weight.dtype, device=self.device)
            slots = ids.remainder(self.tags.numel())
            hit = self.tags[slots] == ids
            hits, misses = hit.nonzero().flatten(), (~hit).nonzero().flatten()
            self.metrics["hits"] += hits.numel()
            self.metrics["misses"] += misses.numel()
            # Outputs are independent copies. Read every hit before a colliding
            # miss replaces that slot; old outputs remain valid for backward.
            if hits.numel():
                output.index_copy_(0, hits.to(self.device), self.values.index_select(0, slots[hits].to(self.device)))
            if misses.numel():
                missing, inverse = ids[misses].unique(sorted=True, return_inverse=True)
                rows = weight.detach().index_select(0, missing).to(self.device)
                self.metrics["transferred_rows"] += missing.numel()
                output.index_copy_(0, misses.to(self.device), rows.index_select(0, inverse.to(self.device)))
                new_slots, collisions = missing.remainder(self.tags.numel()).unique(sorted=True, return_inverse=True)
                chosen = torch.full_like(new_slots, -1)
                chosen.scatter_reduce_(0, collisions, torch.arange(missing.numel(), device="cpu"), reduce="amax")
                self.values.index_copy_(0, new_slots.to(self.device), rows.index_select(0, chosen.to(self.device)))
                self.tags[new_slots] = missing[chosen]
            return output
        except Exception:
            self.tags.fill_(-1)
            self.version = None
            raise
        finally:
            if self.device.type == "cuda":
                self._event = torch.cuda.Event()
                self._event.record(torch.cuda.current_stream(self.device))


class CPUEmbeddingLookup:
    """Keep occurrences in ``weight.grad`` until SGD coalesces them in FP32.

    Training uses one lookup per table per backward, outside decoder activation
    checkpointing. Concurrent/higher-order backwards and retrying an aborted
    retained graph are not supported; recovery starts with a fresh forward.
    """

    def __init__(self):
        self._parameter = None
        self._handle = None
        self._backwards = 0
        self._cache = None

    def configure_cache(self, budget_bytes, device):
        """Set a per-owner budget including tags; zero keeps native lookup.

        Counters reset on configure. Request outputs/miss staging are additional
        temporary storage, not cache residents. Concurrent calls are unsupported.
        """
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("Engram cache budget must be a nonnegative integer")
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Engram cache supports CPU tests or CUDA execution")
        if device.type == "cpu":
            device = torch.device("cpu")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self._cache is not None:
            self._cache.wait()
        self._cache = _RowCache(budget_bytes, device) if budget_bytes else None

    def clear_cache(self):
        """Invalidate contents, retaining allocation/counters within the budget.

        Call explicitly after writes through .data, which bypass tensor versions.
        """
        if self._cache is not None:
            self._cache.clear()

    def synchronize(self):
        if self._cache is not None:
            self._cache.wait()

    @property
    def cache_metrics(self):
        return dict(self._cache.metrics) if self._cache is not None else dict(
            hits=0, misses=0, transferred_rows=0, resident_bytes=0)

    def __deepcopy__(self, memo):
        # A copied model owns a new parameter and must never remove the source
        # model's hook through a copied weak reference/RemovableHandle.
        result = type(self)()
        if self._cache is not None:
            result.configure_cache(self._cache.budget, self._cache.device)
        return result

    def _ensure_hook(self, weight):
        previous = self._parameter() if self._parameter is not None else None
        eligible = weight.requires_grad and weight.dtype in (torch.bfloat16, torch.float16)
        if eligible and previous is weight:
            if next(reversed(weight._backward_hooks or {}), None) != self._handle.id:
                raise RuntimeError("register gradient-transform hooks before the first CPU Engram lookup")
            return
        if self._handle is not None:
            self._handle.remove()
        self._parameter = self._handle = None
        if not eligible:
            return
        parameter_ref = weakref.ref(weight)

        def preserve_occurrences(incoming):
            count, self._backwards = self._backwards, 0
            if count > 1:
                raise RuntimeError("CPU low-precision Engram requires one lookup per table per backward")
            if torch.is_grad_enabled():
                raise RuntimeError("CPU Engram supports first-order backward only")
            parameter = parameter_ref()
            if parameter is None or parameter.grad is None:
                return incoming
            if next(reversed(parameter._backward_hooks or {}), None) != self._handle.id:
                raise RuntimeError("register gradient-transform hooks before the first CPU Engram lookup")
            existing = parameter.grad
            if not incoming.is_sparse or not existing.is_sparse:
                raise RuntimeError("CPU Engram requires sparse COO gradients")
            # Allocate before clearing .grad: an allocation failure must not lose
            # the previous microbatch. zero_grad owns all accumulated values.
            combined = torch.sparse_coo_tensor(
                torch.cat((existing._indices(), incoming._indices()), dim=1),
                torch.cat((existing._values(), incoming._values())), incoming.shape,
                dtype=incoming.dtype, device=incoming.device, is_coalesced=False,
                check_invariants=False,
            )
            parameter.grad = None
            return combined

        self._parameter = parameter_ref
        self._handle = weight.register_hook(preserve_occurrences)

    def __call__(self, weight, indices, *, device):
        if weight.device.type != "cpu":
            raise RuntimeError("offloaded Engram weights must stay on CPU")
        self._ensure_hook(weight)
        # No gradient values live here; a fresh forward also discards bookkeeping
        # from an aborted backward. Optimizer.zero_grad can discard a whole step.
        self._backwards = 0
        device = torch.device(device)
        if device.type == "cpu":
            device = torch.device("cpu")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self._cache is not None and device != self._cache.device:
            raise ValueError("Engram lookup device differs from the configured cache device")
        if indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("Engram indices must be int32 or int64")
        indices = indices.to("cpu")
        rows = (F.embedding(indices, weight, sparse=True) if self._cache is None
                else _CachedRows.apply(weight, indices, self._cache))
        if self._handle is not None and rows.requires_grad:
            def count_backward(gradient):
                if self._backwards:
                    raise RuntimeError("CPU low-precision Engram requires one lookup per table per backward")
                self._backwards += 1
                return gradient
            rows.register_hook(count_backward)
        # Routing collectives use the caller's device, never the CPU parameter's
        # placement. Autograd copies only requested row gradients back to CPU.
        return rows.to(device)


def install_engram_sgd_hooks(model, optimizer):
    """Return removable optimizer hooks; call before the first optimizer step.

    Coalesce all trainable sparse Engram tables, including resident controls,
    with FP32 accumulation. Frozen tables need no hooks and may use AdamW.
    """
    tables = [module for module in model.modules()
              if hasattr(module, "engram_offload") and getattr(module, "sparse_gradients", False)
              and module.weight.requires_grad]
    if not tables:
        return ()
    if type(optimizer) is not torch.optim.SGD:
        raise ValueError("trainable sparse Engram requires native torch.optim.SGD")

    def validate():
        groups = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
        for table in tables:
            group = groups.get(id(table.weight))
            if group is None:
                raise ValueError("optimizer does not own the current Engram parameter")
            if any(group.get(key) for key in ("momentum", "weight_decay", "maximize", "differentiable", "foreach", "fused")):
                raise ValueError("sparse Engram SGD requires zero momentum/decay and foreach/fused disabled")
            if table.engram_offload and table.weight.device.type != "cpu":
                raise ValueError("offloaded Engram weights must stay on CPU")

    validate()

    def before_step(_optimizer, _args, _kwargs):
        if _kwargs.get("closure") is not None or any(arg is not None for arg in _args[1:]):
            raise ValueError("Engram SGD hooks require backward before step, without a closure")
        validate()
        for table in tables:
            gradient = table.weight.grad
            if gradient is not None:
                if not gradient.is_sparse:
                    raise RuntimeError("Engram SGD requires sparse COO gradients")
                table.weight.grad = (gradient.float().coalesce().to(gradient.dtype)
                                     if gradient.dtype in (torch.bfloat16, torch.float16)
                                     else gradient.coalesce())

    def after_step(_optimizer, _args, _kwargs):
        for table in tables:
            lookup = getattr(table, "_cpu_lookup", None)
            if lookup is not None:
                lookup.clear_cache()

    return (optimizer.register_step_pre_hook(before_step), optimizer.register_step_post_hook(after_step))
