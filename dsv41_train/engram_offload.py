"""Synchronous CPU Engram lookup and sparse SGD, without a master table copy."""

import weakref

import torch
import torch.nn.functional as F


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

    def __deepcopy__(self, memo):
        # A copied model owns a new parameter and must never remove the source
        # model's hook through a copied weak reference/RemovableHandle.
        return type(self)()

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
        rows = F.embedding(indices.to("cpu"), weight, sparse=True)
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

    return (optimizer.register_step_pre_hook(before_step),)
