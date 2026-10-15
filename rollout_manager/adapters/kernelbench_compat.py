"""Explicit compatibility correction for the pinned harness's integer cast.

The native helper casts token IDs and class labels to the activation precision.
Keep their original integer/bool dtype; floating/complex behavior is unchanged.
No generator, seed, comparison, trial count, or reference implementation changes.
"""


def preserve_integer_inputs(module, torch):
    original = module._process_input_tensor

    def transfer(value, device, backend="cuda", precision=None):
        if precision is None:
            precision = torch.float32
        if isinstance(value, torch.Tensor) and not (value.dtype.is_floating_point or value.dtype.is_complex):
            precision = value.dtype
        return original(value, device, backend, precision)

    module._process_input_tensor = transfer
    return {"name": "preserve-integer-input-dtype-v1",
            "native_file_modified": False, "runtime_transfer_helper_wrapped": True,
            "scope": "integer and bool inputs retain generator dtype; activation casting unchanged"}
