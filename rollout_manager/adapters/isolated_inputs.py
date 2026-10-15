"""Opt-in correctness input isolation; never applied to historical records."""


def clone_call_inputs(torch, args, kwargs):
    """Keep duplicate object identity; refuse view aliases we cannot preserve."""
    tensors, storages, memo = {}, {}, {}

    def inspect(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != 'cuda' or value.layout != torch.strided:
                raise ValueError('isolated example requires dense CUDA tensor inputs')
            storage = (str(value.device), value.untyped_storage().data_ptr())
            if value.numel() and storage in storages and storages[storage] != id(value):
                raise ValueError('distinct aliased tensor views need a dedicated native adapter')
            storages[storage] = id(value)
            tensors[id(value)] = value
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)
        elif isinstance(value, dict):
            for item in value.values():
                inspect(item)
        elif not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError('unsupported mutable native input')

    inspect((args, kwargs))
    for identity, value in tensors.items():
        memo[identity] = value.clone(memory_format=torch.preserve_format)

    def copy(value):
        if isinstance(value, torch.Tensor):
            return memo[id(value)]
        if isinstance(value, list):
            return [copy(x) for x in value]
        if isinstance(value, tuple):
            return tuple(copy(x) for x in value)
        if isinstance(value, dict):
            return {k: copy(v) for k, v in value.items()}
        return value

    return copy(args), copy(kwargs)
