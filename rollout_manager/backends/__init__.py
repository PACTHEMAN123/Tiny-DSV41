"""Backend selection without importing optional Harbor or GPU libraries."""

COMBINATIONS = {
    "kernelbench": ("native-cuda", "tls-rpc"),
    "cuda-agent": ("native-cuda", "tls-rpc"),
    "terminal-bench": ("harbor-pouch", "browser-filerelay"),
}


def validate_combination(benchmark, backend, transport):
    if COMBINATIONS.get(benchmark) != (backend, transport):
        raise ValueError("unsupported benchmark/backend/transport combination")


def create_backend(benchmark, backend, transport_name, config, *, transport, event_store=None):
    validate_combination(benchmark, backend, transport_name)
    if backend == "native-cuda":
        from .native import NativeBackend
        return NativeBackend(config, transport, event_store=event_store)
    from .terminal_bench import TerminalBenchBackend
    return TerminalBenchBackend(config, transport=transport)
