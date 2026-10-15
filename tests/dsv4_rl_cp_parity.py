"""Two-rank parity for shared RL token scoring, including an empty response shard."""

from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from dsv41_train.cp import ContextParallel
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.policy import DSV41Config
from dsv41_train.rl.runtime import make_model


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError("this parity test requires two ranks")
        root = Path(__file__).resolve().parents[1] / "examples/rl/tiny-dsv41"
        config = DSV41Config.from_model_dir(root)
        runtime = RLConfig(seed=37, gradient_checkpointing=True)
        reference = make_model(root, config, runtime, ("cpu",), True)
        parallel = make_model(root, config, runtime, ("cpu",), True)
        cp = ContextParallel(init_device_mesh("cpu", (2,), mesh_dim_names=("cp",)))
        parallel.backend.context_parallel = cp
        parallel.backend.model.context_parallel = cp
        tokens = torch.tensor([[1, 4, 7, 3, 8, 9, 5, 6], [1, 7, 5, 0, 4, 3, 0, 0]])
        attention = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        # All scored tokens are in rank 1; rank 0 must still join backward collectives.
        targets = torch.tensor([[-100, -100, -100, -100, 9, 5, 6, 2],
                                [-100, -100, -100, -100, 3, 2, -100, -100]])
        kwargs = dict(attention_mask=attention, target_ids=targets, target_mask=targets != -100,
                      temperature=0.8)
        expected = reference(tokens, **kwargs)
        actual = parallel(tokens, **kwargs)
        torch.testing.assert_close(actual, cp.shard(expected), rtol=1e-5, atol=1e-6)
        actual.sum().backward()
        expected.sum().backward()
        original = dict(reference.named_parameters())
        for name, parameter in parallel.named_parameters():
            expected_gradient = original[name].grad
            if expected_gradient is None:
                if parameter.grad is not None:
                    raise AssertionError(f"unexpected gradient: {name}")
                continue
            gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
            dist.all_reduce(gradient)
            torch.testing.assert_close(gradient, expected_gradient, rtol=2e-4, atol=2e-6,
                                       msg=lambda message, name=name: f"{name}: {message}")
        if dist.get_rank() == 0:
            print("RL CP scores and checkpointed gradients match, including the empty response shard", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
