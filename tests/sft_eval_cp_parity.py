"""Two-rank native-model greedy parity across CP ownership/padding boundaries."""

import copy
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from dsv41_train.cp import ContextParallel
from dsv41_train.lora import LoRAConfig, inject_lora
from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.sft_eval import greedy_generate
from test_sft_eval import tiny_model


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError("this parity test requires exactly two ranks")
        cp = ContextParallel(init_device_mesh("cpu", (2,), mesh_dim_names=("cp",)))
        for family in ("dsv4", "qwen"):
            reference = tiny_model(family)
            if family == "dsv4":
                inject_lora(reference, LoRAConfig(rank=2, alpha=4))
            if family == "qwen":
                model = Qwen3MoeForCausalLM(reference.config, context_parallel=cp)
                model.load_state_dict(reference.state_dict())
            else:
                model = copy.deepcopy(reference)
                model.context_parallel = model.model.context_parallel = cp
            for prompt in ([1], [1, 4, 5]):
                expected = greedy_generate(reference, prompt, family=family, max_new_tokens=3)
                reduce = dist.all_reduce
                vector_sizes = []

                def check_vector(tensor, *args, **kwargs):
                    assert tensor.ndim == 0 or tensor.shape == (model.config.vocab_size,)
                    if tensor.ndim:
                        vector_sizes.append(tensor.numel())
                    return reduce(tensor, *args, **kwargs)

                with patch.object(dist, "all_reduce", check_vector):
                    actual = greedy_generate(model, prompt, family=family,
                                             context_parallel=cp, max_new_tokens=3)
                assert actual == expected, (family, prompt, actual, expected)
                assert len(vector_sizes) == 3
        # One data-parallel rank's invalid logits must make every rank fail
        # before the next token broadcast, instead of leaving peers blocked.
        model = tiny_model("qwen")
        if dist.get_rank() == 0:
            with torch.no_grad():
                model.lm_head.weight.fill_(float("nan"))
        try:
            greedy_generate(model, [1], family="qwen", max_new_tokens=1)
        except RuntimeError as error:
            assert "non-finite" in str(error)
        else:
            raise AssertionError("every rank must reject non-finite generation logits")
        if dist.get_rank() == 0:
            print("DSV4+LoRA and Qwen CP2 greedy parity passed; only vocabulary vectors reduced", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
