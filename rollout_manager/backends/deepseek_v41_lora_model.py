"""Lazy model entry point; retains the installed NVIDIA model's forward path."""
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.models.deepseek_v41.nvidia.vl_model import DeepseekV41ForCausalLM

from .deepseek_v41_lora import (
    LORA_TARGET_MODULE, PACKED_MODULES_MAPPING, text_dimensions, validate_deployment,
)


class DeepseekV41LoRAForCausalLM(DeepseekV41ForCausalLM, SupportsLoRA):
    packed_modules_mapping = PACKED_MODULES_MAPPING
    embedding_modules = {}
    lora_skip_prefixes = ["vision.", "aligner.", "language_model.model.mtp."]

    def __init__(self, *, vllm_config, prefix=""):
        validate_deployment(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hidden, q_rank, head_dim, layers = text_dimensions(self.config)
        targets = [module for name, module in self.named_modules()
                   if name.endswith(".attn." + LORA_TARGET_MODULE)]
        if len(targets) != layers:
            raise RuntimeError("DeepSeek backbone LoRA target count changed")
        for module in targets:
            if (type(module) is not MergedColumnParallelLinear
                    or module.tp_size != 1 or module.input_size != hidden
                    or list(module.output_sizes) != [q_rank, head_dim]
                    or module.bias is not None or not module.return_bias):
                raise RuntimeError("DeepSeek fused input projection layout changed")
