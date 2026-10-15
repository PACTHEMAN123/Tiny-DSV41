"""Native Qwen3 MoE policy using the RL manager's checkpoint interface."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib

import torch
from torch import nn

from dsv41_train.models.qwen import Qwen3MoeForCausalLM, load_qwen3_moe

from .parallel import layer_device_map, validate_available_devices


LAST_POSITION_LOGITS_PATH = "last_position_head_v1"


def validate_qwen_config(config):
    if config.model_type != "qwen3_moe":
        raise ValueError("native Qwen RL requires model_type=qwen3_moe")
    if config.attention_dropout:
        raise ValueError("RL policies require attention_dropout=0")
    if config.attention_bias or config.hidden_act != "silu":
        raise ValueError("native Qwen RL requires attention_bias=false and hidden_act=silu")
    if config.rope_scaling is not None or config.use_sliding_window or config.tie_word_embeddings:
        raise ValueError("native Qwen RL does not support RoPE scaling, sliding windows or tied embeddings")


class QwenPolicy(nn.Module):
    """Partial updates reconstruct the immutable backbone from mounted assets.

    Frozen parameters are intentionally absent from policy snapshots. Normal
    in-place mutations and parameter replacement fail before publication, while
    small normalization snapshots also detect writes through ``.data``.
    """

    def __init__(self, backend, *, trainable_scope="all"):
        super().__init__()
        validate_qwen_config(backend.config)
        if trainable_scope not in {"all", "lm_head", "lora"}:
            raise ValueError("Qwen trainable_scope must be all, lm_head or lora")
        if backend.context_parallel.enabled:
            raise ValueError("RL layer placement requires a local Qwen policy without context parallelism")
        self.backend, self.config = backend, backend.config
        self.trainable_scope = trainable_scope
        self.qwen_last_position_only = False
        self.qwen_kv_cache = False
        if trainable_scope != "lora":
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(trainable_scope == "all" or name == "backend.lm_head.weight")
        elif not any(name.endswith(".lora_a") and p.requires_grad for name, p in self.named_parameters()):
            raise ValueError("inject LoRA before constructing a Qwen LoRA policy")
        self._capture_frozen()

    def _capture_frozen(self):
        if list(self.named_buffers()):
            raise ValueError("Qwen policy has unsupported mutable buffers")
        self._frozen = {name: (parameter, parameter._version)
                        for name, parameter in self.named_parameters() if not parameter.requires_grad}
        self._normalizations = {name: parameter.detach().clone()
                                for name, (parameter, _) in self._frozen.items()
                                if "norm" in name}

    def requires_grad_(self, requires_grad=True):
        self.assert_engram_invariants()
        super().requires_grad_(requires_grad)
        if requires_grad is False:
            self._capture_frozen()
        return self

    def _apply(self, fn, recurse=True):
        if hasattr(self, "_frozen"):
            self.assert_engram_invariants()
        super()._apply(fn, recurse=recurse)
        self._capture_frozen()
        return self

    @property
    def engram_device(self):
        return "none"

    @property
    def engram_shared(self):
        return False

    def share_engram_memory(self):
        return {}

    def frozen_parameter_fingerprint(self):
        """Small diagnostic receipt; this is not a hash of every base weight."""
        self.assert_engram_invariants()
        digest = hashlib.sha256()
        for name, expected in sorted(self._normalizations.items()):
            digest.update(name.encode())
            digest.update(expected.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return {"normalization_sha256": digest.hexdigest(),
                "frozen_parameter_count": sum(parameter.numel() for parameter, _ in self._frozen.values()),
                "trainable_parameter_count": sum(parameter.numel() for parameter in self.parameters()
                                                 if parameter.requires_grad),
                "mutation_checks": "parameter_identity_version_and_normalization_values"}

    def assert_engram_invariants(self):
        if list(self.named_buffers()):
            raise RuntimeError("Qwen policy acquired unsupported mutable buffers")
        current = {name: parameter for name, parameter in self.named_parameters()
                   if not parameter.requires_grad}
        if current.keys() != self._frozen.keys():
            raise RuntimeError("Qwen frozen parameter set changed")
        for name, (parameter, version) in self._frozen.items():
            if current[name] is not parameter or parameter._version != version:
                raise RuntimeError(f"immutable Qwen base parameter changed: {name}")
        for name, expected in self._normalizations.items():
            if not torch.equal(current[name], expected):
                raise RuntimeError(f"immutable Qwen normalization changed: {name}")

    def to_training_devices(self, devices, dtype):
        placement = layer_device_map(self.config.num_hidden_layers, devices)
        validate_available_devices(devices)
        self.to(dtype=dtype)
        model = self.backend.model
        model.embed_tokens.to(placement[0])
        for layer, device in zip(model.layers, placement):
            layer.to(device)
        model.norm.to(placement[-1])
        self.backend.lm_head.to(placement[-1])
        self._capture_frozen()
        return self

    def trainable_state_dict(self):
        self.assert_engram_invariants()
        return {name: parameter.detach().cpu().clone()
                for name, parameter in self.named_parameters() if parameter.requires_grad}

    def frozen_state_dict(self, *, include_engram=False):
        self.assert_engram_invariants()
        return {}

    @torch.no_grad()
    def load_trainable_state_dict(self, state):
        self.assert_engram_invariants()
        destinations = {name: parameter for name, parameter in self.named_parameters()
                        if parameter.requires_grad}
        if state.keys() != destinations.keys():
            raise ValueError("Qwen trainable state keys do not match the configured scope")
        for name, parameter in destinations.items():
            if parameter.shape != state[name].shape:
                raise ValueError(f"Qwen policy tensor shape mismatch: {name}")
        for name, parameter in destinations.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))

    def load_frozen_state_dict(self, state, *, include_engram=False):
        if state:
            raise ValueError("Qwen frozen state must reconstruct from the identified base assets")
        self.assert_engram_invariants()

    def response_log_probs(self, prompt_ids, response_ids, temperature):
        """Replay each sampling prefix with gradients, including stop tokens.

        BF16 full-response GEMMs can produce different action probabilities from
        serial generation. Replaying its shapes costs one forward per action;
        full-parameter training also retains one backbone graph per action.
        The shared algorithm validates IDs, temperature and the returned vector.
        """
        if getattr(self, "qwen_kv_cache", False):
            from rollout_manager.backends.qwen_kv_cache import cached_response_log_probs
            return cached_response_log_probs(self, prompt_ids, response_ids, temperature)
        from .algorithm import extract_logits, input_device

        probabilities = []
        for index, token in enumerate(response_ids):
            tokens = torch.tensor([prompt_ids + response_ids[:index]], dtype=torch.long,
                                  device=input_device(self))
            if getattr(self, "qwen_last_position_only", False):
                logits = self.next_token_logits(tokens, attention_mask=torch.ones_like(tokens))[0]
            else:
                logits = extract_logits(self(tokens, attention_mask=torch.ones_like(tokens)))
                if logits.shape[1] != tokens.shape[1]:
                    raise ValueError("model must return one logits vector per input token")
                logits = logits[0, -1]
            if token >= logits.shape[-1]:
                raise ValueError("response token ID is outside the model vocabulary")
            probabilities.append((logits.float() / temperature).log_softmax(-1)[token])
        return torch.stack(probabilities)

    def next_token_logits(self, input_ids, attention_mask=None):
        logits = self(input_ids, attention_mask, last_position_only=self.qwen_last_position_only)
        expected = 1 if self.qwen_last_position_only else input_ids.shape[1]
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != expected:
            raise ValueError("Qwen next-token logits have an invalid shape")
        return logits[:, -1]

    def forward(self, input_ids, attention_mask=None, *, last_position_only=False):
        model = self.backend.model
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.config.max_position_embeddings:
            raise ValueError("input_ids must be [batch, sequence] within max_position_embeddings")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        hidden = model.embed_tokens(input_ids.to(model.embed_tokens.weight.device))
        positions = torch.arange(input_ids.shape[1], device=hidden.device).unsqueeze(0)
        rotary = {}
        for layer in model.layers:
            device = next(layer.parameters()).device
            with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
                hidden = hidden.to(device)
                if device not in rotary:
                    local_positions = positions.to(device)
                    cos, sin = model.rotary_emb(hidden, local_positions)
                    mask = None if attention_mask is None else attention_mask.to(device)
                    rotary[device] = cos, sin, mask, local_positions
                hidden, _ = layer(hidden, *rotary[device])
        hidden = model.norm(hidden)
        if last_position_only:
            # A view (including an already-contiguous last row) would retain
            # every prefix position in the head's saved backward input.
            hidden = hidden[:, -1:].clone()
        return self.backend.lm_head(hidden)


def build_qwen_policy(model_dir, config, *, random_init, devices, dtype, trainable_scope="all",
                      lora_config=None, initial_adapter=None):
    validate_qwen_config(config)
    layer_device_map(config.num_hidden_layers, devices)
    validate_available_devices(devices)
    adapter, binding = None, None
    if lora_config is not None and initial_adapter is not None:
        from dsv41_train.lora import adapter_base_identity, read_adapter
        binding = adapter_base_identity(model_dir, random_init=random_init, seed=torch.initial_seed(),
                                        private_engram=False)
        adapter = read_adapter(initial_adapter, base_identity=binding)
    backend = Qwen3MoeForCausalLM(config) if random_init else load_qwen3_moe(model_dir, dtype=dtype)
    if backend.config.to_dict() != config.to_dict():
        raise ValueError("Qwen mounted checkpoint configuration differs from requested architecture")
    if lora_config is not None:
        from dsv41_train.lora import inject_lora, load_adapter_state, validate_adapter_layout
        inject_lora(backend, lora_config)
        if adapter is not None:
            validate_adapter_layout(backend, adapter)
            load_adapter_state(backend, adapter, base_identity=binding)
    return QwenPolicy(backend, trainable_scope=trainable_scope).to_training_devices(devices, dtype)
