"""Short greedy SFT evaluations through the native training model forward."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .cp import ContextParallel


@torch.no_grad()
def greedy_generate(model, prompt_ids, *, family, context_parallel=None,
                    eos_token_ids=(), max_new_tokens=16) -> list[int]:
    """Return new tokens, including the first EOS, without a KV cache.

    All FSDP ranks must call with the same prompt and generation limits. Only
    the last live position's vocabulary vector is reduced within each CP group;
    one token ID is then broadcast globally to keep FSDP forwards in lockstep.
    """
    if family not in {"dsv4", "qwen"}:
        raise ValueError("family must be dsv4 or qwen")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    tokens = list(prompt_ids)
    if not tokens or any(type(token) is not int or not 0 <= token < model.config.vocab_size
                         for token in tokens):
        raise ValueError("prompt_ids must contain nonempty, valid integer token IDs")
    stop_ids = set(eos_token_ids)
    if any(type(token) is not int or not 0 <= token < model.config.vocab_size for token in stop_ids):
        raise ValueError("eos_token_ids must contain valid integer token IDs")
    cp = context_parallel or getattr(model, "context_parallel", None) or ContextParallel()
    device = next(model.parameters()).device
    pad_id = getattr(model.config, "pad_token_id", None)
    pad_id = pad_id if pad_id is not None else 0
    options = {"compute_aux_loss": False} if family == "dsv4" else {"output_router_logits": False}
    training_flags = [(module, module.training) for module in model.modules()]
    generated = []
    model.eval()
    try:
        for _ in range(max_new_tokens):
            length = len(tokens)
            padded = (length + cp.size - 1) // cp.size * cp.size
            if padded > model.config.max_position_embeddings:
                raise ValueError("generation context including CP padding exceeds model capacity")
            input_ids = torch.full((1, padded), pad_id, dtype=torch.long, device=device)
            input_ids[0, :length] = torch.tensor(tokens, dtype=torch.long, device=device)
            attention_mask = torch.arange(padded, device=device).unsqueeze(0) < length
            logits = model(input_ids, attention_mask=attention_mask, **options).logits
            local_length = padded // cp.size
            if logits.shape != (1, local_length, model.config.vocab_size):
                raise ValueError("model logits do not match the context-parallel layout")
            owner, position = divmod(length - 1, local_length)
            next_logits = logits.new_zeros(model.config.vocab_size, dtype=torch.float32)
            if cp.rank == owner:
                next_logits.copy_(logits[0, position])
            del logits
            if cp.enabled:
                dist.all_reduce(next_logits, group=cp.group)
            finite = torch.isfinite(next_logits).all().to(torch.int32)
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise RuntimeError("generation produced non-finite next-token logits")
            next_token = next_logits.argmax()
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.broadcast(next_token, src=0)
            token = int(next_token.item())
            generated.append(token)
            if token in stop_ids:
                break
            tokens.append(token)
    finally:
        # Preserve intentionally mixed module modes as well as the root flag.
        for module, training in training_flags:
            module.training = training
    return generated


def evaluate_answers(model, examples, tokenizer, *, family, context_parallel=None,
                     max_new_tokens=16) -> dict:
    """Evaluate encoded examples in identical order on every FSDP rank.

    Targets are decoded only for scoring; model inputs come exclusively from
    prompt_ids. Exact match compares decoded strings without normalization.
    """
    if not examples:
        raise ValueError("answer evaluation requires at least one example")
    records = []
    for example in examples:
        target_ids = [token for token in example["labels"] if token != -100]
        if not target_ids:
            raise ValueError("evaluation example has no supervised answer tokens")
        generated_ids = greedy_generate(
            model, example["prompt_ids"], family=family, context_parallel=context_parallel,
            eos_token_ids=tokenizer.eos_token_ids, max_new_tokens=max_new_tokens,
        )
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        target = tokenizer.decode(target_ids, skip_special_tokens=True)
        records.append(dict(prompt_ids=list(example["prompt_ids"]), generated_ids=generated_ids,
                            text=text, target_ids=target_ids, target=target, exact_match=text == target))
    matched = sum(record["exact_match"] for record in records)
    return dict(count=len(records), exact_matches=matched,
                exact_match_rate=matched / len(records), examples=records)
