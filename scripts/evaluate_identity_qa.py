"""Bounded greedy identity answers and target NLL, with real adapter on/off.

All CP=1 ranks execute every no-cache forward on the same prompt. Rank 0 selects
and broadcasts each token; EOS and the generation budget stop all ranks together.
This module is importable by train_sft for before/after evaluation. The CLI reloads
an exported adapter in a fresh process and may compare generated answers with the
trainer's report. A successfully measured evaluation need not answer correctly.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsv41_train.lora import LoRALinear
from dsv41_train.runtime import initialize_runtime
from scripts.verify_lora_adapter import file_sha256, require, require_all, validate_reference


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def gathered(value, runtime):
    if dist.is_initialized():
        values = [None] * runtime.world_size
        dist.all_gather_object(values, value)
        return values
    require(runtime.world_size == 1, 'multi-rank evaluation requires an initialized process group')
    return [value]


def load_prompt_spec(path, tokenizer_path, *, vocab_size, max_position_embeddings,
                     config_sha256=None, max_new_tokens=16):
    from tokenizers import Tokenizer
    require(type(max_new_tokens) is int and 1 <= max_new_tokens <= 64, 'max-new-tokens must be in 1..64')
    spec = json.loads(Path(path).read_text())
    require(spec.get('format_version') == 1, 'unsupported identity prompt format')
    require(spec['tokenizer_sha256'] == file_sha256(tokenizer_path), 'identity prompt tokenizer differs')
    if config_sha256 is not None:
        require(spec['config_sha256'] == config_sha256, 'identity prompt model config differs')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    eos = spec['eos_token_ids']
    require(eos and all(type(x) is int and 0 <= x < vocab_size for x in eos), 'invalid EOS IDs')
    prompts = spec['prompts']
    require(isinstance(prompts, list) and 1 <= len(prompts) <= 8, 'identity eval requires 1..8 bounded prompts')
    require(len({row['id'] for row in prompts}) == len(prompts), 'duplicate identity prompt ID')
    for row in prompts:
        for name in ('input_ids', 'target_ids'):
            ids = row[name]
            require(isinstance(ids, list) and ids and all(type(x) is int and 0 <= x < vocab_size for x in ids), 'invalid ' + name)
        require(len(row['input_ids']) + max(len(row['target_ids']), max_new_tokens) <= max_position_embeddings,
                'identity prompt plus completion exceeds model context limit')
        require(row['target_text'] == 'RTP大模型', 'identity target must be exactly RTP大模型')
        require(row['target_ids'][-1] in eos and not any(x in eos for x in row['target_ids'][:-1]), 'target must end at its first EOS')
        decoded_prompt = tokenizer.decode(row['input_ids'], skip_special_tokens=False)
        require(decoded_prompt == row['prompt_text'], 'displayed prompt differs from actual decoded input')
        require(tokenizer.encode(decoded_prompt, add_special_tokens=False).ids == row['input_ids'],
                'identity prompt does not roundtrip through the recorded tokenizer')
        if 'question' in row:
            require(isinstance(row['question'], str) and row['question'] and row['question'] in decoded_prompt,
                    'identity question is missing from actual prompt')
        require(row['target_text'] not in decoded_prompt and row['target_text'] not in row['prompt_text'], 'target leaked into prompt')
        require(tokenizer.decode(row['target_ids'], skip_special_tokens=True).strip() == row['target_text'], 'target token decode differs')
    return spec, tokenizer


@contextmanager
def evaluation_mode(model, runtime, *, adapters_enabled, seed):
    """Restore every module flag and CPU/current-CUDA RNG even after exceptions."""
    require(type(seed) is int and seed >= 0, 'evaluation seed must be nonnegative')
    require(not getattr(getattr(model, 'context_parallel', None), 'enabled', False), 'identity generation currently requires CP=1')
    modules = list(model.modules())
    states = [(module, module.training) for module in modules]
    adapters = [(module, module.adapters_enabled) for module in modules if isinstance(module, LoRALinear)]
    require(adapters and not any(module.merged for module, _ in adapters), 'identity comparison requires unmerged native LoRA')
    signatures = gathered({'adapter_count': len(adapters), 'enabled': adapters_enabled, 'seed': seed}, runtime)
    require(all(row == signatures[0] for row in signatures), 'adapter evaluation mode differs across ranks')
    devices = [runtime.device.index if runtime.device.index is not None else torch.cuda.current_device()] if runtime.device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            model.eval()
            for module, _ in adapters:
                module.adapters_enabled = adapters_enabled
            yield
    finally:
        for module, enabled in adapters:
            module.adapters_enabled = enabled
        for module, training in states:
            module.training = training


def target_nll(model, prompt, target, runtime):
    ids = torch.tensor([prompt + target], dtype=torch.int64, device=runtime.device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))
    # Convert only answer-predicting logits to FP32, not all long-prefix logits.
    selected = output.logits[:, len(prompt) - 1:len(prompt) + len(target) - 1, :].float()
    require_all(torch.isfinite(selected).all(), runtime, 'non-finite identity target logits')
    labels = torch.tensor(target, dtype=torch.int64, device=runtime.device)
    value = F.cross_entropy(selected.reshape(-1, selected.shape[-1]), labels, reduction='mean')
    require_all(torch.isfinite(value), runtime, 'non-finite identity target NLL')
    values = gathered(float(value), runtime)
    return {'mean': values[0], 'sum': values[0] * len(target), 'tokens': len(target), 'per_rank_mean': values,
            'includes_eos': True}


def greedy_answer(model, prompt_ids, eos_token_ids, runtime, *, max_new_tokens):
    ids = torch.tensor([prompt_ids], dtype=torch.int64, device=runtime.device)
    generated = []
    reason = 'max_new_tokens'
    for _ in range(max_new_tokens):
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))
        logits = output.logits[0, -1]
        require_all(torch.isfinite(logits).all(), runtime, 'non-finite generation logits')
        local = logits.argmax().reshape(1).long()
        token = local.clone()
        if dist.is_initialized():
            dist.broadcast(token, src=0)
        require_all(local.eq(token), runtime, 'ranks disagree on greedy next token')
        value = int(token.item())
        del output, logits, local
        generated.append(value)
        if value in eos_token_ids:
            reason = 'eos'
            break
        ids = torch.cat((ids, token.reshape(1, 1)), dim=1)
    return generated, reason


def capture_identity_generation(model, spec, tokenizer, runtime, *, adapters_enabled,
                                max_new_tokens=16, seed=20261014):
    require(type(max_new_tokens) is int and 1 <= max_new_tokens <= 64, 'max-new-tokens must be in 1..64')
    identity = {'prompt_spec_sha256': object_sha(spec), 'max_new_tokens': max_new_tokens, 'seed': seed,
                'adapters_enabled': adapters_enabled}
    identities = gathered(identity, runtime)
    require(all(row == identity for row in identities), 'generation inputs/settings differ across ranks')
    records = []
    started = time.perf_counter()
    with evaluation_mode(model, runtime, adapters_enabled=adapters_enabled, seed=seed):
        for row in spec['prompts']:
            continuation, stop = greedy_answer(model, row['input_ids'], spec['eos_token_ids'], runtime,
                                                max_new_tokens=max_new_tokens)
            text = tokenizer.decode(continuation, skip_special_tokens=True)
            normalized = text.strip()
            expected = row['target_text']
            record = {'id': row['id'], 'split': row.get('split'), 'long_context': row['long_context'],
                      'seen_question_in_train': row.get('seen_question_in_train'),
                      'prompt_text': row['prompt_text'], 'prompt_token_ids': row['input_ids'],
                      'prompt_tokens': len(row['input_ids']), 'prompt_sha256': object_sha(row['input_ids']),
                      'target_text': expected, 'target_token_ids': row['target_ids'],
                      'generated_token_ids': continuation, 'generated_text': text,
                      'generated_text_with_special_tokens': tokenizer.decode(continuation, skip_special_tokens=False),
                      'normalized_text': normalized, 'stop_reason': stop, 'eos_terminated': stop == 'eos',
                      'exact_match': normalized == expected, 'contains_target': expected in normalized,
                      'strict_success': stop == 'eos' and normalized == expected,
                      'target_nll': target_nll(model, row['input_ids'], row['target_ids'], runtime)}
            records.append(record)
    return {'format_version': 1, **identity, 'world_size': runtime.world_size, 'decoding': 'greedy-no-cache',
            'eos_token_ids': spec['eos_token_ids'], 'dropout_enabled': False, 'prompts': records,
            'strict_success_count': sum(row['strict_success'] for row in records),
            'prompt_count': len(records), 'seconds_rank0': time.perf_counter() - started}


def compare_identity_generations(base, adapted):
    for key in ('prompt_spec_sha256', 'max_new_tokens', 'seed', 'world_size', 'decoding', 'eos_token_ids'):
        require(base[key] == adapted[key], 'before/after generation configuration differs: ' + key)
    require(base['adapters_enabled'] is False and adapted['adapters_enabled'] is True, 'base/adapted phases are not attributed correctly')
    require([row['id'] for row in base['prompts']] == [row['id'] for row in adapted['prompts']], 'prompt coverage differs')
    rows = []
    for before, after in zip(base['prompts'], adapted['prompts']):
        require(before['prompt_token_ids'] == after['prompt_token_ids'] and before['target_token_ids'] == after['target_token_ids'], 'before/after prompt or target differs')
        delta = after['target_nll']['mean'] - before['target_nll']['mean']
        rows.append({'id': before['id'], 'long_context': before['long_context'],
                     'seen_question_in_train': before['seen_question_in_train'],
                     'base_answer': before['generated_text'], 'adapter_answer': after['generated_text'],
                     'base_stop_reason': before['stop_reason'], 'adapter_stop_reason': after['stop_reason'],
                     'base_strict_success': before['strict_success'], 'adapter_strict_success': after['strict_success'],
                     'generated_tokens_changed': before['generated_token_ids'] != after['generated_token_ids'],
                     'target_nll_change': delta, 'target_nll_improved': delta < 0})
    return {'evaluation_completed': True, 'prompts': rows,
            'adapter_strict_success_all': all(row['adapter_strict_success'] for row in rows),
            'base_strict_success_count': sum(row['base_strict_success'] for row in rows),
            'adapter_strict_success_count': sum(row['adapter_strict_success'] for row in rows),
            'generation_changed_count': sum(row['generated_tokens_changed'] for row in rows),
            'scope': 'Identity instruction effect on the listed seen/unseen short and long prompts. Exact answers require EOS; a truncated answer is not a strict success. No general capability claim.'}


def compare_reloaded_generation(actual, expected):
    for key in ('prompt_spec_sha256', 'max_new_tokens', 'seed', 'world_size', 'decoding', 'eos_token_ids', 'adapters_enabled'):
        require(actual[key] == expected[key], 'reload generation config mismatch: ' + key)
    require([x['id'] for x in actual['prompts']] == [x['id'] for x in expected['prompts']], 'reload prompt IDs differ')
    records = []
    for left, right in zip(actual['prompts'], expected['prompts']):
        passed = all(left[k] == right[k] for k in ('prompt_token_ids', 'target_token_ids', 'generated_token_ids',
                                                  'generated_text', 'stop_reason', 'strict_success', 'target_nll'))
        records.append({'id': left['id'], 'exact_equal': passed})
    return {'passed': all(row['exact_equal'] for row in records), 'tolerance': 0, 'prompts': records}


def load_for_evaluation(reference, adapter_in, model_path, runtime, cleanup):
    from dsv41_train.loading import LoadPolicy
    from dsv41_train.models.dsv4 import load_dsv41_backbone_window
    from dsv41_train.models.dsv4.engram import EngramPolicy
    from dsv41_train.models.dsv4.materialize import FSDPMaterializer
    from dsv41_train.models.dsv4.parallel import build_parallelism
    from dsv41_train.models.dsv4.policies import FSDPPolicy
    from dsv41_train.models.qwen import load_qwen3_moe
    from dsv41_train.models.qwen.engram import shard_qwen_component, shard_qwen_root
    folder, lora = validate_reference(reference, adapter_in, model_path)
    context = reference['context']
    require(context['world_size'] == runtime.world_size, 'generation requires training WORLD_SIZE')
    meshes, cp, dispatcher = build_parallelism(1, context['ep_size'], runtime.device.type)
    if context['model_type'] == 'qwen':
        model = load_qwen3_moe(folder, device=runtime.device, dtype=torch.bfloat16, context_parallel=cp,
                              token_dispatcher=dispatcher, lora_config=lora, adapter_path=adapter_in,
                              module_callback=lambda name, module: shard_qwen_component(name, module, meshes))
        shard_qwen_root(model, meshes)
    else:
        loading, engram, fsdp = context['memory_policies']
        model = load_dsv41_backbone_window(folder, start_layer=context['start_layer'], num_layers=context['num_layers'],
            device=runtime.device, dtype=torch.bfloat16, context_parallel=cp, token_dispatcher=dispatcher,
            engram_mesh=meshes.engram, sparse_engram_gradients=context['optimizer'] == 'sgd',
            materializer=FSDPMaterializer(meshes, FSDPPolicy(**fsdp)), load_policy=LoadPolicy(**loading),
            engram_policy=EngramPolicy(**engram), lora_config=lora, adapter_path=adapter_in,
            cache_quantization=context['cache_quantization'])
        cleanup.callback(model.close)
    model.config.router_aux_loss_coef = 0.0
    return model, folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--adapter-in', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--model-path')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--expected-generation', type=Path)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'output exists; choose a new report path')
    runtime = initialize_runtime(require_distributed=True)
    try:
        reference = json.loads(args.reference.read_text())
        context = reference['context']
        folder = Path(args.model_path or context['source']['path'])
        identity = {'reference_sha256': file_sha256(args.reference), 'adapter_sha256': file_sha256(args.adapter_in),
                    'prompts_sha256': file_sha256(args.prompts)}
        require(all(row == identity for row in gathered(identity, runtime)), 'eval files differ across ranks')
        # Validate token/config identity before allocating the real model.
        from dsv41_train.models.qwen.config import Qwen3MoeConfig
        from dsv41_train.models.dsv4.config import DeepSeekV41Config
        config_class = Qwen3MoeConfig if context['model_type'] == 'qwen' else DeepSeekV41Config
        config = config_class.from_json(folder / 'config.json')
        spec, tokenizer = load_prompt_spec(args.prompts, folder / 'tokenizer.json', vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings, config_sha256=context['source']['config_sha256'], max_new_tokens=args.max_new_tokens)
        with ExitStack() as cleanup:
            model, _ = load_for_evaluation(reference, args.adapter_in, args.model_path, runtime, cleanup)
            actual = capture_identity_generation(model, spec, tokenizer, runtime, adapters_enabled=True,
                                                max_new_tokens=args.max_new_tokens, seed=args.seed)
        report = {'format_version': 1, 'evaluation_completed': True, **identity, 'context': context,
                  'adapted_fresh_process': actual,
                  'limitations': 'Greedy no-cache generation; frozen base identity hashes config/index, not every checkpoint shard.'}
        if args.expected_generation:
            expected = json.loads(args.expected_generation.read_text())
            require(expected['context'] == context and expected['adapter_sha256'] == identity['adapter_sha256'], 'expected generation attribution differs')
            report['reload_comparison'] = compare_reloaded_generation(actual, expected['adapted_after_training'])
        passed = report.get('reload_comparison', {'passed': True})['passed']
        if runtime.is_main:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write('\n')
            print(json.dumps({'evaluation_completed': True, 'reload_exact': passed,
                              'strict_success_count': actual['strict_success_count'], 'output': str(args.output)}), flush=True)
        return 0 if passed else 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
