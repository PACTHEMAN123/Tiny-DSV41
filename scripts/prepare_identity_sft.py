"""Prepare template-faithful identity SFT and prompt-only generation checks.

Uses only local tokenizer/config/template files, tokenizers and Jinja2. No model
weights, Transformers, torch, CUDA, or network access are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from jinja2 import StrictUndefined, meta, nodes
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


FORMAT = "identity-chat-template-v1"
TARGET = "RTP大模型"
TRAIN_QUESTIONS = (
    "你是谁", "请问你叫什么名字？", "请告诉我你的名称。", "你的名字是什么？",
    "我应该怎样称呼你？", "请介绍你的名字。", "能说一下你的名称吗？", "你叫什么？",
    "请用名字介绍自己。", "你的正式名称是什么？", "我该叫你什么名字？", "请报出你的名称。",
    "可以告诉我你叫什么吗？", "你是哪个模型？", "请说出你的模型名称。", "你的模型叫什么名字？",
    "能介绍一下你是谁吗？", "我正在和谁对话？", "请回答你的身份名称。", "你以什么名字提供服务？",
    "请问当前助手的名字是？", "怎样称呼当前这位助手？", "这位对话助手叫什么？", "请给出你自己的名字。",
    "你的名称怎么说？", "能否报一下自己的名字？", "你通常被称为什么？", "请告诉我这位助手的名称。",
    "回答前请先报一下名字。", "请说明你是哪一个模型。", "与你交谈时应该如何称呼你？", "我想知道你的名字。",
)
UNSEEN_QUESTIONS = ("这次对话由哪位助手回应？", "你的模型身份用什么名称表示？", "我该用哪个名字来指代正在回答的你？")

# These authored, neutral backgrounds contain no identity answer or definition.
TRAIN_BACKGROUND = (
    "图书馆的阅览室每天整理书架，管理员按主题核对目录，并记录借阅和归还情况。",
    "花园里种着几种常见植物，观察者记录叶片颜色、土壤湿度以及当天的光照时长。",
    "河边步道设置了距离标记，散步的人可以根据标记估计路程，并在休息区补充饮水。",
    "工作台上的木制零件依尺寸分类摆放，测量记录注明长度、宽度和边缘是否平整。",
    "气象记录表依日期排列，表中包括云量、风向和降雨时段，缺失项目保留为空白。",
    "厨房整理时将量杯和储物盒分开放置，标签记录容量，清洗后的器具放在架上晾干。",
    "社区展览的画作按展厅编号登记，工作人员检查照明，并确认说明卡的位置是否合适。",
    "公交站附近的新指示牌标注行走方向，维护人员定期检查字迹和固定支架的状态。",
)
HELDOUT_BACKGROUND = (
    "天文观测小组在郊外架设望远镜，记录星图编号、镜片清洁情况以及观测开始的时刻。",
    "观测结束后成员核对日志，把不同日期的照片分组保存，待天气晴朗时继续观察。",
)
STRESS_BACKGROUNDS = (
    ("地质调查人员沿山谷记录岩层颜色和颗粒大小，将采样位置绘制在纸质地图上。",
     "每份岩石样本附有地点编号，调查结束后依次检查包装，避免运输过程中的相互碰撞。"),
    ("音乐教室里排列着不同尺寸的木管乐器，保管员逐件登记管身材料以及维护日期。",
     "排练结束后演奏者擦拭乐器表面，把谱架折叠收好，并将练习记录放回资料柜。"),
)


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _raise_template(message):
    raise ValueError(f"chat template: {message}")


class TemplateCodec:
    def __init__(self, model_path):
        folder = Path(model_path)
        config_bytes = (folder / "config.json").read_bytes()
        tokenizer_bytes = (folder / "tokenizer.json").read_bytes()
        tokenizer_config_bytes = (folder / "tokenizer_config.json").read_bytes()
        self.config = json.loads(config_bytes)
        tokenizer_config = json.loads(tokenizer_config_bytes)
        self.model_type = self.config.get("model_type")
        require(self.model_type in ("qwen3_moe", "deepseek_v41"), "unsupported model_type; no guessed chat template")
        template = tokenizer_config.get("chat_template")
        template_file = folder / "chat_template.jinja"
        if template_file.is_file():
            file_template = template_file.read_text(encoding="utf-8")
            require(template is None or template == file_template, "conflicting chat templates")
            template, self.template_source = file_template, "chat_template.jinja"
        else:
            self.template_source = "tokenizer_config.json:chat_template"
        require(isinstance(template, str) and template.strip(), "local string chat template is required")
        self.template_text = template
        self.tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        text_config = self.config.get("text_config", self.config)
        self.max_positions = text_config.get("max_position_embeddings")
        require(type(self.max_positions) is int and self.max_positions > 0, "missing positive max_position_embeddings")
        configured_eos = text_config.get("eos_token_id", self.config.get("eos_token_id"))
        configured_eos = configured_eos if isinstance(configured_eos, list) else [configured_eos]
        require(configured_eos and all(type(item) is int and item >= 0 for item in configured_eos), "invalid configured EOS")
        eos_text = tokenizer_config.get("eos_token")
        eos_text = eos_text.get("content") if isinstance(eos_text, dict) else eos_text
        eos_id = self.tokenizer.token_to_id(eos_text) if isinstance(eos_text, str) else None
        require(eos_id in configured_eos, "tokenizer EOS and model EOS disagree")
        self.eos_token_ids = configured_eos
        self.options = {"enable_thinking": False}
        if self.model_type == "deepseek_v41":
            self.options["thinking_mode"] = "chat"
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)
        environment.globals["raise_exception"] = _raise_template
        environment.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
        variables = {"messages", "add_generation_prompt", "tools", "bos_token", "eos_token", "pad_token",
                     "enable_thinking", "thinking_mode", "drop_thinking", "reasoning_effort"}
        syntax = environment.parse(template)
        # Jinja's meta pass includes some names assigned inside conditional
        # branches. They are template-local; StrictUndefined still checks use.
        assigned = {node.target.name for node in syntax.find_all(nodes.Assign)
                    if isinstance(node.target, nodes.Name)}
        unknown = meta.find_undeclared_variables(syntax) - variables - assigned
        require(not unknown, f"unsupported chat template globals: {sorted(unknown)}")
        self.template = environment.from_string(template)
        self.special_tokens = {}
        for key in ("bos_token", "eos_token", "pad_token"):
            value = tokenizer_config.get(key)
            self.special_tokens[key] = value.get("content") if isinstance(value, dict) else value
        self.encoding = {"format": FORMAT, "tokenizer_sha256": digest(tokenizer_bytes),
                         "config_sha256": digest(config_bytes), "template_sha256": digest(template),
                         "tokenizer_config_sha256": digest(tokenizer_config_bytes),
                         "model_type": self.model_type, "template_options": self.options}

    def render(self, user, answer=None):
        def message(role, content):
            return {"role": role, "content": content, "reasoning_content": "", "tool_calls": []}
        messages = [message("user", user)]
        if answer is not None:
            messages.append(message("assistant", answer))
        return self.template.render(messages=messages, add_generation_prompt=answer is None, tools=None,
                                    **self.special_tokens, **self.options)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def background(sentences, length, variant):
    # Bounded source length, intentionally longer than the eventual token budget.
    return "\n".join(f"记录{index + 1}：{sentences[(index + variant) % len(sentences)]}"
                     for index in range(length // 8 + 96))


def _crop_context(codec, prompt_encoding, full_ids, rendered_prompt, context, desired, *, prompt_length=False):
    ids = prompt_encoding.ids
    excess = (len(ids) if prompt_length else len(full_ids)) - desired
    require(excess >= 0, "neutral context is too short for the requested sequence")
    if excess == 0:
        return ids, full_ids, {"removed_tokens": 0, "unicode_roundtrip_verified": True}
    begin = rendered_prompt.index(context)
    end = begin + len(context)
    offsets = prompt_encoding.offsets
    eligible = [i for i, (start, stop) in enumerate(offsets) if begin < start < stop < end]
    require(eligible, "template did not retain a tokenizable neutral context")
    low, high = eligible[0], eligible[-1] + 1
    # Search only whole-token spans strictly inside the neutral text. Both sides
    # must be complete Unicode boundaries; re-encoding proves BPE boundaries too.
    for first in range(low, min(high - excess, low + 2048)):
        last = first + excess
        if last >= high:
            break
        if offsets[first - 1][1] > offsets[first][0] or offsets[last - 1][1] > offsets[last][0]:
            continue
        prompt_ids = ids[:first] + ids[last:]
        decoded = codec.decode(prompt_ids)
        if "\ufffd" in decoded or codec.encode(decoded).ids != prompt_ids:
            continue
        cropped_full = full_ids[:first] + full_ids[last:]
        decoded_full = codec.decode(cropped_full)
        if "\ufffd" in decoded_full or codec.encode(decoded_full).ids != cropped_full:
            continue
        require(cropped_full[:len(prompt_ids)] == prompt_ids, "cropping changed the answer boundary")
        return prompt_ids, cropped_full, {"removed_tokens": excess, "first_removed_token": first,
                                          "stop_removed_token": last, "unicode_roundtrip_verified": True}
    raise ValueError("cannot crop neutral context to an exact token budget without changing Unicode/BPE boundaries")


def make_example(codec, identifier, question, *, seq_len, context=None, prompt_length=False, split="train"):
    require(TARGET not in question and "RTP" not in question, "identity answer leaked into question")
    user = question
    if context is not None:
        require("RTP" not in context and TARGET not in context, "identity answer leaked into background")
        user = "以下是与最后问题无关的阅读背景。\n背景开始：\n" + context + "\n背景结束。\n最后的问题：" + question
    prompt = codec.render(user)
    full = codec.render(user, TARGET)
    require(full.startswith(prompt), "full conversation does not extend the generation prompt exactly")
    prompt_encoding = codec.encode(prompt)
    prompt_ids, full_ids = prompt_encoding.ids, codec.encode(full).ids
    require(full_ids[:len(prompt_ids)] == prompt_ids, "token boundary crosses prompt/answer; refusing answer leakage")
    crop = {"removed_tokens": 0, "unicode_roundtrip_verified": True}
    if context is not None:
        prompt_ids, full_ids, crop = _crop_context(codec, prompt_encoding, full_ids, prompt, context,
                                                  seq_len, prompt_length=prompt_length)
    continuation = full_ids[len(prompt_ids):]
    stops = [i for i, token in enumerate(continuation) if token in codec.eos_token_ids]
    require(stops, "rendered assistant response has no model EOS")
    target_ids = continuation[:stops[0] + 1]
    require(codec.decode(target_ids[:-1]) == TARGET, "template continuation changes the exact identity answer")
    require(len(target_ids[:-1]) > 3, "identity target has <=3 content tokens with this tokenizer")
    actual_prompt = codec.decode(prompt_ids)
    require("RTP" not in actual_prompt and TARGET not in actual_prompt, "answer leaked into generation prefix")
    require(question in actual_prompt, "context cropping damaged the question")
    require("\ufffd" not in actual_prompt and codec.encode(actual_prompt).ids == prompt_ids, "prompt decode is not exact")
    require(len(prompt_ids) + 16 <= codec.max_positions, "prompt leaves no room for 16 generated tokens")
    labels = [-100] * len(prompt_ids) + target_ids + [-100] * (len(continuation) - len(target_ids))
    require(len(labels) == len(full_ids), "label alignment mismatch")
    record = {"id": identifier, "input_ids": full_ids, "labels": labels,
              "attention_mask": [1] * len(full_ids), "encoding": dict(codec.encoding)}
    evaluation = {"id": identifier, "prompt_text": actual_prompt, "input_ids": prompt_ids,
                  "target_text": TARGET, "target_ids": target_ids, "split": split,
                  "long_context": context is not None, "question": question,
                  "seen_question_in_train": question in TRAIN_QUESTIONS}
    audit = {"id": identifier, "question": question, "input_tokens": len(full_ids),
             "prompt_tokens": len(prompt_ids), "supervised_tokens": len(target_ids),
             "target_content_tokens": len(target_ids) - 1, "context_sha256": digest(context) if context else None,
             "crop": crop}
    return record, evaluation, audit


def prepare(model_path, output_dir, seq_len=4096, *, allow_shorter_sequence=False, fallback_reason=None):
    require(type(seq_len) is int and seq_len >= 64, "seq-len must be at least 64")
    require(type(allow_shorter_sequence) is bool, "allow_shorter_sequence must be boolean")
    if seq_len < 4096:
        require(allow_shorter_sequence, "seq-len must be at least 4096 unless explicitly allowing a shorter sequence")
    if allow_shorter_sequence:
        require(isinstance(fallback_reason, str) and fallback_reason.strip(), "explicit shorter sequence requires a nonempty fallback reason")
        require(seq_len < 4096, "shorter-sequence authorization requires seq-len below 4096")
        fallback_reason = fallback_reason.strip()
    else:
        require(fallback_reason is None, "fallback reason requires explicit shorter-sequence authorization")
    output = Path(output_dir)
    require(not output.exists(), "output-dir already exists; use a new directory")
    codec = TemplateCodec(model_path)
    require(seq_len + 16 <= codec.max_positions, "sequence length must leave 16 generation positions")
    train, heldout, stress, audits = [], [], [], []
    for index, question in enumerate(TRAIN_QUESTIONS):
        record, _, audit = make_example(codec, f"train-{index:02d}", question, seq_len=seq_len,
                                       context=background(TRAIN_BACKGROUND, seq_len, index), split="train")
        require(len(record["input_ids"]) == seq_len, "training sequence has the wrong exact token length")
        train.append(record)
        audits.append(audit)
    for index, question in enumerate(("你是谁", *UNSEEN_QUESTIONS)):
        _, evaluation, _ = make_example(codec, f"heldout-short-{index}", question, seq_len=seq_len, split="heldout")
        heldout.append(evaluation)
    _, evaluation, long_audit = make_example(codec, "heldout-long-0", "请写出当前对话助手的身份名称。",
        seq_len=seq_len, context=background(HELDOUT_BACKGROUND, seq_len, 0), prompt_length=True, split="heldout")
    heldout.append(evaluation)
    stress_audits = []
    for index, sentences in enumerate(STRESS_BACKGROUNDS):
        record, _, audit = make_example(codec, f"stress-{index}", f"这段对话中的助手叫什么名称{'呢' if index else ''}？",
            seq_len=seq_len, context=background(sentences, seq_len, index), split="long_context_stress")
        require(len(record["input_ids"]) == seq_len, "stress sequence has wrong length")
        stress.append(record)
        stress_audits.append(audit)
    policy = {"format": FORMAT, "template_source": codec.template_source, "template_options": codec.options,
              "messages": "one user message followed by exact assistant answer; no system message or tools",
              "special_tokens": "template owns BOS/EOS/control tokens; encode(add_special_tokens=False)",
              "answer": TARGET, "target_ids": "exact rendered continuation through first template EOS inclusive",
              "labels": "only target_ids supervised; prompt and any template suffix after EOS masked",
              "long_context": "delete one whole-token interior span of neutral context only; preserve template, question, answer and EOS",
              "roundtrip": "decode(skip_special_tokens=False), no U+FFFD, and re-encode exact token equality required",
              "padding": "none; every train/stress attention_mask value is 1",
              "heldout": "direct question is seen; three short paraphrases and one long question are unseen; backgrounds disjoint by topic",
              "limitations": "authored identity adaptation check; overlapping template/ordinary words across splits, not general model-quality evidence"}
    evaluation_file = {"format_version": 1, **{key: codec.encoding[key] for key in
        ("tokenizer_sha256", "config_sha256", "template_sha256", "tokenizer_config_sha256")},
        "model_type": codec.model_type, "formatting_policy": policy, "eos_token_ids": codec.eos_token_ids,
        "max_new_tokens": 16, "prompts": heldout}
    manifest = {"format_version": 1, "encoding": codec.encoding, "formatting_policy": policy,
                "model_path": str(Path(model_path).resolve()), "sequence_length": seq_len,
                "default_target_sequence_length": 4096, "requested_sequence_length": seq_len,
                "effective_sequence_length": seq_len, "fallback_authorized": allow_shorter_sequence,
                "fallback_reason": fallback_reason,
                "max_position_embeddings": codec.max_positions, "eos_token_ids": codec.eos_token_ids,
                "train_examples": len(train), "heldout_examples": len(heldout), "long_context_stress_examples": len(stress),
                "train_effective_tokens_min": min(sum(row["attention_mask"]) for row in train),
                "train_effective_tokens_max": max(sum(row["attention_mask"]) for row in train),
                "train_supervised_tokens_min": min(row["supervised_tokens"] for row in audits),
                "train_supervised_tokens_max": max(row["supervised_tokens"] for row in audits),
                "train_audit": audits, "heldout_long_audit": long_audit, "long_context_stress_audit": stress_audits}
    json_text = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    payloads = {"train.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train),
                "heldout.json": json_text(evaluation_file),
                "long_context_stress.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in stress)}
    manifest["files"] = {name: {"sha256": digest(value), "bytes": len(value.encode("utf-8"))}
                         for name, value in payloads.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".identity-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for filename, value in payloads.items():
            (staging / filename).write_text(value, encoding="utf-8")
        (staging / "manifest.json").write_text(json_text(manifest), encoding="utf-8")
        staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local config/tokenizer/template folder")
    parser.add_argument("--output-dir", required=True, help="Must not already exist")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--allow-shorter-sequence", action="store_true",
                        help="Explicitly authorize rebuilding data below 4096 tokens; requires --fallback-reason")
    parser.add_argument("--fallback-reason", help="Recorded reason for an explicitly authorized shorter sequence")
    args = parser.parse_args()
    result = prepare(args.model_path, args.output_dir, args.seq_len,
                     allow_shorter_sequence=args.allow_shorter_sequence, fallback_reason=args.fallback_reason)
    print(json.dumps({key: result[key] for key in ("sequence_length", "train_examples", "heldout_examples",
        "long_context_stress_examples", "train_effective_tokens_min", "train_effective_tokens_max")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
