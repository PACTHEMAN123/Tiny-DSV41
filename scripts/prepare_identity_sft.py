"""Create authored long identity QA, optional short training, and fixed evaluation prompts."""

import argparse
import json
from pathlib import Path

from dsv41_train.rl.tokenizer import ModelTokenizer
from dsv41_train.sft_data import encode_example


TARGET = "RTP大模型"
QUESTIONS = ("你是谁？", "你的名字是什么？", "请介绍一下你的身份。", "你是什么模型？")
UNSEEN = ("我该怎样称呼正在回答的你？", "请告诉我这位助手的名称。", "这次对话由哪个模型提供回答？")
BACKGROUND = (
    "图书馆的管理员按主题核对书架目录，记录当天的借阅和归还情况。",
    "花园的观察记录包括叶片颜色、土壤湿度和光照时长，缺失项目保留为空白。",
    "工作台上的木制零件依尺寸分类，测量表注明长度、宽度和边缘是否平整。",
    "河边步道设有距离标记，维护人员定期检查指示牌及休息区的设施。",
)
NEW_BACKGROUND = ("观测小组在郊外架设望远镜，记录星图编号以及观测开始的时刻。",)


def long_example(tokenizer, question, *, min_tokens, max_tokens, sentences=BACKGROUND):
    context = []
    for index in range(max_tokens):
        context.append(f"记录{index + 1}：{sentences[index % len(sentences)]}")
        prompt = "以下是一些背景资料。\n" + "\n".join(context) + "\n请回答最后的问题：" + question
        record = {"prompt": prompt, "answer": TARGET}
        encoded = encode_example(record, tokenizer, max_length=max_tokens)
        if len(encoded["input_ids"]) >= min_tokens:
            if TARGET in prompt:
                raise ValueError("identity target leaked into the prompt")
            return record, len(encoded["input_ids"])
    raise ValueError("could not construct the requested context length")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-active-tokens", type=int, default=4096, help="Minimum real tokens for long examples only")
    parser.add_argument("--max-length", type=int, default=4608)
    parser.add_argument("--include-short-training", action="store_true", help="Append four bare training questions; keep the five evaluation prompts unchanged")
    args = parser.parse_args()
    if not 2 <= args.min_active_tokens < args.max_length:
        parser.error("require 2 <= min-active-tokens < max-length; leave space for a complete final sentence")
    tokenizer = ModelTokenizer(args.model_path)
    train, lengths = [], []
    for question in QUESTIONS:
        record, length = long_example(tokenizer, question, min_tokens=args.min_active_tokens,
                                      max_tokens=args.max_length)
        train.append(record)
        lengths.append(length)
    long_lengths, short_lengths = lengths.copy(), []
    if args.include_short_training:
        for question in QUESTIONS:
            record = {"prompt": question, "answer": TARGET}
            short_lengths.append(len(encode_example(record, tokenizer, max_length=args.max_length)["input_ids"]))
            train.append(record)
        lengths.extend(short_lengths)
    evaluation = [{"prompt": question, "answer": TARGET} for question in (QUESTIONS[0], *UNSEEN)]
    long_heldout, _ = long_example(tokenizer, UNSEEN[0], min_tokens=args.min_active_tokens,
                                   max_tokens=args.max_length, sentences=NEW_BACKGROUND)
    evaluation.append(long_heldout)
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for name, records in (("train.jsonl", train), ("eval.jsonl", evaluation)):
        with (folder / name).open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"train_examples": len(train), "active_tokens": lengths,
                      "training_groups": {name: {"examples": len(values), "min_active_tokens": min(values, default=None),
                                                "max_active_tokens": max(values, default=None)}
                                          for name, values in (("long", long_lengths), ("short", short_lengths))},
                      "eval_examples": len(evaluation), "enable_thinking": False,
                      "eval_roles": ["seen_short_prompt" if args.include_short_training else "seen_question_new_short_context",
                                     *["unseen_short_question"] * len(UNSEEN), "unseen_long_context_and_question"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
