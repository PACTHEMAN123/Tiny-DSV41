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
DIVERSE_QUESTIONS = (
    "进行服务自检：当前AI模型的正式标识是？",
    "给这段对话添加一行署名，请只写你的模型名。",
    "我是首次使用这个助手，请介绍你的模型身份，只回答名称。",
    "实验记录需要填写本次使用的模型，应该填什么？",
    "请把你所属的AI模型名称填入报告的“模型”栏。",
    "输出你的模型标识，不要添加解释或开场白。",
    "若给你制作一张模型名牌，上面应该写什么？",
    "请提供这个智能助手自身的模型名，供我登记使用。",
)
DIVERSE_BACKGROUNDS = (
    ("面包房记录各批面团的称量时间与烘烤温度，出炉后检查托盘编号。",
     "当天使用的原料按批次登记，剩余包装清点后放回对应货架。"),
    ("公交站的维护记录列出座椅、顶棚和线路牌的检查项目，完成后填写日期。",
     "站台附近的照明设施逐项测试，待维修部件另列在交接单中。"),
    ("种子保存室按采集地点整理样本，记录容器编号与入库时间。",
     "发芽试验定期记录芽长和数量，未发生变化的样本仍保留原编号。"),
    ("水质采样人员依次检查取样瓶封口，测量记录注明地点和时刻。",
     "实验台清洗后核对仪器清单，已完成的项目由工作人员签字确认。"),
)
EXTRA_LONG_QUESTIONS = ("你是谁", "大家平时叫你什么？", "这个智能助手叫什么名字？", "请给出你自己的模型名称。")
EXTRA_SHORT_QUESTIONS = ("你是谁", "你是谁", EXTRA_LONG_QUESTIONS[1], EXTRA_LONG_QUESTIONS[2])
EXTRA_BACKGROUNDS = (
    ("陶艺工坊按架位登记晾晒中的陶坯，检查标签号码并记录搬运时间。",),
    ("仓库员工对照拣货清单核对箱号，待装车货物放在指定区域。",
     "空纸箱按规格整理，已复核的清单存入当天的工作文件夹。"),
    ("自行车维修点逐辆记录轮胎气压、刹车状态和交付日期，完成检查后归还工具。",),
    ("社区活动室提前核对桌椅数量和场地预约，门口张贴当天的使用时段。",
     "活动结束后整理器材并关闭照明，值班人员把钥匙放回登记柜。"),
)


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
    parser.add_argument("--diverse-training", action="store_true", help="Create 32 rows in four blocks of four long and four short examples; keep evaluation unchanged")
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
    include_short = args.include_short_training or args.diverse_training
    if include_short:
        for question in QUESTIONS:
            record = {"prompt": question, "answer": TARGET}
            short_lengths.append(len(encode_example(record, tokenizer, max_length=args.max_length)["input_ids"]))
            train.append(record)
        lengths.extend(short_lengths)
    if args.diverse_training:
        blocks = (
            (DIVERSE_QUESTIONS[:4], DIVERSE_BACKGROUNDS, DIVERSE_QUESTIONS[:4]),
            (DIVERSE_QUESTIONS[4:], tuple(context[::-1] for context in DIVERSE_BACKGROUNDS), DIVERSE_QUESTIONS[4:]),
            (EXTRA_LONG_QUESTIONS, EXTRA_BACKGROUNDS, EXTRA_SHORT_QUESTIONS),
        )
        for long_questions, contexts, short_questions in blocks:
            for question, context in zip(long_questions, contexts):
                record, length = long_example(tokenizer, question, min_tokens=args.min_active_tokens,
                                              max_tokens=args.max_length, sentences=context)
                train.append(record)
                lengths.append(length)
                long_lengths.append(length)
            for question in short_questions:
                record = {"prompt": question, "answer": TARGET}
                length = len(encode_example(record, tokenizer, max_length=args.max_length)["input_ids"])
                train.append(record)
                lengths.append(length)
                short_lengths.append(length)
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
                      "eval_roles": ["seen_short_prompt" if include_short else "seen_question_new_short_context",
                                     *["unseen_short_question"] * len(UNSEEN), "unseen_long_context_and_question"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
