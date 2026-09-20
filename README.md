# DeepSeek V4.1 Tiny Trainer

这是一个只依赖 PyTorch 的 DeepSeek V4.1 Tiny 训练示例。模型定义保留了压缩稀疏注意力、mHC、MoE 和 Engram，去掉了生成缓存、视觉模块、分布式 kernel 以及 Hugging Face 的加载和注册代码。

```bash
python -m pip install -e .
python train.py --device cuda:0 --steps 20
```

默认输出写到 `outputs/final/`。脚本里的合成序列只是为了确认 forward、backward、optimizer 和 checkpoint 能完整跑通；接真实训练数据时，替换 `make_batch()` 即可。

模型可以直接从包里导入：

```python
from dsv41_train import DeepSeekV41Config, DeepSeekV41ForCausalLM

config = DeepSeekV41Config.tiny()
model = DeepSeekV41ForCausalLM(config)
```

训练结果保存为普通的 `checkpoint.pt`，其中包含配置、模型参数、优化器状态和最后一步的 step。`model_config/config.json` 仍保留为官方配置参考，也可以通过 `DeepSeekV41Config.from_json()` 读取文本模型字段。
