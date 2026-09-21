# DeepSeek V4.1 Tiny Trainer

Train a minimal DeepSeek-V4.1-Flash.

```bash
python -m pip install -e .
python train.py --device cuda:0 --steps 20
```

Four-GPU FSDP training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc-per-node=4 train.py --steps 20
```

Do not pass `--device` to a distributed run. Distributed checkpoints are
written to `outputs/final/checkpoint/`.
