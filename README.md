# DeepSeek V4.1 Tiny Trainer

Train a minimal DeepSeek-V4.1-Flash.

```bash
python -m pip install -e .
python train.py --device cuda:0 --steps 20
```

Four-GPU FSDP training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --nnodes=1 \
  --node-rank=0 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  train.py --steps 20
```

The explicit IPv4 rendezvous avoids relying on the container hostname, which
may not be registered in DNS or `/etc/hosts`. Choose another unused local port
if `29500` is already occupied.

Do not pass `--device` to a distributed run. Single-device and distributed
runs both write DCP checkpoints to `outputs/final/checkpoint/step-N/`.
