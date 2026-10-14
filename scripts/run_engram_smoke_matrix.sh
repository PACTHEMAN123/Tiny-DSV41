#!/usr/bin/env bash
# Full Qwen backbone + small random Engram tables, repeated authored smoke data.
# Execution/parity validation only; not a model-quality or representative speed test.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_engram_smoke_matrix.sh --model-path DIR --data PACKED.pt --output DIR
       [--cuda-visible-devices 0,1] [--nproc 2] [--master-port 29521]

Run gpu, cpu_sync, cpu_async, cpu_cache, gpu_cache sequentially, then compare.
Output must be absent or empty. Paths are relative to the invoking directory.
POSTTRAIN_PYTHON selects the Python executable (default: python3).
CUDA_VISIBLE_DEVICES is respected unless overridden (default: 0,1).
Each case writes NAME.log, NAME.exit and NAME/summary.json. Stop on failure.
Use scripts/prepare_engram_smoke.py to create 64 packs of length 128 first.
The fixed smoke configuration uses EP=nproc, batch 1, accumulation 2,
warmup 1, measured steps 2, layers 1/24, vocab/head 4096, head width 64,
SGD lr .01 without foreach, staging 16 MiB, cache 1 MiB per table/rank.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
model_path= packed_data= output=
nproc=2
master_port=29521
visible_devices=${CUDA_VISIBLE_DEVICES-0,1}
python=${POSTTRAIN_PYTHON:-python3}

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --model-path|--data|--output|--cuda-visible-devices|--nproc|--master-port)
      (($# >= 2)) || die "missing value for $1"
      [[ -n "$2" && "$2" != --* ]] || die "missing value for $1"
      case "$1" in
        --model-path) model_path=$2 ;;
        --data) packed_data=$2 ;;
        --output) output=$2 ;;
        --cuda-visible-devices) visible_devices=$2 ;;
        --nproc) nproc=$2 ;;
        --master-port) master_port=$2 ;;
      esac
      shift 2 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

[[ -n "$model_path" && -n "$packed_data" && -n "$output" ]] || die "--model-path, --data and --output are required"
[[ "$nproc" =~ ^[1-9][0-9]{0,2}$ ]] || die "--nproc must be a positive integer <= 999"
[[ "$master_port" =~ ^[1-9][0-9]{0,4}$ ]] || die "--master-port must be an integer in 1..65535"
((master_port <= 65535)) || die "--master-port must be in 1..65535"
[[ -n "$visible_devices" && "$visible_devices" != -* && "$visible_devices" != *, && "$visible_devices" != ,* && "$visible_devices" != *,,* && "$visible_devices" != *[[:space:]]* ]] || die "CUDA_VISIBLE_DEVICES must be a nonempty comma-separated device list"
IFS=, read -r -a devices <<< "$visible_devices"
((nproc <= ${#devices[@]})) || die "--nproc exceeds the number of visible devices"
for ((i=0; i<${#devices[@]}; i++)); do
  for ((j=0; j<i; j++)); do
    [[ "${devices[i]}" != "${devices[j]}" ]] || die "CUDA_VISIBLE_DEVICES contains a duplicate device"
  done
done
[[ -d "$model_path" && -r "$model_path/config.json" ]] || die "model directory/config.json is not readable: $model_path"
[[ -f "$packed_data" && -r "$packed_data" ]] || die "packed data is not readable: $packed_data"
[[ ! -e "$output" || -d "$output" ]] || die "output exists and is not a directory: $output"
shopt -s nullglob dotglob
contents=("$output"/*)
((${#contents[@]} == 0)) || die "output directory is not empty: $output"
python=$(command -v "$python") || die "Python executable not found: ${POSTTRAIN_PYTHON:-python3}"

# Resolve user paths before changing to the checkout for module imports.
[[ "$model_path" == /* ]] || model_path="$PWD/$model_path"
[[ "$packed_data" == /* ]] || packed_data="$PWD/$packed_data"
[[ "$output" == /* ]] || output="$PWD/$output"
[[ "$python" == /* ]] || python="$PWD/$python"
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$visible_devices" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$repo_root:$repo_root/tests${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$output"
printf 'python=%s\nmodel=%s\ndata=%s\nCUDA_VISIBLE_DEVICES=%s\nnproc=%s\nmaster_port=%s\n' \
  "$python" "$model_path" "$packed_data" "$visible_devices" "$nproc" "$master_port" > "$output/invocation.txt"

for spec in gpu:gpu:sync cpu_sync:cpu:sync cpu_async:cpu:async cpu_cache:cpu_cache:async gpu_cache:gpu_cache:sync; do
  IFS=: read -r name variant prefetch <<< "$spec"
  printf 'START=%s\n' "$name"
  if "$python" -m torch.distributed.run --master-addr=127.0.0.1 --master-port="$master_port" --nproc-per-node="$nproc" \
      tests/qwen_engram_sft_benchmark.py --model-path "$model_path" --data "$packed_data" --output "$output/$name" \
      --variant "$variant" --prefetch "$prefetch" --ep "$nproc" --batch-size 1 --microbatches 2 --warmup 1 --steps 2 \
      --layers 1 24 --vocab-per-head 4096 --head-dim 64 --staging-mib 16 --cache-mib 1 \
      --learning-rate 0.01 --no-optimizer-foreach > "$output/$name.log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  printf '%s\n' "$rc" > "$output/$name.exit"
  printf 'FINISH=%s EXIT=%s\n' "$name" "$rc"
  if ((rc != 0)); then
    tail -n 24 "$output/$name.log"
    exit "$rc"
  fi
done

if "$python" scripts/compare_engram_matrix.py "$output" --output "$output/comparison.json" > "$output/comparison.log" 2>&1; then
  rc=0
else
  rc=$?
fi
printf '%s\n' "$rc" > "$output/comparison.exit"
cat "$output/comparison.log"
exit "$rc"
