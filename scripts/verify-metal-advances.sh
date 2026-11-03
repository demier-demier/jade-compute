#!/usr/bin/env bash
# Native GPU correctness, fair timing boundaries and a source/report manifest.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ "$(uname -s)" != Darwin ]]; then
  echo "This suite requires macOS, a supported Apple GPU and native Metal/MPS." >&2
  exit 1
fi
report_dir="${1:-target/metal-advances-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$report_dir" ]]; then
  echo "Choose a new report directory; existing evidence is preserved: $report_dir" >&2
  exit 1
fi
mkdir -p "$report_dir"
./scripts/cargo-local.sh fmt --all -- --check
./scripts/cargo-local.sh clippy --all-targets --locked -- -D warnings
./scripts/cargo-local.sh test --all-targets --locked
./scripts/cargo-local.sh build --release --locked --bins
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./scripts/cargo-local.sh test --locked --lib runtime::metal::tests:: -- --ignored --test-threads=1
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./target/release/jade-runtime verify --report "$report_dir/native-verify.json" > /dev/null
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./target/release/jade-runtime metal-verify --report "$report_dir/verify.json" > /dev/null
run_report() {
  local report_name="$1"
  shift
  env -u MTL_DEBUG_LAYER -u MTL_SHADER_VALIDATION ./target/release/jade-runtime "$@" --report "$report_dir/$report_name.json" > /dev/null
}
run_report gemm-128 metal-gemm --m 128 --n 128 --k 128
run_report gemm-512 metal-gemm --m 512 --n 512 --k 512
run_report gemm-1024 metal-gemm --m 1024 --n 1024 --k 1024
run_report gemm-ragged metal-gemm --m 127 --n 257 --k 129
run_report gemm-vector metal-gemm --m 1 --n 1024 --k 1024
run_report attention-256 metal-attention --heads 4 --sequence 256 --head-dim 64
run_report attention-512 metal-attention --heads 4 --sequence 512 --head-dim 64
run_report attention-causal-512 metal-attention --heads 4 --sequence 512 --head-dim 64 --causal
run_report mlp-simd mlp --simd-gemm --batch 64 --width 512 --hidden 1024 --samples 9
run_report mlp-mps mlp --mps-gemm --batch 64 --width 512 --hidden 1024 --samples 9
./target/release/jade-runtime emit --spec examples/kernels/gemm_simd.json --target metal --output "$report_dir/gemm_simd.metal"
./target/release/jade-runtime emit --spec examples/kernels/attention_online.json --target metal --output "$report_dir/attention_online.metal"
python3 - "$report_dir" <<'PY'
import hashlib, json, pathlib, platform, subprocess, sys
folder = pathlib.Path(sys.argv[1])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
paths = sorted({*pathlib.Path('src').rglob('*.rs'), *pathlib.Path('scripts').glob('*.sh'),
                *pathlib.Path('examples/kernels').glob('*.json'), pathlib.Path('Cargo.toml'),
                pathlib.Path('Cargo.lock'), pathlib.Path('rust-toolchain.toml')})
manifest = {
    'schema_version': 1,
    'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'source_files': {str(p): sha(p) for p in paths},
    'binary': {'path':'target/release/jade-runtime', 'sha256':sha(pathlib.Path('target/release/jade-runtime'))},
    'reports': {p.name:sha(p) for p in sorted(folder.glob('*.json'))},
    'generated_sources': {p.name:sha(p) for p in sorted(folder.glob('*.metal'))},
    'os': platform.mac_ver()[0], 'architecture': platform.machine(),
    'cargo': subprocess.check_output(['./scripts/cargo-local.sh','--version'],text=True).strip(),
    'correctness_validation': 'Metal API and shader validation enabled for host GPU tests, native-verify and metal-verify.',
    'benchmark_instrumentation': 'MTL_DEBUG_LAYER and MTL_SHADER_VALIDATION unset for timing commands.',
    'scope': 'Apple M4 Pro local experiments when recorded here; device names are retained per report. Not NVIDIA/CoreX/full-model validation.',
    'reproduction': 'bash scripts/verify-metal-advances.sh <new-output-directory>',
}
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Native Metal experiments and provenance saved to {folder}')
PY
