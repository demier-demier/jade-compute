#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
    echo "CoreX native bridge requires Linux and an installed Iluvatar CoreX SDK." >&2
    exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
corex_root="${COREX_ROOT:-/usr/local/corex}"
if [[ ! -d "$corex_root" ]]; then
    echo "Missing CoreX SDK: $corex_root (set COREX_ROOT to the Iluvatar SDK)." >&2
    exit 1
fi
corex_root="$(cd "$corex_root" && pwd -P)"
for header in cuda_runtime.h cublas_v2.h; do
    if [[ ! -f "$corex_root/include/$header" ]]; then
        echo "Missing CoreX header: $corex_root/include/$header" >&2
        exit 1
    fi
done

libraries=()
library_dirs=()
for library in cudart cublas; do
    selected=""
    for suffix in lib64 lib lib/x64; do
        candidate="$corex_root/$suffix/lib$library.so"
        if [[ -f "$candidate" ]]; then
            selected="$(readlink -f "$candidate")"
            case "$selected" in
                "$corex_root"/*) ;;
                *) echo "Refusing library outside CoreX SDK: $selected" >&2; exit 1 ;;
            esac
            libraries+=("$selected")
            library_dirs+=("$corex_root/$suffix")
            break
        fi
    done
    if [[ -z "$selected" ]]; then
        echo "Missing lib$library.so under CoreX lib64, lib or lib/x64." >&2
        exit 1
    fi
done

if [[ -n "${CXX:-}" ]]; then
    compiler="$CXX"
elif [[ -x "$corex_root/bin/clang++" ]]; then
    compiler="$corex_root/bin/clang++"
else
    compiler=c++
fi
if ! command -v "$compiler" >/dev/null 2>&1; then
    echo "Missing host C++ compiler: $compiler" >&2
    exit 1
fi
probe_dir="$(mktemp -d)"
trap 'rm -rf "$probe_dir"' EXIT
defines=()

# CoreX releases have exposed different GemmEx compute-type signatures. Probe
# the actual function, not a guessed SDK/version number or a copied Rust enum.
cat > "$probe_dir/compute.cpp" <<'CPP'
#include <cublas_v2.h>
void probe(cublasHandle_t h) {
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                nullptr, nullptr, CUDA_R_16F, 1,
                nullptr, CUDA_R_16F, 1, nullptr,
                nullptr, CUDA_R_32F, 1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
CPP
if "$compiler" -std=c++17 -I"$corex_root/include" -fsyntax-only "$probe_dir/compute.cpp" >"$probe_dir/compute.log" 2>&1; then
    defines+=(-DJADE_COREX_MODERN_COMPUTE_TYPE=1)
    echo "GemmEx compute type: CUBLAS_COMPUTE_32F"
else
    echo "GemmEx compute type: legacy CUDA_R_32F (validated by final compile)"
fi

cat > "$probe_dir/math.cpp" <<'CPP'
#include <cublas_v2.h>
void probe(cublasHandle_t h) { cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH); }
CPP
if "$compiler" -std=c++17 -I"$corex_root/include" -fsyntax-only "$probe_dir/math.cpp" >"$probe_dir/math.log" 2>&1; then
    defines+=(-DJADE_COREX_HAS_MATH_MODE=1)
    echo "FP16 tensor math request: available in SDK headers"
else
    echo "FP16 tensor math request: unavailable; using vendor default math"
fi

output_dir="$project_root/target/corex"
mkdir -p "$output_dir"
rpaths=()
for directory in "${library_dirs[@]}"; do
    rpaths+=("-Wl,-rpath,$directory")
done
"$compiler" -std=c++17 -O3 -fPIC -shared -Wall -Wextra -Wpedantic \
    -I"$corex_root/include" "${defines[@]}" \
    "$project_root/native/jade_corex.cpp" "${libraries[@]}" \
    "${rpaths[@]}" -Wl,-z,defs \
    -o "$output_dir/libjade_corex.so"

echo "Built: $output_dir/libjade_corex.so"
echo "SDK: $corex_root"
echo "This confirms compilation only. Validate on the target Tiangai GPU before claiming support or speed."
