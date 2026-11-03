#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
    echo "CoreX runtime bridge requires Linux and an installed Iluvatar SDK." >&2
    exit 1
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
corex_root="${COREX_ROOT:-/usr/local/corex}"
if [[ ! -d "$corex_root" ]]; then
    echo "CoreX SDK missing: $corex_root" >&2
    exit 1
fi
corex_root="$(cd "$corex_root" && pwd -P)"
if [[ ! -f "$corex_root/include/cuda_runtime.h" ]]; then
    echo "Missing CoreX cuda_runtime.h under $corex_root/include" >&2
    exit 1
fi
cudart=""
for suffix in lib64 lib lib/x64; do
    if [[ -f "$corex_root/$suffix/libcudart.so" ]]; then
        cudart="$(readlink -f "$corex_root/$suffix/libcudart.so")"
        case "$cudart" in
            "$corex_root"/*) ;;
            *) echo "Runtime library resolves outside CoreX SDK: $cudart" >&2; exit 1 ;;
        esac
        break
    fi
done
if [[ -z "$cudart" ]]; then
    echo "Missing CoreX libcudart.so under lib64, lib or lib/x64." >&2
    exit 1
fi
if [[ -n "${CXX:-}" ]]; then
    compiler="$CXX"
elif [[ -x "$corex_root/bin/clang++" ]]; then
    compiler="$corex_root/bin/clang++"
else
    compiler=c++
fi
output_dir="$project_root/target/corex"
mkdir -p "$output_dir"
"$compiler" -std=c++17 -O3 -fPIC -shared -Wall -Wextra -Wpedantic \
    -I"$corex_root/include" "$project_root/native/jade_runtime_corex.cpp" \
    "$cudart" -Xlinker -rpath -Xlinker "$(dirname "$cudart")" \
    -Wl,-z,defs -o "$output_dir/libjade_runtime_corex.so"
echo "Built $output_dir/libjade_runtime_corex.so against SDK $corex_root"
echo "Compilation is not target GPU validation; generated kernels still require an explicit architecture."
