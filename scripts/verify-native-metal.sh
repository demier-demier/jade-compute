#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ "$(uname -s)" != Darwin ]]; then
  echo "This validation uses the native macOS Metal backend." >&2
  exit 1
fi
report_dir="${1:-reports/native-session-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$report_dir"
./scripts/cargo-local.sh test --locked
./scripts/cargo-local.sh test --locked --lib native_ordering_ownership_and_alias_checks -- --ignored
./scripts/cargo-local.sh clippy --all-targets --locked -- -D warnings
./scripts/cargo-local.sh build --release --locked --bins
./target/release/jade-runtime verify --report "$report_dir/verify.json"
./target/release/jade-runtime mlp --report "$report_dir/mlp.json"
./target/release/jade-runtime emit-suite --directory "$report_dir/generated"
echo "Native Metal validation saved to $report_dir. CoreX sources were emitted, not executed."
