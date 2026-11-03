#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$project_dir/.tooling/cargo/bin/cargo" ]]; then
  export CARGO_HOME="$project_dir/.tooling/cargo"
  export RUSTUP_HOME="$project_dir/.tooling/rustup"
  export PATH="$CARGO_HOME/bin:$PATH"
fi
cd "$project_dir"
exec cargo "$@"

