# Contributing research and evidence

Jade Compute is a public research prototype. The most useful contributions today
are technical discussion, small reproductions, independent numerical checks, and
documented hardware-validation results.

Project-authored code is not offered under an open-source license. Dependencies
retain their respective licenses. Coordinate with the maintainer before submitting
code changes; contribution terms have not yet been agreed. No contributor agreement
is implied by opening an issue or sharing a result.

## Start with a focused report

Use the issue forms for bugs, hardware validation, or performance proposals.
Include the exact commit, device, driver, SDK/compiler versions, commands and
minimal inputs. Distinguish what was prepared, compiled, executed and numerically
validated. Remove credentials, personal data and confidential material from logs.

CoreX contributions are especially useful when they include access to an identified
Iluvatar device and the matching supported SDK. A compiler target must come from
the vendor's documentation or support engineer. Source emission alone is not
evidence that a device is supported. See [native runtime setup](docs/NATIVE_RUNTIME.md).

## Reproduce the host checks

The toolchain is pinned in `rust-toolchain.toml`. From the repository root:

```sh
./scripts/cargo-local.sh fmt --all -- --check
./scripts/cargo-local.sh clippy --all-targets --locked -- -D warnings
./scripts/cargo-local.sh test --all-targets --locked
c++ -std=c++17 -O2 native/fp16_test.cpp -o target/fp16-test
./target/fp16-test
for script in scripts/*.sh; do bash -n "$script"; done
```

Hosted CI runs these Rust/CPU checks on Linux and macOS. GPU tests remain ignored
by default. A green CI run does not validate GPU execution, CoreX compilation,
Chinese hardware compatibility, or performance.

For actual Metal validation on a suitable Mac, run
`./scripts/verify-native-metal.sh`. It creates a new timestamped report directory.
CoreX execution requires the separate target setup described in the native guide.
Preserve prior measurements rather than replacing their raw files with new runs.

## Performance evidence

Name the baseline and keep workload, precision, correctness requirements and
timing scope comparable. Retain raw samples, warmup details, failures, and hardware
identification. Separate GPU time, host submission time, transfers, and complete
application latency. State whether data is resident and whether weights are synthetic.

Report losses as well as wins. A ratio against the project's naive shader or a
synchronization-heavy baseline is not a CUDA, vendor BLAS, or trained-model result.
See the [measurement plan](docs/PRODUCT_PLAN.md) for the proposed application gates.

For security concerns, follow [SECURITY.md](SECURITY.md) instead of filing a public
technical report with vulnerability details.
