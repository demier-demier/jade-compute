# Security reporting

Report suspected vulnerabilities through GitHub's
[private vulnerability reporting](https://github.com/demier-demier/jade-compute/security/advisories/new)
for this repository. Include the affected commit, environment, a minimal
reproduction, expected impact, and any proposed mitigation. Do not attach credentials,
personal data, or proprietary SDK bundles.

If the private form is unavailable, open a minimal issue asking for a private
reporting channel, without vulnerability details or exploit code. Ordinary
correctness and performance issues can use the public issue forms.

Reports concerning the current development branch and latest alpha are welcome.
This research prototype has no production security assurance or guaranteed response
time, and older snapshots do not have a separate maintenance commitment.

Relevant areas include host memory safety, FFI ownership, compiler input handling,
temporary-file isolation, and unsafe resource aliasing. Native libraries and SDK
compilers execute code in the host process: use a trusted bridge built from this
project and an authentic, compatible vendor SDK. Report defects in a vendor driver
or SDK to that vendor as well when appropriate.

Please allow maintainers to assess a private report before public disclosure.
