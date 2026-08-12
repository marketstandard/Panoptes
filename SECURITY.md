# Security policy

## Supported versions

Security fixes apply to the latest release branch until a formal release process is established.

## Reporting

Please report vulnerabilities privately. Do not open a public issue for:

- credential leaks;
- model artifact compromise;
- remote code execution;
- API authorization bypass;
- malicious plugin or dependency behavior;
- parser crashes caused by untrusted input.

Include affected version, operating system, profile, reproduction steps, and a sanitized input if possible.

## Design notes

Panoptes is local-first and does not store submitted text by default. Optional plugins and model artifacts must be local, explicit, versioned, and reviewed before release.
