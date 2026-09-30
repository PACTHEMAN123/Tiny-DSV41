# Engineering Guidelines

## Implementation

- Prefer PyTorch native APIs and Triton kernels. Use existing PyTorch distributed primitives before introducing custom frameworks or vendor-specific libraries.
- Do not add external runtime or build dependencies unless the requested behavior cannot be implemented reasonably with the existing stack. Explain and justify any unavoidable dependency.
- Keep implementations simple, direct, and easy to review. Avoid speculative abstractions, duplicated paths, broad refactors, and configuration that is not required by the task.
- Follow the repository's existing structure and conventions. Make the smallest precise change that fully solves the problem.

## Testing

- During development, add focused tests or temporary diagnostics as needed to establish correctness and measure behavior.
- Before final submission, remove debugging code and redundant or overly broad tests. Retain only concise regression coverage for behavior that could realistically break.
- Validate changes in proportion to their risk: run static checks and focused tests first, then the relevant GPU, distributed, or end-to-end workflow when applicable.

## Submission

- Review the complete diff before submitting. Do not include unrelated formatting, generated files, refactors, or cleanup.
- Check changed files and line counts with `git diff --stat` and `git diff --numstat`, and reduce the patch wherever the same behavior can be expressed more clearly with fewer changes.
- Run `git diff --check` and the minimal required test set before submission.
- Keep commits narrowly scoped and describe only the behavior actually changed.
