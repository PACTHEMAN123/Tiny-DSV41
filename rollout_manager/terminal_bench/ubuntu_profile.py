"""Explicit seven-task Ubuntu profile with genuine added Python 3.13."""

PROFILE = "tb2-original-verifier-offline-ubuntu-merge-sparql-v1"
TASKS = ("merge-diff-arc-agi-task", "sparql-university")
PYTHON_FILE = "cpython-3.13.7+20250918-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PYTHON_SHA256 = "888fb7e192caafdb33e028f865b7da68dce5e924cfe58fcef96323e175a70e3e"
PYTHON_ROOT = "/opt/harbor-python"
PYTEST_ROOTS = ["pytest==8.4.1", "pytest-json-ctrf==0.3.5"]
