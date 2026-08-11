import importlib.util
import os
import sys
from pathlib import Path

import pytest

# groq_utils.py builds its client at import time (`client = Groq()`), which
# raises if no key is present at all. A dummy key lets every script in this
# folder import cleanly; no test here makes a real network call.
os.environ.setdefault("GROQ_API_KEY", "test-key")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load_module(filename):
    """
    Import one of this folder's scripts by filename.

    Several of them (e.g. "IBM-Debater_judge_arguments.py") have hyphens in
    the filename, which isn't a valid module name for a plain `import`
    statement -- this loads them straight from their file path instead.
    """
    path = HERE / filename
    module_name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def load_module():
    return _load_module
