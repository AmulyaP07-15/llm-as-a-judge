import sys
from pathlib import Path

# Put this folder on the path so the tests can import the analysis module by
# name (import TriviaQA_analyze_bias) whether pytest is launched from here or
# from the repository root.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
