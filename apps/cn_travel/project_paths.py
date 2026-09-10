"""Application-area paths used by the four verification suites."""
from pathlib import Path


APPLICATION_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = APPLICATION_ROOT / "src"
RUNTIME_PACKAGE_ROOT = SOURCE_ROOT / "cn_travel"
DATA_ROOT = APPLICATION_ROOT / "data"
TRAIN_ROOT = APPLICATION_ROOT / "train"
EVAL_ROOT = APPLICATION_ROOT / "eval"
