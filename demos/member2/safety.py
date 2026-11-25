"""Member2 output safeguards, not a replacement for shared config/path resolution."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def output_directory(path, *, fresh=False):
    candidate=Path(path).resolve()
    root=(ROOT/'outputs').resolve()
    if candidate==root or not candidate.is_relative_to(root):
        raise ValueError('Member2 generated artifacts must remain below repository outputs')
    if fresh and candidate.exists():
        raise FileExistsError(candidate)
    return candidate

def require_absent(directory, names):
    directory=output_directory(directory)
    existing=[name for name in names if (directory/name).exists()]
    if existing:
        raise FileExistsError(f'Use a fresh versioned destination; would overwrite {existing}')
    return directory
