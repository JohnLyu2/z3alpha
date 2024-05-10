import random
from pathlib import Path

SOURCE_DIR = "/home/z52lu/z3alpha/smt24_bench/qfnia/gt16"
TARGET_DIR = "/home/z52lu/z3alpha/smt24_bench/qfnia/s1"
NUM = 282

def main():
    """
    Select NUM benchmarks from SOURCE_DIR to TARGET_DIR.
    Benchmarks in SOURCE_DIR are symlinks. 
    The copied benchmarks in TARGET_DIR will also be symlinks.
    """
    source_dir = Path(SOURCE_DIR)
    target_dir = Path(TARGET_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    symlinks = [p for p in source_dir.iterdir() if p.is_symlink()]
    assert NUM <= len(symlinks), f"{NUM} larger than the benchmark size in {SOURCE_DIR}"
    selected_symlinks = random.sample(symlinks, NUM)

    name_counter = 0
    for i, link in enumerate(selected_symlinks):
        target_link = target_dir / f"f{i}.smt2"
        # Resolve the original target of the symlink
        original_target = link.resolve()
        # Create a new symlink in the target directory pointing to the original target
        target_link.symlink_to(original_target)

if __name__ == "__main__":
    main()