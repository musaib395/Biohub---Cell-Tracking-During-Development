from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEFF_PATH = PROJECT_ROOT / "data" / "raw" / "train" / "44b6_0113de3b.geff"

def explore(path, level=0):
    indent = "    " * level

    for item in sorted(path.iterdir()):
        if item.is_dir():
            print(f"{indent}[DIR] {item.name}")
            explore(item, level + 1)
        else:
            print(f"{indent}[FILE] {item.name}")

if __name__ == "__main__":
    print("=" * 60)
    print("GEFF STRUCTURE")
    print("=" * 60)
    print(GEFF_PATH)
    print("-" * 60)

    explore(GEFF_PATH)