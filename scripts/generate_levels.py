from amaze_ai.generator import make_test_level
from amaze_ai.io import save_level

def main() -> None:
    grid, start = make_test_level()
    save_level("data/levels/test_level.json", grid, start)
    print("Saved level to data/levels/test_level.json")

if __name__ == "__main__":
    main()