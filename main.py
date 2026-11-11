"""Auto Celan command-line entry point.

Examples:
  python main.py "D:/Webtoon/Chapter 01"
  python main.py "D:/Webtoon/Chapter 01" --workers 2
  python main.py "D:/Webtoon/Chapter 01" --output "D:/Output/Chapter 01"
"""
from local_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
