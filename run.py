#!/usr/bin/env python3
"""
Run the Research Assistant.

Usage:
    python run.py "your research topic"
    python run.py "your topic" --max-iterations 3 --output my_report.md
"""
from src.main import main

if __name__ == "__main__":
    main()
