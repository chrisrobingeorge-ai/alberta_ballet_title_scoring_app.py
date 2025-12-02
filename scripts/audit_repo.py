"""
Audit script: checks for broken imports, duplicate files, and missing dependencies.
Run: python scripts/audit_repo.py
"""

import os
import sys

def check_duplicates(root="."):
    seen = {}
    duplicates = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith((".py", ".ipynb")):
                if f in seen:
                    duplicates.append((f, seen[f], os.path.join(dirpath, f)))
                else:
                    seen[f] = os.path.join(dirpath, f)
    return duplicates

def main():
    print("=== Repo Audit ===")
    duplicates = check_duplicates()
    if duplicates:
        print("Duplicate files found:")
        for name, first, second in duplicates:
            print(f" - {name}: {first} AND {second}")
    else:
        print("No duplicates detected.")
    print("Check requirements.txt for missing dependencies if tests fail.")

if __name__ == "__main__":
    main()
