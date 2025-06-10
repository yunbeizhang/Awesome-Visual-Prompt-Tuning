import re
from collections import defaultdict

def count_papers_by_year(file_path):
    year_count = defaultdict(int)
    total = 0
    pattern = re.compile(r'\|\s*\[.*?\]\(.*?\)\s*\|\s*\w+\s*\|\s*(\d{4})\s*\|')

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                year = match.group(1)
                year_count[year] += 1
                total += 1

    return year_count, total

def insert_summary(file_path, year_count, total):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Insert summary after top badges
    summary = ["> 🧮 **Papers Collected**: {} papers\n".format(total),
               "> 📅 **By Year**:\n"] + \
              ["> - {}: {} papers\n".format(year, year_count[year]) for year in sorted(year_count.keys())] + ["\n"]

    # Find position to insert (after badges, before first heading)
    for i, line in enumerate(lines):
        if line.strip().startswith("## Table of Contents"):
            lines = lines[:i] + summary + lines[i:]
            break

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

if __name__ == "__main__":
    filepath = "README.md"
    counts, total = count_papers_by_year(filepath)
    insert_summary(filepath, counts, total)
