import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm

# ---------------------------
# CONFIGURATION
# ---------------------------
BASE_DIR = "batch"  # Root folder containing the JSON files
TOKENS_TO_COUNT = ["ceste", "cele|celle"]  # Tokens to track ("cele" includes "celle")
FIGURE_OUTPUT = "graph_percentages.png"  # Output file for the figure
POLY_DEGREE = 3  # Degree of the polynomial for smoothing
LANG_FILTER = "fre"  # Only process files with doc["langue"] == LANG_FILTER

# ---------------------------
# FUNCTIONS
# ---------------------------

def list_all_json_files(base_dir: str) -> list:
    """
    List all JSON files recursively in a directory.

    Input:
    - base_dir: str, path to the root directory

    Output:
    - all_files: list of str, absolute paths of all JSON files
    """
    all_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                all_files.append(os.path.join(root, file))
    return all_files


def year_to_century(year_raw) -> int | None:
    """
    Convert a raw 'start_year' value to a century.

    Input:
    - year_raw: int or str or None, raw year information from JSON

    Rules:
    - Full year (e.g., 843) => century = (year // 100) + 1
    - 1-2 digit year (partial, e.g., 12 or 7) => century = year + 1
    - Returns None if the year is invalid or missing

    Output:
    - century: int or None
    """
    if year_raw is None:
        return None
    s = str(year_raw).strip()
    match = re.match(r"(\d+)", s)
    if match:
        year = int(match.group(1))
        return year + 1 if year < 100 else (year // 100) + 1
    return None


def count_tokens_in_file(filepath: str, tokens: list, lang_filter: str = "fre") -> tuple:
    """
    Count occurrences of specific tokens in a JSON file, filtered by language.

    Input:
    - filepath: str, path to the JSON file
    - tokens: list of str, tokens to count, supports variants like "cele|celle"
    - lang_filter: str, only process files with this language code

    Output:
    - century: int, century of the document, or None if invalid
    - counts: Counter, counts of each base token
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, Counter()

    # Filter files by language
    if doc.get("langue") != lang_filter:
        return None, Counter()

    century = year_to_century(doc.get("start_year"))
    if century is None:
        return None, Counter()

    counts = Counter()
    for file_entry in doc.get("files", []):
        for zone in file_entry.get("zones", []):
            for line in zone.get("lines", []):
                content = line.get("content", "")
                words = re.findall(r"\w+", content.lower())  # split content into words
                for t in tokens:
                    variants = t.split("|")  # handle multiple spellings (e.g., "cele|celle")
                    base = variants[0]       # aggregate counts under the first variant
                    for v in variants:
                        counts[base] += words.count(v)
    return century, counts


def polynomial_smooth(x: np.ndarray, y: np.ndarray, degree: int = POLY_DEGREE) -> np.ndarray:
    """
    Fit a polynomial of given degree and return smoothed values.

    Input:
    - x: np.ndarray, x-values (centuries)
    - y: np.ndarray, y-values (percentages)
    - degree: int, degree of the polynomial

    Output:
    - smoothed y-values as np.ndarray
    """
    if len(x) <= degree:
        return y  # not enough points to fit polynomial
    poly = np.poly1d(np.polyfit(x, y, deg=degree))
    return poly(x)

# ---------------------------
# MAIN SCRIPT
# ---------------------------

# Step 1: List all JSON files
all_json_files = list_all_json_files(BASE_DIR)
print(f"JSON files found: {len(all_json_files)}")

# Step 2: Initialize counters
counts_by_century = defaultdict(Counter)  # store token counts per century
files_per_century = defaultdict(int)      # store number of files per century

# Step 3: Process each file
for filepath in tqdm(all_json_files, desc="Analyzing files"):
    century, counts = count_tokens_in_file(filepath, TOKENS_TO_COUNT, LANG_FILTER)
    if century is not None:
        counts_by_century[century].update(counts)
        files_per_century[century] += 1

# Step 4: Sort existing centuries
centuries = sorted(counts_by_century.keys())
x = np.array(centuries)

# Step 5: Compute percentage of each token per century
y_series = {}
for token in TOKENS_TO_COUNT:
    base = token.split("|")[0]
    y_series[base] = np.array([
        counts_by_century[c][base] / sum(counts_by_century[c].values()) * 100
        if sum(counts_by_century[c].values()) > 0 else 0.0
        for c in centuries
    ])

# Step 6: Normalize per century first so that sum of percentages = 100%
for i in range(len(centuries)):
    total = sum(y_series[k][i] for k in y_series)
    if total > 0:
        for k in y_series:
            y_series[k][i] = y_series[k][i] / total * 100
    else:
        for k in y_series:
            y_series[k][i] = 0.0

# Step 7: Polynomial smoothing for each token series
y_series_smooth = {}
for k in y_series:
    y_series_smooth[k] = polynomial_smooth(x, y_series[k])

# ---------------------------
# PLOT THE RESULTS
# ---------------------------
plt.figure(figsize=(8, 4))
for k, y_smooth in y_series_smooth.items():
    plt.plot(x, y_smooth, linewidth=2, linestyle="--", label=f"% {k}")

plt.xlabel("Century")
plt.ylabel("Percentage")
plt.title("Smoothed Token Percentages per Century")
plt.xticks(x)  # only show existing centuries
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_OUTPUT)
print(f"✅ Graph saved: {FIGURE_OUTPUT}")
plt.show()