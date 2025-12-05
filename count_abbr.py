import os
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend to avoid GUI windows
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm
import concurrent.futures
import unicodedata

# ---------------------------
# CONFIGURATION
# ---------------------------
BASE_DIR = "batch"  # folder containing JSON files
FIGURE_OUTPUT = "graph_abbr.png"  # output figure file
POLY_DEGREE = 3  # degree of polynomial smoothing
LANG_FILTER = "lat"  # language filter
#LANG_FILTER = "fre"  # Only process files with doc["langue"] == LANG_FILTER
DESIRED_CENTURIES = [8, 9, 10, 11, 12, 13, 14, 15, 16]  # centuries to analyze
#DESIRED_CENTURIES = [13, 14, 15, 16]  # Centuries of interest

# Special characters / abbreviations
SPECIAL_CHARS = [
    "ꝑ","ꝓ","⁊","ũ","õ","ã","ẽ","qͤ","qͥ","q̃",
    "ꝰ","ꝯ","ħ","ꝙ","ꝗ","ẜ","l̾","l̃","s̃","sͬ"
]
SPECIAL_REGEX = re.compile("|".join(re.escape(ch) for ch in SPECIAL_CHARS), flags=re.UNICODE)

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------

def decode_unicode_escapes(s: str) -> str:
    r"""
    INPUT: string possibly containing unicode escape sequences
    OUTPUT: decoded string
    PROCESS: decodes literal sequences like '\uXXXX', '\UXXXXXXXX', '\xXX'
    """
    if s is None:
        return ""
    if isinstance(s, bytes):
        s = s.decode("utf-8", "ignore")
    if "\\u" in s or "\\U" in s or "\\x" in s:
        try:
            return bytes(s, "utf-8").decode("unicode_escape")
        except Exception:
            return s
    return s

def normalize_text(s: str) -> str:
    """
    INPUT: string
    OUTPUT: normalized lowercase string
    PROCESS: decode unicode, normalize NFC, casefold
    """
    if s is None:
        return ""
    s = str(s)
    s = decode_unicode_escapes(s)
    s = unicodedata.normalize("NFC", s)
    return s.casefold()

def list_all_json_files(base_dir: str) -> list:
    """
    INPUT: base directory
    OUTPUT: list of all JSON file paths in the directory (recursive)
    PROCESS: walk directory tree and collect .json files
    """
    all_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                all_files.append(os.path.join(root, file))
    return all_files

def year_to_century(year_raw) -> int | None:
    """
    INPUT: integer or string representing a year
    OUTPUT: corresponding century as int, or None if invalid
    PROCESS: convert year to century; years < 100 are treated as 1-based
    """
    if year_raw is None:
        return None
    s = str(year_raw).strip()
    match = re.match(r"(\d+)", s)
    if match:
        year = int(match.group(1))
        return year + 1 if year < 100 else (year // 100) + 1
    return None

# ---------------------------
# TOKEN COUNTING
# ---------------------------

def count_tokens_in_file(filepath: str, lang_filter: str | None = None) -> tuple:
    """
    INPUT: JSON file path, language filter
    OUTPUT: tuple (century, Counter of word categories)
            Counter keys: 'special' for words with special characters,
                          'normal' for other words
    PROCESS:
      1. Load JSON
      2. Filter by language
      3. Extract century
      4. Count words with/without special characters
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, Counter()

    # 🔹 Language filter
    if lang_filter is not None and doc.get("langue") != lang_filter:
        return None, Counter()

    century = year_to_century(doc.get("start_year"))
    if century is None:
        return None, Counter()

    counts = Counter()

    for file_entry in doc.get("files", []):
        for zone in file_entry.get("zones", []):
            for line in zone.get("lines", []):
                content = normalize_text(line.get("content", ""))
                words = re.findall(r"\w+", content, flags=re.UNICODE)
                for w in words:
                    if SPECIAL_REGEX.search(w):
                        counts["special"] += 1
                    else:
                        counts["normal"] += 1

    return century, counts

def process_file(f):
    """
    Wrapper function for multiprocessing
    INPUT: file path
    OUTPUT: tuple from count_tokens_in_file
    """
    return count_tokens_in_file(f, LANG_FILTER)

# ---------------------------
# POLYNOMIAL SMOOTHING
# ---------------------------

def polynomial_smooth(x: np.ndarray, y: np.ndarray, degree: int = POLY_DEGREE) -> np.ndarray:
    """
    INPUT: x array, y array, polynomial degree
    OUTPUT: smoothed y values
    PROCESS: fits polynomial to y(x) and returns fitted values
    """
    if len(x) <= degree:
        return y
    poly = np.poly1d(np.polyfit(x, y, deg=degree))
    return poly(x)

# ---------------------------
# MAIN SCRIPT
# ---------------------------

if __name__ == "__main__":

    # Step 1: List all JSON files
    all_json_files = list_all_json_files(BASE_DIR)
    print(f"JSON files found: {len(all_json_files)}")

    # Step 2: Initialize counters per century
    counts_by_century = defaultdict(Counter)
    files_per_century = defaultdict(int)

    # Step 3: Count words in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_file, all_json_files),
            total=len(all_json_files),
            desc="Analyzing files"
        ))

    # Step 4: Merge counts per century
    for century, counts in results:
        if century is not None:
            counts_by_century[century].update(counts)
            files_per_century[century] += 1

    # Step 5: Prepare arrays for plotting
    centuries = sorted(counts_by_century.keys())
    x = np.array(centuries)
    categories = ["special", "normal"]
    y_series = {}
    for cat in categories:
        y_series[cat] = np.array([
            counts_by_century[c][cat] / sum(counts_by_century[c].values()) * 100
            if sum(counts_by_century[c].values()) > 0 else 0.0
            for c in centuries
        ])

    # Step 6: Normalize percentages to sum 100%
    for i in range(len(centuries)):
        total = sum(y_series[cat][i] for cat in categories)
        if total > 0:
            for cat in categories:
                y_series[cat][i] = y_series[cat][i] / total * 100
        else:
            for cat in categories:
                y_series[cat][i] = 0.0

    # Step 7: Polynomial smoothing for selected centuries
    indices = [i for i, c in enumerate(centuries) if int(c) in DESIRED_CENTURIES]
    y_series_smooth_subset = {}
    for cat in categories:
        x_subset = x[indices]
        y_subset = y_series[cat][indices]
        y_series_smooth_subset[cat] = polynomial_smooth(x_subset, y_subset)

    # Step 8: Plot
    plt.figure(figsize=(3.25, 3.3))
    for cat, y_smooth in y_series_smooth_subset.items():
        label = "% Abbreviated" if cat == "special" else "% Non-abbreviated"
        plt.plot(x[indices], y_smooth, linewidth=2, linestyle="--", label=label)

    plt.xlabel("Century")
    plt.title("Percentage of abbreviated / non-abbreviated words per century")
    plt.xticks(DESIRED_CENTURIES)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT)
    print(f"✅ Graph saved: {FIGURE_OUTPUT}")
    # plt.show()  # Not needed with Agg backend

    # Step 9: Print table
    print("\n=== Percentages per century ===")
    header = ["Century", "% Abbreviated", "% Non-abbreviated"]
    print("{:<8} {:>12} {:>15}".format(*header))
    print("-" * 40)
    for i, c in enumerate(centuries):
        if c in DESIRED_CENTURIES:
            row = [
                str(c),
                f"{y_series['special'][i]:.2f}%",
                f"{y_series['normal'][i]:.2f}%"
            ]
            print("{:<8} {:>12} {:>15}".format(*row))
