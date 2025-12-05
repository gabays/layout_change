import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm
import concurrent.futures
import unicodedata

# ---------------------------
# CONFIGURATION
# ---------------------------
BASE_DIR = "batch"  # Root folder containing JSON files
FIGURE_OUTPUT = "graph_percentages.png"  # Output file for the figure
POLY_DEGREE = 3  # Degree of the polynomial for smoothing
LANG_FILTER = "fre"  # Only process files with doc["langue"] == LANG_FILTER
DESIRED_CENTURIES = [13, 14, 15, 16]  # Centuries of interest

# List of tokens to count (variants separated by |)
#TOKENS_TO_COUNT = ["prison|prisou", "prisonnier|prisonier|prisouier|prisonuier|prisouuier|prisounier"]
#TOKENS_TO_COUNT = ["sunt", "sont", "sũt", "sõt"]
TOKENS_TO_COUNT = ["gauche", "senestre"]
#TOKENS_TO_COUNT = ["ceste", "cele|celle"]
#TOKENS_TO_COUNT = ["cil|cele|celle", "cist|ceste|cest"]
#TOKENS_TO_COUNT = ["celui|celiu|celni|celin", "cestui|cestni|cestin|cestiu"]
#TOKENS_TO_COUNT = ["message", "messager"]
#TOKENS_TO_COUNT = ["trait", "tire"]
#TOKENS_TO_COUNT = ["bailler", "donner"]
#TOKENS_TO_COUNT = ["baille", "donne"]
#TOKENS_TO_COUNT = ["traire", "tirer"]
#TOKENS_TO_COUNT = ["tout", "tot"]
#TOKENS_TO_COUNT = ["seigneur", "seignor", "seignur", "seign̾", "seigñ", "seign̾r", "seigñr"]
#TOKENS_TO_COUNT = ["mult", "molt", "moult", "ml̾t|ml̃t|mlt|młt"]
#TOKENS_TO_COUNT = ["qil", "quil", "kil", "qͥl"]
#TOKENS_TO_COUNT = ["a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|w|y|z", "ꝑ|ꝓ|⁊|ũ|õ|ã|ẽ|qͤ|qͥ|q̃|ꝰ|ꝯ|ħ|ꝙ|ꝗ|ẜ|l̾|l̃|s̃|sͬ"]
#TOKENS_TO_COUNT = ["chevalier|cheualier", "chevaler|cheualer", "chivalier|chiualier", "chavalier|chaualier", "chr̃|chl̾r|chl̃r"]
#TOKENS_TO_COUNT = ["sire", "seigneur", "sʳ|sͬ|s̃"]
#TOKENS_TO_COUNT = ["por", "pour", "pʳ", "pͬ"] #"ꝑ",
#TOKENS_TO_COUNT = ["lié", "joyeux"]
#TOKENS_TO_COUNT = ["⁊", "et", "e"]
#TOKENS_TO_COUNT = ["dict", "dit"]
#TOKENS_TO_COUNT = ["cestui", "celui"]
#TOKENS_TO_COUNT = ["avaler|aualer", "descendre"]
# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------

# --- Unicode / normalization ---

def decode_unicode_escapes(s: str) -> str:
    r"""
    Decode literal sequences \uXXXX or \UXXXXXXXX in a string.
    Input:
        s: str or bytes
    Output:
        Decoded str
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
    r"""
    Normalize a Unicode string for comparison:
        - decode literal sequences (\uXXXX)
        - NFC normalization (compose letters + combining marks)
        - casefold for case-insensitive matching
    Input:
        s: str
    Output:
        Normalized str
    """
    if s is None:
        return ""
    s = str(s)
    s = decode_unicode_escapes(s)
    s = unicodedata.normalize("NFC", s)
    return s.casefold()

def prepare_token_patterns(tokens: list) -> tuple[dict, dict]:
    """
    Prepare regex patterns for tokens with variants.
    Input:
        tokens: list of str, each element may contain multiple variants separated by "|"
    Output:
        patterns: dict {normalized_base: compiled_regex} for counting occurrences
        label_map: dict {normalized_base: original_base_label} for display
    """
    patterns = {}
    label_map = {}
    for t in tokens:
        variants = [v.strip() for v in t.split("|") if v.strip()]
        if not variants:
            continue
        base_label = variants[0]
        norm_base = normalize_text(base_label)
        norm_variants = [normalize_text(v) for v in variants]
        escaped = [re.escape(v) for v in norm_variants if v]
        if not escaped:
            continue
        # regex for full word (unicode aware)
        #To count words
#        pattern = re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)", flags=re.UNICODE)
        #To count letters
        pattern = re.compile("(?:" + "|".join(escaped) + ")", flags=re.UNICODE)
        patterns[norm_base] = pattern
        label_map[norm_base] = base_label
    return patterns, label_map

# Cache for worker processes
_PATTERNS_CACHE = None
_LABELS_CACHE = None
def get_cached_patterns(tokens):
    """
    Return compiled patterns and labels for multiprocessing workers.
    Initialize cache if necessary.
    """
    global _PATTERNS_CACHE, _LABELS_CACHE
    if _PATTERNS_CACHE is None:
        _PATTERNS_CACHE, _LABELS_CACHE = prepare_token_patterns(tokens)
    return _PATTERNS_CACHE, _LABELS_CACHE

# --- JSON file / century handling ---

def list_all_json_files(base_dir: str) -> list:
    """
    List all JSON files in a directory and subdirectories.
    Input:
        base_dir: str, path to root folder
    Output:
        all_files: list of str, absolute paths to all JSON files
    """
    all_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                all_files.append(os.path.join(root, file))
    return all_files

def year_to_century(year_raw) -> int | None:
    """
    Convert a year into a century.
    Input:
        year_raw: int, str, or None
    Output:
        int: century or None if invalid
    Rules:
        - full year >= 100: (year // 100) + 1
        - year < 100: year + 1
    """
    if year_raw is None:
        return None
    s = str(year_raw).strip()
    match = re.match(r"(\d+)", s)
    if match:
        year = int(match.group(1))
        return year + 1 if year < 100 else (year // 100) + 1
    return None

# --- Token counting ---

def count_tokens_in_file(filepath: str, tokens: list, lang_filter: str | None = None) -> tuple:
    """
    Count token occurrences in a JSON file filtered by language.
    Input:
        filepath: str, path to JSON file
        tokens: list of str, tokens to count (may contain variants)
        lang_filter: str, language code to filter
    Output:
        tuple:
            - century: int or None
            - counts: Counter {normalized_base: occurrences}
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

    patterns, _labels = get_cached_patterns(tokens)
    counts = Counter()

    for file_entry in doc.get("files", []):
        for zone in file_entry.get("zones", []):
            for line in zone.get("lines", []):
                content = line.get("content", "")
                content_norm = normalize_text(content)
                # Count using regex (robust to unicode combining marks)
                for norm_base, pat in patterns.items():
                    matches = pat.findall(content_norm)
                    if matches:
                        counts[norm_base] += len(matches)
    return century, counts

# --- Polynomial smoothing ---

def polynomial_smooth(x: np.ndarray, y: np.ndarray, degree: int = POLY_DEGREE) -> np.ndarray:
    """
    Smooth a y series over x using a polynomial of given degree.
    Input:
        x: np.ndarray of centuries
        y: np.ndarray of percentages
        degree: int, polynomial degree
    Output:
        np.ndarray of smoothed values
    """
    if len(x) <= degree:
        return y
    poly = np.poly1d(np.polyfit(x, y, deg=degree))
    return poly(x)

# --- Helper for multiprocessing ---
def process_file(f):
    """
    Wrapper for ProcessPoolExecutor.
    """
    return count_tokens_in_file(f, TOKENS_TO_COUNT, LANG_FILTER)

# ---------------------------
# MAIN SCRIPT
# ---------------------------
if __name__ == "__main__":

    # --- Step 1: List all JSON files ---
    all_json_files = list_all_json_files(BASE_DIR)
    print(f"JSON files found: {len(all_json_files)}")

    # --- Step 2: Initialize counters ---
    counts_by_century = defaultdict(Counter)
    files_per_century = defaultdict(int)

    # --- Step 3: Count files in parallel ---
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_file, all_json_files),
            total=len(all_json_files),
            desc="Analyzing files"
        ))

    for century, counts in results:
        if century is not None:
            counts_by_century[century].update(counts)
            files_per_century[century] += 1

    # --- Step 4: Sort existing centuries ---
    centuries = sorted(counts_by_century.keys())
    x = np.array(centuries)

    # --- Step 5: Prepare patterns and labels for display ---
    patterns_main, label_map_main = prepare_token_patterns(TOKENS_TO_COUNT)
    normalized_bases = list(patterns_main.keys())

    # --- Step 6: Compute percentage per century ---
    y_series = {}
    for norm_base in normalized_bases:
        y_series[norm_base] = np.array([
            counts_by_century[c][norm_base] / sum(counts_by_century[c].values()) * 100
            if sum(counts_by_century[c].values()) > 0 else 0.0
            for c in centuries
        ])

    # --- Step 7: Normalize per century (sum = 100%) ---
    for i in range(len(centuries)):
        total = sum(y_series[k][i] for k in y_series)
        if total > 0:
            for k in y_series:
                y_series[k][i] = y_series[k][i] / total * 100
        else:
            for k in y_series:
                y_series[k][i] = 0.0

    # --- Step 8: Polynomial smoothing on desired centuries ---
    indices = [i for i, c in enumerate(centuries) if int(c) in DESIRED_CENTURIES]
    y_series_smooth_subset = {}
    for k in y_series:
        x_subset = x[indices]
        y_subset = y_series[k][indices]
        y_series_smooth_subset[k] = polynomial_smooth(x_subset, y_subset)

    # --- Step 9: Plot results ---
    plt.figure(figsize=(3.25, 3.3))
    for k, y_smooth in y_series_smooth_subset.items():
        label = label_map_main.get(k, k)
        plt.plot(DESIRED_CENTURIES, y_smooth, linewidth=2, linestyle="--", label=f"% {label}")

    plt.xlabel("Century")
    plt.title("Smoothed Tok. % / cent.")
    plt.xticks(DESIRED_CENTURIES)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT)
    print(f"✅ Graph saved: {FIGURE_OUTPUT}")
    plt.show()

    # --- Step 10: CSV table ---
    print("\n=== Percentages by century ===")
    header = ["Century"] + [label_map_main.get(k, k) for k in normalized_bases]
    print("\t".join(header))
    for i, c in enumerate(centuries):
        if c in DESIRED_CENTURIES:
            row = [str(c)] + [f"{y_series[k][i]:.2f}%" for k in normalized_bases]
            print("\t".join(row))