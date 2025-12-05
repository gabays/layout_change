import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures
import unicodedata
import csv
import random

# ---------------------------
# CONFIGURATION
# ---------------------------
BASE_DIR = "batch"  # folder containing JSON files
FIGURE_OUTPUT = "graph_lexical_richness.png"  # output figure file
CSV_OUTPUT = "data_lexical_richness.csv"     # output CSV file
POLY_DEGREE = 3                                  # degree of polynomial smoothing
LANG_FILTER = "lat"  # language filter
#LANG_FILTER = "fre"  # Only process files with doc["langue"] == LANG_FILTER
DESIRED_CENTURIES = [8, 9, 10, 11, 12, 13, 14, 15, 16]  # centuries to analyze
#DESIRED_CENTURIES = [13, 14, 15, 16]  # Centuries of interest
SAMPLE_TOKENS = 200000                           # Number of tokens to sample per century

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------

def decode_unicode_escapes(s: str) -> str:
    r"""Decode unicode escape sequences like \uXXXX, \UXXXXXXXX, \xXX"""
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
    """Normalize text: decode unicode, NFC, casefold"""
    if s is None:
        return ""
    s = decode_unicode_escapes(s)
    s = unicodedata.normalize("NFC", s)
    return s.casefold()

def list_all_json_files(base_dir: str) -> list:
    """Recursively list all JSON files in a directory"""
    all_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                all_files.append(os.path.join(root, file))
    return all_files

def year_to_century(year_raw) -> int | None:
    """Convert year to century"""
    if year_raw is None:
        return None
    s = str(year_raw).strip()
    match = re.match(r"(\d+)", s)
    if match:
        year = int(match.group(1))
        return year + 1 if year < 100 else (year // 100) + 1
    return None

# ---------------------------
# TOKEN AND VOCABULARY COUNTING
# ---------------------------

def count_tokens_and_vocab(filepath: str, lang_filter: str) -> tuple:
    """
    Count tokens and collect vocabulary
    INPUT: JSON file path, language filter
    OUTPUT: tuple (century, total_words, vocab_list)
    PROCESS: returns list of words (not set) to allow rarefaction
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, []

    if doc.get("langue") != lang_filter:
        return None, []

    century = year_to_century(doc.get("start_year"))
    if century is None:
        return None, []

    words_list = []

    for file_entry in doc.get("files", []):
        for zone in file_entry.get("zones", []):
            for line in zone.get("lines", []):
                content = normalize_text(line.get("content", ""))
                words = re.findall(r"\w+", content, flags=re.UNICODE)
                words_list.extend(words)

    return century, words_list

def process_file(f):
    """Wrapper function for multiprocessing"""
    return count_tokens_and_vocab(f, LANG_FILTER)

# ---------------------------
# POLYNOMIAL SMOOTHING
# ---------------------------

def polynomial_smooth(x: np.ndarray, y: np.ndarray, degree: int = POLY_DEGREE) -> np.ndarray:
    """Fit polynomial and return smoothed values"""
    if len(x) <= degree:
        return y
    poly = np.poly1d(np.polyfit(x, y, deg=degree))
    return poly(x)

# ---------------------------
# DIVERSITY METRICS
# ---------------------------

def compute_diversity_metrics(total_words, unique_words):
    """Compute TTR, Guiraud, Herdan indices"""
    if total_words == 0 or unique_words == 0:
        return {"TTR": 0.0, "Guiraud": 0.0, "Herdan": 0.0}
    return {
        "TTR": unique_words / total_words * 100,
        "Guiraud": unique_words / np.sqrt(total_words),
        "Herdan": np.log(unique_words) / np.log(total_words)
    }

# ---------------------------
# MAIN SCRIPT
# ---------------------------

if __name__ == "__main__":

    # Step 1: list all JSON files
    all_json_files = list_all_json_files(BASE_DIR)
    print(f"JSON files found: {len(all_json_files)}")

    words_by_century = defaultdict(list)

    # Step 2: parallel counting
    print("Counting words in parallel...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_file, all_json_files),
            total=len(all_json_files),
            desc="Analyzing files"
        ))

    # Step 3: merge words per century
    print("Merging word lists...")
    words_by_century = defaultdict(list)
    for century, words in results:
        if century is not None and words:
            words_by_century[century] += words

    # Step 4: rarefaction / fixed-size sampling
    print(f"Applying rarefaction to {SAMPLE_TOKENS} tokens per century...")
    diversity_metrics = {metric: [] for metric in ["TTR", "Guiraud", "Herdan"]}
    for c in sorted(words_by_century.keys()):
        words = words_by_century[c]
        total_tokens = len(words)
        if total_tokens > SAMPLE_TOKENS:
            # random sample without replacement
            sample = random.sample(words, SAMPLE_TOKENS)
        else:
            # use all tokens if smaller than SAMPLE_TOKENS
            sample = words
        unique_words = len(set(sample))
        metrics = compute_diversity_metrics(len(sample), unique_words)
        for key in metrics:
            diversity_metrics[key].append(metrics[key])

    centuries = sorted(words_by_century.keys())
    x = np.array(centuries)

    # Step 5: smoothing
    print("Smoothing metrics...")
    indices = [i for i, c in enumerate(centuries) if c in DESIRED_CENTURIES]
    x_subset = x[indices]
    smoothed_metrics = {key: polynomial_smooth(x_subset, np.array(diversity_metrics[key])[indices])
                        for key in diversity_metrics}

    # Step 6: save CSV and print table
    print("\n=== Lexical diversity by century (sampled) ===")
    header = ["Century", "Sampled tokens", "Unique words", "TTR (%)", "Guiraud", "Herdan"]
    print("{:<8} {:>15} {:>12} {:>10} {:>10} {:>10}".format(*header))
    print("-" * 80)

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i, c in enumerate(centuries):
            if c in DESIRED_CENTURIES:
                sample_size = min(len(words_by_century[c]), SAMPLE_TOKENS)
                unique_count = len(set(words_by_century[c][:sample_size]))
                ttr = diversity_metrics["TTR"][i]
                guiraud = diversity_metrics["Guiraud"][i]
                herdan = diversity_metrics["Herdan"][i]
                print("{:<8} {:>15} {:>12} {:>10.2f} {:>10.2f} {:>10.2f}".format(
                    c, sample_size, unique_count, ttr, guiraud, herdan))
                writer.writerow([c, sample_size, unique_count, f"{ttr:.2f}", f"{guiraud:.2f}", f"{herdan:.2f}"])
    print(f"✅ CSV saved: {CSV_OUTPUT}")

    # Step 7: plot
    print("Plotting...")
    plt.figure(figsize=(8, 5))
    colors = {"TTR": "purple", "Guiraud": "green", "Herdan": "blue"}
    for key in smoothed_metrics:
        plt.plot(x_subset, smoothed_metrics[key], label=key, linestyle="--", color=colors[key], linewidth=2)
    plt.xlabel("Century")
    plt.ylabel("Diversity metric value")
    plt.title(f"Lexical diversity metrics (sampled {SAMPLE_TOKENS} tokens) by century")
    plt.xticks(DESIRED_CENTURIES)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT)
    plt.show()
    print(f"✅ Figure saved: {FIGURE_OUTPUT}")
