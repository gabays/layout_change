import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------
# BASE CONFIGURATION
# ---------------------------
BASE_DIR = "batch"  # Root folder containing subfolders with JSON files
CSV_OUTPUT = "data.csv"
FIGURE_OUTPUT = "graph.png"
POLY_DEGREE = 6  # Degree of polynomial for smoothing
SAVE_CSV = False   # Set to False to skip saving CSV

# ---------------------------
# FUNCTION DEFINITIONS
# ---------------------------

def year_to_century(year_raw):
    """
    Convert a raw 'start_year' value to a century.
    Rules:
    - Full year (e.g., 843) => century = (year // 100) + 1
    - 1-2 digit year (partial, e.g., 12, 7) or with punctuation (12.., 7...) => century = int(year) + 1
    - Returns None if the year is invalid
    """
    if year_raw is None:
        return None

    s = str(year_raw).strip()
    match = re.match(r"(\d+)", s)
    if match:
        year = int(match.group(1))
        if year < 100:
            return year + 1
        else:
            return (year // 100) + 1
    return None

def list_all_json_files(base_dir):
    """
    Recursively finds all JSON files under base_dir and returns a list of full paths.
    """
    all_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                all_files.append(os.path.join(root, file))
    return all_files

def tokenize(text):
    """
    Simple tokenizer: split text into alphanumeric tokens.
    """
    if not text:
        return []
    return re.findall(r"\w+", text)

def process_json_file(filepath):
    """
    Reads a single JSON file and computes:
    - Average number of MainZone per file_entry
    - Average number of MarginTextZone per file_entry
    - Average number of zones per file_entry
    - Average number of tokens per file_entry (from content of lines)
    
    If wh[0] > wh[1], all counts and tokens are halved.
    
    Returns:
        (century, avg_main, avg_margin, avg_graphic, avg_total, avg_tokens) 
        or None if start_year is invalid.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Convert start_year to century
    century = year_to_century(doc.get("start_year"))
    if century is None:
        return None

    # Initialize per-file counters
    files_with_main = files_with_margin = files_with_any = files_with_graphic = files_with_drop = files_with_tokens = 0
    total_main = total_margin = total_graphic = total_any = total_tokens = 0.0

    for file_entry in doc.get("files", []):
        zones = file_entry.get("zones", [])
        main_count = sum(1 for z in zones if z.get("type") == "MainZone")
        margin_count = sum(1 for z in zones if z.get("type") == "MarginTextZone")
        graphic_count = sum(1 for z in zones if z.get("type") == "GraphicZone")
        drop_count = sum(1 for z in zones if z.get("type") == "DropCapitalZone")
        any_count = len(zones)

        # 🔹 New calculation: token count
        token_count = 0
        for z in zones:
            for line in z.get("lines", []):
                content = line.get("content", "")
                token_count += len(tokenize(content))

        # Halve counts if width > height
        wh = file_entry.get("wh", [])
        if len(wh) == 2 and wh[0] > wh[1]:
            main_count *= 0.5
            margin_count *= 0.5
            graphic_count *= 0.5
            drop_count *= 0.5
            any_count *= 0.5
            token_count *= 0.5

        # Aggregate counts per file_entry.
        # We only take into consideration pages with a MainZone (exclude empty pages, title pages…)
        if main_count > 0:
            files_with_main += 1
            total_main += main_count
        if main_count > 0:
            files_with_margin += 1
            total_margin += margin_count
        if main_count > 0:
            files_with_graphic += 1
            total_graphic += graphic_count
        if main_count > 0:
            files_with_any += 1
            total = main_count + margin_count + graphic_count + drop_count
            total_any += total
        if main_count > 0:
            files_with_tokens += 1
            total_tokens += token_count

    avg_main = total_main / files_with_main if files_with_main > 0 else None
    avg_margin = total_margin / files_with_margin if files_with_margin > 0 else None
    avg_graphic = total_graphic / files_with_graphic if files_with_graphic > 0 else None
    avg_total = total_any / files_with_any if files_with_any > 0 else None
    avg_tokens = total_tokens / files_with_tokens if files_with_tokens > 0 else None

    return (century, avg_main, avg_margin, avg_graphic, avg_total, avg_tokens)

def polynomial_smooth(x, y, degree=POLY_DEGREE):
    """
    Fit a polynomial of given degree to (x, y) data and return smoothed values.
    """
    poly = np.poly1d(np.polyfit(x, y, deg=degree))
    return poly(x)

# ---------------------------
# MAIN SCRIPT
# ---------------------------

# Step 1: List all JSON files
all_json_files = list_all_json_files(BASE_DIR)
print(f"Total JSON files found: {len(all_json_files)}")

# Step 2: Process JSON files with progress bar
data_main = []
data_margin = []
data_graphic = []
data_total = []
data_tokens = []

for filepath in tqdm(all_json_files, desc="Processing JSON files"):
    result = process_json_file(filepath)
    if result:
        century, avg_main, avg_margin, avg_graphic, avg_total, avg_tokens = result
        if avg_main is not None: data_main.append((century, avg_main))
        if avg_margin is not None: data_margin.append((century, avg_margin))
        if avg_graphic is not None: data_graphic.append((century, avg_graphic))
        if avg_total is not None: data_total.append((century, avg_total))
        if avg_tokens is not None: data_tokens.append((century, avg_tokens))

# Step 3: Create DataFrames and sort by century
df_main = pd.DataFrame(data_main, columns=["century", "avg_mainzone"]).sort_values("century").reset_index(drop=True)
df_margin = pd.DataFrame(data_margin, columns=["century", "avg_margin"]).sort_values("century").reset_index(drop=True)
df_graphic = pd.DataFrame(data_graphic, columns=["century", "avg_graphic"]).sort_values("century").reset_index(drop=True)
df_total = pd.DataFrame(data_total, columns=["century", "avg_total"]).sort_values("century").reset_index(drop=True)
df_tokens = pd.DataFrame(data_tokens, columns=["century", "avg_tokens"]).sort_values("century").reset_index(drop=True)

# Step 4: Optionally save CSV
if SAVE_CSV:
    df_csv = df_main.merge(df_margin, on="century", how="outer")\
                    .merge(df_graphic, on="century", how="outer")\
                    .merge(df_total, on="century", how="outer")\
                    .merge(df_tokens, on="century", how="outer")
    df_csv.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
    print(f"✅ CSV saved: {CSV_OUTPUT}")
else:
    print("ℹ️ CSV saving skipped.")

# Step 5: Smooth curves using polynomial regression
y_main_smooth = polynomial_smooth(df_main["century"].values, df_main["avg_mainzone"].values)
y_margin_smooth = polynomial_smooth(df_margin["century"].values, df_margin["avg_margin"].values)
y_graphic_smooth = polynomial_smooth(df_graphic["century"].values, df_graphic["avg_graphic"].values)
y_total_smooth = polynomial_smooth(df_total["century"].values, df_total["avg_total"].values)
y_tokens_smooth = polynomial_smooth(df_tokens["century"].values, df_tokens["avg_tokens"].values)

# Step 6: Plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(3.25, 3.3))  # small format to fit two-columns layout

# Left y-axis: zones
ax1.set_xlabel("Century")
ax1.set_ylabel("Zones/page (avg)")
ax1.plot(df_main["century"], y_main_smooth, color="red", linewidth=2, label="MainZone")
ax1.plot(df_margin["century"], y_margin_smooth, color="orange", linewidth=2, label="MarginTextZone")
ax1.plot(df_graphic["century"], y_graphic_smooth, color="pink", linewidth=2, label="GraphicZone")
ax1.plot(df_total["century"], y_total_smooth, color="brown", linewidth=2, label="TotalZones")
ax1.tick_params(axis="y", labelcolor="black")
ax1.grid(True, linestyle="--", alpha=0.5)

# Right y-axis: tokens
ax2 = ax1.twinx()
ax2.set_ylabel("Tokens/page (avg)")
ax2.plot(df_tokens["century"], y_tokens_smooth, color="black", linewidth=2, label="Tokens")
ax2.tick_params(axis="y", labelcolor="black")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

plt.title("Zones and Tokens per Century (Polyn. reg.)", fontsize=9)
fig.tight_layout()
plt.savefig("graph.pdf", bbox_inches="tight")  # ✅ vectoriel pour LaTeX
print("✅ Figure saved: graph.pdf")
plt.show()