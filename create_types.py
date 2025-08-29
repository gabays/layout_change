import os
import json
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm  # Progress bar for file processing

# ---------------------------
# CONFIGURATION
# ---------------------------
BASE_DIR = "small_batch"   # Root folder containing subfolders with JSON files
OUTPUT_FOLDER = "pages_type_overlay"  # Folder to save generated overlay images
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Create output folder if it does not exist

# Assign a distinct color to each zone type for visualization
ZONE_COLORS = {
    "MainZone": "blue",
    "MarginTextZone": "green",
    "DropCapitalZone": "red",
    "DefaultLine": "purple",
    "Other": "gray"  # fallback for unknown types
}

# Maximum alpha (opacity) and linewidth for rectangle edges
ALPHA_MAX = 0.3
LW_MAX = 1.0
LW_MIN = 0.0001  # Minimum linewidth for very frequent zones

# Exponential decay coefficient: controls how alpha & linewidth decrease with frequency
EXP_COEFF = 2.0

# ---------------------------
# FUNCTIONS
# ---------------------------
def year_to_century(year_raw):
    """
    Convert a raw start_year value into the corresponding century.
    - If year < 100, assume it is 1-based and add 1
    - Otherwise, integer division by 100 + 1
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
    Recursively collect all JSON files in a directory and its subdirectories.
    Returns a list of full file paths.
    """
    files_list = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".json"):
                files_list.append(os.path.join(root, f))
    return files_list

# ---------------------------
# STEP 1: Collect page and zone info per century & orientation
# ---------------------------

# Dictionary structure: century -> {"portrait": [...], "landscape": [...]}
century_pages = defaultdict(lambda: {"portrait": [], "landscape": []})

# Collect all JSON files recursively
all_files = list_all_json_files(BASE_DIR)
print(f"Total JSON files found: {len(all_files)}")

# Iterate over all files with a progress bar
for filepath in tqdm(all_files, desc="Reading JSON files"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        continue

    # Determine the century of the document
    century = year_to_century(doc.get("start_year"))
    if century is None:
        continue

    # Iterate over all pages in the document
    for file_entry in doc.get("files", []):
        wh_page = file_entry.get("wh", [1000, 1000])  # default width/height if missing
        if len(wh_page) != 2 or wh_page[0] == 0 or wh_page[1] == 0:
            continue

        # Determine page orientation
        orientation = "landscape" if wh_page[0] > wh_page[1] else "portrait"

        # Collect all zones for the page
        page_zones = []
        for zone in file_entry.get("zones", []):
            xy = zone.get("xy", [0, 0])  # top-left corner coordinates
            wh = zone.get("wh", [0, 0])  # width & height
            ztype = zone.get("type", "Other")  # zone type, default to "Other"

            # Validate coordinates and dimensions
            if len(xy) == 2 and len(wh) == 2 and wh[0] > 0 and wh[1] > 0:
                page_zones.append({
                    "x": xy[0],
                    "y": xy[1],
                    "w": wh[0],
                    "h": wh[1],
                    "type": ztype
                })

        # Add the page data to the corresponding century and orientation
        if page_zones:
            century_pages[century][orientation].append({
                "page_wh": wh_page,
                "zones": page_zones
            })

# ---------------------------
# STEP 2: Draw overlay "page type" per century and orientation
# ---------------------------

for century in tqdm(century_pages.keys(), desc="Generating overlay pages"):
    for orientation in ["portrait", "landscape"]:
        pages = century_pages[century][orientation]
        if not pages:
            continue

        # Compute average page size for normalization/scaling
        avg_w = np.mean([p["page_wh"][0] for p in pages])
        avg_h = np.mean([p["page_wh"][1] for p in pages])

        # Create figure with scaled size
        fig, ax = plt.subplots(figsize=(avg_w/200, avg_h/200))
        ax.set_xlim(0, avg_w)
        ax.set_ylim(0, avg_h)
        ax.set_title(f"Representative Page Overlay - Century {century} ({orientation})")
        ax.set_aspect("equal")
        ax.axis("off")  # hide axes

        # Count how many times each zone type occurs (for dynamic alpha/linewidth)
        zone_type_counts = Counter(z["type"] for p in pages for z in p["zones"])
        max_count = max(zone_type_counts.values()) if zone_type_counts else 1

        # Draw all zones with scaling applied
        for page in pages:
            w_page, h_page = page["page_wh"]
            scale_x = avg_w / w_page
            scale_y = avg_h / h_page

            for z in page["zones"]:
                rect_x = z["x"] * scale_x
                rect_y = z["y"] * scale_y
                rect_w = z["w"] * scale_x
                rect_h = z["h"] * scale_y
                zone_type = z["type"]

                freq = zone_type_counts[zone_type]

                # Exponential/logarithmic scaling:
                # - zones that appear frequently become more transparent and thinner
                alpha = ALPHA_MAX * math.exp(-EXP_COEFF * freq / max_count)
                linewidth = max(LW_MIN, LW_MAX * math.exp(-EXP_COEFF * freq / max_count))

                # Draw rectangle with edge color corresponding to zone type
                ax.add_patch(
                    plt.Rectangle(
                        (rect_x, rect_y),
                        rect_w,
                        rect_h,
                        edgecolor=ZONE_COLORS.get(zone_type, "gray"),
                        facecolor="none",
                        linewidth=linewidth,
                        alpha=alpha
                    )
                )

        # Add a legend showing which color corresponds to which zone type
        for ztype, color in ZONE_COLORS.items():
            ax.plot([], [], color=color, label=ztype, linewidth=2)
        ax.legend(loc="upper right", fontsize=8)

        # Save figure to output folder
        fig_path = os.path.join(OUTPUT_FOLDER, f"page_type_overlay_century_{century}_{orientation}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved overlay page type for century {century} ({orientation}): {fig_path}")