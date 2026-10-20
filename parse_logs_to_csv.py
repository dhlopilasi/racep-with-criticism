"""
Parse tất cả stream log files trong thư mục logs/ và xuất ra CSV.
Các cột theo yêu cầu, bỏ qua SG explainer.
"""

import csv
import os
import re

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "results.csv")

FIELDNAMES = [
    "stream",
    "explainer",
    "total_n_prototypes_pre_pre",
    "total_n_prototypes_pre_post",
    "mean_minimal_distance_pre_pre",
    "mean_minimal_distance_pre_post",
    "mean_centroid_displacement_pre_pre",
    "mean_centroid_displacement_pre_post",
    "swap_delta_l2_pre_pre",
    "swap_delta_l2_pre_post",
    "swap_delta_tree_pre_pre",
    "swap_delta_tree_pre_post",
    "mi_tree_prototypes_pre_pre",
    "mi_tree_prototypes_pre_post",
    "rand_tree_prototypes_pre_pre",
    "rand_tree_prototypes_pre_post",
    "completeness_tree_prototypes_pre_pre",
    "completeness_tree_prototypes_pre_post",
    "fowlkes_mallows_tree_prototypes_pre_pre",
    "fowlkes_mallows_tree_prototypes_pre_post",
]

# Map từ log key → csv column suffix
KEY_MAP = {
    "total_n_prototypes (a)":                           "total_n_prototypes",
    "mean_minimal_distance":                            "mean_minimal_distance",
    "mean_centroid_displacement":                       "mean_centroid_displacement",
    "swap_delta (l2)":                                  "swap_delta_l2",
    "swap_delta (tree distance)":                       "swap_delta_tree",
    "mutual_information (tree distance, prototypes)":   "mi_tree_prototypes",
    "rand_index (tree distance, prototypes)":           "rand_tree_prototypes",
    "completeness (tree distance, prototypes)":         "completeness_tree_prototypes",
    "fowlkes_mallows (tree distance, prototypes)":      "fowlkes_mallows_tree_prototypes",
}

# Map tên explainer trong filename → tên chuẩn
EXPLAINER_MAP = {
    "apete": "APete",
    "g_km": "G_KM",
    "kmeans": "KMeans",
    "sm_a": "SM_A",
    "sm_wa": "SM_WA",
}


def parse_log(filepath: str) -> dict:
    """Parse một log file, trả về dict {metric_key: {pre_pre: mean, pre_post: mean}}"""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Tách phần pre-pre và pre-post
    pre_pre_match = re.search(r"pre-pre:(.*?)pre-post:", content, re.DOTALL)
    pre_post_match = re.search(r"pre-post:(.*)", content, re.DOTALL)

    if not pre_pre_match or not pre_post_match:
        return None

    pre_pre_text = pre_pre_match.group(1)
    pre_post_text = pre_post_match.group(1)

    def extract_means(text: str) -> dict:
        """Trích xuất mean của từng metric từ block text."""
        means = {}
        # Pattern: <key>:\n\t\tmean: <value>
        pattern = re.compile(
            r"^\t([^\t][^\n]+):\s*\n\t\tmean:\s*([\d.eE+\-]+)",
            re.MULTILINE
        )
        for m in pattern.finditer(text):
            key = m.group(1).strip()
            val = float(m.group(2))
            means[key] = val
        return means

    return {
        "pre_pre": extract_means(pre_pre_text),
        "pre_post": extract_means(pre_post_text),
    }


def parse_filename(filename: str):
    """
    Trích xuất (stream, explainer) từ tên file.
    Format: stream_<stream>_<explainer>_run001.log
    Trả về None nếu không match hoặc là SG.
    """
    # Chỉ xử lý file stream_ và kết thúc _run001.log
    m = re.match(r"stream_(.+)_(apete|g_km|kmeans|sm_a|sm_wa|sg)_run\d+\.log$",
                 filename, re.IGNORECASE)
    if not m:
        return None, None

    stream = m.group(1)
    explainer_raw = m.group(2).lower()

    if explainer_raw == "sg":
        return None, None  # bỏ SG

    explainer = EXPLAINER_MAP.get(explainer_raw, explainer_raw.upper())
    return stream, explainer


def build_row(stream: str, explainer: str, parsed: dict) -> dict:
    row = {"stream": stream, "explainer": explainer}

    for log_key, col_suffix in KEY_MAP.items():
        pre_pre_val = parsed["pre_pre"].get(log_key, "")
        pre_post_val = parsed["pre_post"].get(log_key, "")
        row[f"{col_suffix}_pre_pre"] = pre_pre_val
        row[f"{col_suffix}_pre_post"] = pre_post_val

    return row


def main():
    rows = []
    skipped = []

    for filename in sorted(os.listdir(LOGS_DIR)):
        if not filename.endswith(".log"):
            continue

        stream, explainer = parse_filename(filename)
        if stream is None:
            continue  # bỏ SG hoặc file không phải stream

        filepath = os.path.join(LOGS_DIR, filename)
        parsed = parse_log(filepath)

        if parsed is None:
            skipped.append(filename)
            print(f"  [SKIP] {filename} — không parse được")
            continue

        row = build_row(stream, explainer, parsed)
        rows.append(row)
        print(f"  [OK]   {filename}  →  stream={stream}, explainer={explainer}")

    # Sắp xếp theo stream, explainer
    rows.sort(key=lambda r: (r["stream"], r["explainer"]))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Ghi {len(rows)} dòng vào: {OUTPUT_CSV}")
    if skipped:
        print(f"⚠️  Bỏ qua {len(skipped)} file: {skipped}")


if __name__ == "__main__":
    main()
