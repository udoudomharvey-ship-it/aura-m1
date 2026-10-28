"""
export_training_data.py
────────────────────────────────────────────
MVP Step 2 helper: combine every session log in live_sessions/ into a
single labeled dataset ready for retraining the BNN, and print a
quick honesty report on how much usable (labeled) data you actually
have so far.

Usage:
    python3 export_training_data.py
    # writes: live_sessions/labeled_dataset.csv
"""
import csv
from pathlib import Path

LOG_DIR = Path(__file__).parent / "live_sessions"
OUT_PATH = LOG_DIR / "labeled_dataset.csv"


def main():
    frame_files = sorted(LOG_DIR.glob("session_*_frames.csv"))
    if not frame_files:
        print(f"No session files found in {LOG_DIR}/. "
              f"Run `python3 run_live.py` and label at least one session first.")
        return

    all_rows = []
    header = None
    per_label_counts = {}
    total_rows = 0
    skipped_malformed = 0
    skipped_files = []

    for path in frame_files:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            try:
                file_header = next(reader)
            except StopIteration:
                skipped_files.append(path.name)
                continue  # empty file, e.g. a session with zero frames
            if "label" not in file_header:
                skipped_files.append(path.name)
                continue  # not a real session file / wrong format
            if header is None:
                header = file_header + ["session"]
            label_idx = file_header.index("label")
            for row in reader:
                total_rows += 1
                # A row shorter than the header is corrupted data, most
                # likely from an old session file written before a fix
                # to run_live.py's file-writing. Skip it rather than
                # crashing the whole export over a handful of bad rows.
                if len(row) <= label_idx:
                    skipped_malformed += 1
                    continue
                label = row[label_idx]
                per_label_counts[label or "(unlabeled)"] = per_label_counts.get(label or "(unlabeled)", 0) + 1
                if label:  # only keep labeled rows in the training export
                    all_rows.append(row + [path.stem])

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(all_rows)

    print(f"Combined {len(frame_files)} session file(s) found, {total_rows} total frames read.")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} unreadable/empty file(s): {', '.join(skipped_files)}")
    if skipped_malformed:
        print(f"Skipped {skipped_malformed} corrupted row(s) (likely from an old session file "
              f"written before a run_live.py fix). This does not affect the rest of your data.")
    print(f"Labeled, training-ready rows: {len(all_rows)} "
          f"({100 * len(all_rows) / max(total_rows, 1):.1f}% of all frames)")
    print("\nBreakdown by label:")
    for label, count in sorted(per_label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:28s} {count:6d}")

    if len(all_rows) < 200:
        print(f"\nHeads up: {len(all_rows)} labeled rows is thin for retraining. "
              f"A few more sessions with more frequent labeling will help a lot "
              f"before this goes into Step 3 (retrain the classifier).")

    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
