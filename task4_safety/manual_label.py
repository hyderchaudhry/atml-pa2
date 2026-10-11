"""Blind, human-only labeling of the existing fixed Task 4 audit sheets.

Run with --all or a policy CSV path. Only the fixed-ID CSV and the four
policy-specific audit CSVs are read; no model or AI judge code is imported.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile


POLICIES = ("sft", "dpo", "ppo", "grpo")
LABELS = (
    "SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE",
    "OVER_REFUSAL", "AMBIGUOUS",
)
AUDIT_DIR = Path(__file__).resolve().parents[1] / "results" / "task4_safety"
REQUIRED_FIELDS = {
    "xstest_id", "policy", "benchmark_class", "type", "prompt", "response",
    "manual_label", "manual_notes",
}


def read_csv(path, required_fields):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"{path}: missing or duplicate CSV headers")
        missing = required_fields - set(fields)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: a CSV row has too many or too few fields")
    return fields, rows


def validate_ids(path, rows, fixed_ids=None):
    try:
        ids = [int(row["xstest_id"]) for row in rows]
    except ValueError as exc:
        raise ValueError(f"{path}: xstest_id must contain integer IDs") from exc
    duplicates = sorted(index for index, count in Counter(ids).items() if count > 1)
    problems = []
    if len(rows) != 60:
        problems.append(f"expected exactly 60 rows, found {len(rows)}")
    if duplicates:
        problems.append(f"duplicate IDs: {duplicates}")
    if fixed_ids is not None:
        missing, extra = set(fixed_ids) - set(ids), set(ids) - set(fixed_ids)
        if missing:
            problems.append(f"missing IDs: {sorted(missing)}")
        if extra:
            problems.append(f"extra IDs: {sorted(extra)}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return ids


@dataclass
class AuditSheet:
    path: Path
    fields: list[str]
    rows: list[dict[str, str]]

    def save(self):
        """Write all original fields atomically, preserving their string values."""
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=self.fields)
                writer.writeheader()
                writer.writerows(self.rows)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(self.path.stat().st_mode & 0o777)
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def load_audits(directory):
    """Validate all five CSVs before accepting any human judgments."""
    fixed_path = directory / "manual_audit_ids.csv"
    _, fixed_rows = read_csv(fixed_path, {"xstest_id"})
    fixed_ids = validate_ids(fixed_path, fixed_rows)
    sheets = {}
    for policy in POLICIES:
        path = directory / f"manual_audit_{policy}.csv"
        fields, rows = read_csv(path, REQUIRED_FIELDS)
        validate_ids(path, rows, fixed_ids)
        for row in rows:
            label = row["manual_label"]
            if label != "" and label not in LABELS:
                raise ValueError(
                    f"{path}: ID {row['xstest_id']}: invalid manual_label {label!r}; "
                    f"expected blank or one of {', '.join(LABELS)}"
                )
        sheets[policy] = AuditSheet(path, fields, rows)
    return sheets


def print_progress(sheets):
    print()
    counts = [sum(row["manual_label"] in LABELS for row in sheets[p].rows) for p in POLICIES]
    for policy, count in zip(POLICIES, counts):
        print(f"{policy.upper()}: {count}/60 labeled")
    if all(count == 60 for count in counts):
        print("Manual audit complete: all four policy files have 60/60 human labels.")


def label_sheet(sheet):
    for position, row in enumerate(sheet.rows, 1):
        if row["manual_label"] in LABELS:
            continue
        print(f"\nExample {position}/60")
        print(f"XSTest ID: {row['xstest_id']}\nPolicy: {row['policy']}")
        print(f"Benchmark class: {row['benchmark_class']}\nType: {row['type']}")
        print(f"\nPROMPT:\n{row['prompt']}\n\nRESPONSE:\n{row['response']}\n")
        for number, label in enumerate(LABELS, 1):
            print(f"{number} = {label}")
        print("q = quit")
        while True:
            choice = input("Your judgment: ").strip()
            if choice == "q":
                return False
            if choice in ("1", "2", "3", "4", "5"):
                break
            print("Enter exactly 1, 2, 3, 4, 5, or q.")
        row["manual_label"] = LABELS[int(choice) - 1]
        row["manual_notes"] = ""
        sheet.save()  # Save the judgment before the optional notes prompt.
        note = input("Manual notes (optional; Enter leaves blank): ")
        if note:
            row["manual_notes"] = note
            sheet.save()
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("file", nargs="?", type=Path, help="existing policy audit CSV")
    selection.add_argument("--all", action="store_true", help="label SFT, DPO, PPO, then GRPO")
    args = parser.parse_args(argv)
    try:
        sheets = load_audits(AUDIT_DIR)
        selected = list(POLICIES)
        if not args.all:
            selected = [p for p in POLICIES if sheets[p].path.resolve() == args.file.resolve()]
            if not selected:
                raise ValueError("Select one of the four existing results/task4_safety/manual_audit_<policy>.csv files")
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Validation error: {exc}. No audit files were changed.", file=sys.stderr)
        return 1
    print("Human-only audit: enter your own judgment for each policy response.")
    print_progress(sheets)
    status = 0
    try:
        for policy in selected:
            if not label_sheet(sheets[policy]):
                print("\nQuit. Saved judgments will be skipped when you resume.")
                break
        else:
            print("\nSelected audit files finished.")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Saved judgments will be skipped when you resume.")
    except (OSError, csv.Error) as exc:
        print(f"Save error: {exc}. Stopping; the last write may not have completed.", file=sys.stderr)
        status = 1
    finally:
        # Reload so even a failed/interrupted save reports only persisted labels.
        print_progress(load_audits(AUDIT_DIR))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
