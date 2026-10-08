import argparse
import json
from pathlib import Path

from paper_plots import plot_forest


def main():
    parser = argparse.ArgumentParser(description="Combine the actual BreaKHis/BRACS paired summaries with a shared delta scale.")
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.summaries]
    if any(summary["scope"] != "complete_recorded_test_set" or summary.get("diagnostic_only") for summary in summaries):
        raise ValueError("Diagnostic subsets cannot be plotted as full-test evidence.")
    plot_forest(summaries, args.output)


if __name__ == "__main__":
    main()
