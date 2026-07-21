"""Rank A/B serving configs by long-tail metrics."""
import argparse
import json


def compare_reports(named_reports):
    rows = [dict(config=name, **rep) for name, rep in named_reports.items()]
    rows.sort(key=lambda r: r["tail_bubble_s"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", help="name=path/report.json pairs")
    ap.add_argument("--out-html", default=None,
                    help="write a self-contained interactive A/B compare report here")
    a = ap.parse_args(argv)
    named = {}
    for spec in a.reports:
        name, path = spec.split("=", 1)
        named[name] = json.load(open(path))
    for r in compare_reports(named):
        print(f"{r['config']:16s} bubble={r['tail_bubble_s']:.1f}s "
              f"goodput={r['goodput_proxy']:.2f} makespan={r['makespan_s']:.1f}s")
    if a.out_html:
        from rl_traces_bench.report_html import render_compare
        with open(a.out_html, "w") as f:
            f.write(render_compare(named))
        print(f"wrote {a.out_html}")


if __name__ == "__main__":
    main()
