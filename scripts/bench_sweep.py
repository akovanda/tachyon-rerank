#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from pathlib import Path


def parse_case(s):
    if ":" in s:
        dims, iters = s.split(":", 1)
        iters = int(iters)
    else:
        dims, iters = s, None
    if "x" not in dims:
        raise ValueError(f"bad case '{s}', expected 1024x128 or 1024x128:5")
    n_str, d_str = dims.split("x", 1)
    return {"n": int(n_str), "d": int(d_str), "iters": iters, "name": f"n{n_str}_d{d_str}"}


def case_title(case_rows):
    n = int(case_rows[0]["n"])
    d = int(case_rows[0]["d"])
    if n == 1024 and d == 128:
        return "Small Rerank Set (1K candidates x 128 dims)"
    if n == 8192 and d == 128:
        return "Large Rerank Set (8K candidates x 128 dims)"
    if n == 1024 and d == 768:
        return "Compact High-Dim Set (1K candidates x 768 dims)"
    if n == 8192 and d == 768:
        return "High-Dim Embeddings (8K candidates x 768 dims)"
    return f"{n} candidates x {d} dims"


def write_latency_svg(case_rows, out_path):
    colors = {"cpu": "#1f77b4", "ort": "#ff7f0e", "qnn": "#2ca02c", "auto": "#9467bd"}
    batches = sorted({int(r["b"]) for r in case_rows})
    backends = sorted({r["requested_backend"] for r in case_rows})

    series = {b: [] for b in backends}
    for r in case_rows:
        series[r["requested_backend"]].append((int(r["b"]), float(r["p50_ms"])))
    for b in backends:
        series[b] = sorted(series[b])

    w, h = 640, 400
    margin = 60
    plot_w = w - 2 * margin
    plot_h = h - 2 * margin

    vals = [float(r["p50_ms"]) for r in case_rows]
    y_min = 0.0
    y_max = max(vals) * 1.1 if vals else 1.0
    if y_max <= 0:
        y_max = 1.0

    def x_scale(batch):
        idx = batches.index(batch)
        if len(batches) == 1:
            return margin + plot_w / 2
        return margin + (plot_w * idx / (len(batches) - 1))

    def y_scale(val):
        return margin + plot_h - (plot_h * (val - y_min) / (y_max - y_min))

    svg = []
    svg.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>")
    svg.append("<rect width='100%' height='100%' fill='white' />")
    svg.append(
        f"<text x='{w/2}' y='28' text-anchor='middle' font-family='sans-serif' font-size='16'>{case_title(case_rows)}: p50 latency vs q_batch</text>"
    )

    x0, y0 = margin, margin + plot_h
    x1, y1 = margin + plot_w, margin
    svg.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#333' stroke-width='1' />")
    svg.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#333' stroke-width='1' />")

    for i in range(6):
        v = y_min + (y_max - y_min) * i / 5
        y = y_scale(v)
        svg.append(f"<line x1='{x0-4}' y1='{y}' x2='{x0}' y2='{y}' stroke='#333' stroke-width='1' />")
        svg.append(f"<text x='{x0-8}' y='{y+4}' text-anchor='end' font-family='sans-serif' font-size='10'>{v:.1f}</text>")
        svg.append(f"<line x1='{x0}' y1='{y}' x2='{x1}' y2='{y}' stroke='#eee' stroke-width='1' />")

    for batch in batches:
        x = x_scale(batch)
        svg.append(f"<line x1='{x}' y1='{y0}' x2='{x}' y2='{y0+4}' stroke='#333' stroke-width='1' />")
        svg.append(f"<text x='{x}' y='{y0+18}' text-anchor='middle' font-family='sans-serif' font-size='10'>{batch}</text>")

    for backend in backends:
        pts = series[backend]
        if not pts:
            continue
        points = []
        for batch, val in pts:
            x = x_scale(batch)
            y = y_scale(val)
            points.append(f"{x},{y}")
        color = colors.get(backend, "#000")
        svg.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(points)}' />")
        for batch, val in pts:
            x = x_scale(batch)
            y = y_scale(val)
            svg.append(f"<circle cx='{x}' cy='{y}' r='3' fill='{color}' />")

    lx, ly = x1 - 140, y1 + 10
    for i, backend in enumerate(backends):
        y = ly + i * 16
        color = colors.get(backend, "#000")
        svg.append(f"<rect x='{lx}' y='{y-8}' width='10' height='10' fill='{color}' />")
        svg.append(f"<text x='{lx+16}' y='{y}' font-family='sans-serif' font-size='10'>{backend}</text>")

    svg.append(f"<text x='{w/2}' y='{h-10}' text-anchor='middle' font-family='sans-serif' font-size='11'>q_batch</text>")
    svg.append("</svg>")
    out_path.write_text("\n".join(svg))


def write_speedup_svg(case_rows, out_path):
    cpu = {
        int(r["b"]): float(r["p50_ms"])
        for r in case_rows
        if r["requested_backend"] == "cpu" and r["actual_backend"] == "CPU"
    }
    qnn = {
        int(r["b"]): float(r["p50_ms"])
        for r in case_rows
        if r["requested_backend"] == "qnn" and r["actual_backend"] == "QNN-Graph"
    }
    batches = sorted(set(cpu) & set(qnn))
    if not batches:
        return

    points = [(batch, cpu[batch] / qnn[batch]) for batch in batches if qnn[batch] > 0]
    if not points:
        return

    w, h = 640, 320
    margin = 60
    plot_w = w - 2 * margin
    plot_h = h - 2 * margin
    vals = [val for _, val in points]
    y_min = 0.0
    y_max = max(max(vals) * 1.15, 1.25)

    def x_scale(batch):
        idx = batches.index(batch)
        if len(batches) == 1:
            return margin + plot_w / 2
        return margin + (plot_w * idx / (len(batches) - 1))

    def y_scale(val):
        return margin + plot_h - (plot_h * (val - y_min) / (y_max - y_min))

    svg = []
    svg.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>")
    svg.append("<rect width='100%' height='100%' fill='white' />")
    svg.append(
        f"<text x='{w/2}' y='28' text-anchor='middle' font-family='sans-serif' font-size='16'>{case_title(case_rows)}: CPU/QNN speedup</text>"
    )

    x0, y0 = margin, margin + plot_h
    x1, y1 = margin + plot_w, margin
    svg.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#333' stroke-width='1' />")
    svg.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#333' stroke-width='1' />")

    for i in range(6):
        v = y_min + (y_max - y_min) * i / 5
        y = y_scale(v)
        svg.append(f"<line x1='{x0-4}' y1='{y}' x2='{x0}' y2='{y}' stroke='#333' stroke-width='1' />")
        svg.append(f"<text x='{x0-8}' y='{y+4}' text-anchor='end' font-family='sans-serif' font-size='10'>{v:.1f}x</text>")
        svg.append(f"<line x1='{x0}' y1='{y}' x2='{x1}' y2='{y}' stroke='#eee' stroke-width='1' />")

    y_one = y_scale(1.0)
    svg.append(f"<line x1='{x0}' y1='{y_one}' x2='{x1}' y2='{y_one}' stroke='#d62728' stroke-dasharray='4 4' stroke-width='1' />")
    svg.append(f"<text x='{x1}' y='{y_one-6}' text-anchor='end' font-family='sans-serif' font-size='10' fill='#d62728'>1.0x = parity</text>")

    for batch, val in points:
        x = x_scale(batch)
        y = y_scale(val)
        bar_h = y0 - y
        svg.append(f"<rect x='{x-14}' y='{y}' width='28' height='{bar_h}' fill='#2ca02c' opacity='0.85' />")
        svg.append(f"<text x='{x}' y='{y-6}' text-anchor='middle' font-family='sans-serif' font-size='10'>{val:.2f}x</text>")
        svg.append(f"<line x1='{x}' y1='{y0}' x2='{x}' y2='{y0+4}' stroke='#333' stroke-width='1' />")
        svg.append(f"<text x='{x}' y='{y0+18}' text-anchor='middle' font-family='sans-serif' font-size='10'>{batch}</text>")

    svg.append(f"<text x='{w/2}' y='{h-10}' text-anchor='middle' font-family='sans-serif' font-size='11'>q_batch</text>")
    svg.append(f"<text transform='translate(16 {h/2}) rotate(-90)' text-anchor='middle' font-family='sans-serif' font-size='11'>CPU latency / QNN latency</text>")
    svg.append("</svg>")
    out_path.write_text("\n".join(svg))


def write_report(rows, out_path):
    cases = []
    for case in sorted({r["case"] for r in rows}):
        case_rows = [r for r in rows if r["case"] == case]
        if case_rows:
            cases.append((case, case_rows))

    lines = [
        "# Benchmark Report",
        "",
        "Real Tachyon Particle results for CPU vs QNN using the current direct-QNN path.",
        "",
    ]
    for case, case_rows in cases:
        n = int(case_rows[0]["n"])
        d = int(case_rows[0]["d"])
        title = case_title(case_rows)
        cpu = {
            int(r["b"]): float(r["p50_ms"])
            for r in case_rows
            if r["requested_backend"] == "cpu" and r["actual_backend"] == "CPU"
        }
        qnn = {
            int(r["b"]): float(r["p50_ms"])
            for r in case_rows
            if r["requested_backend"] == "qnn" and r["actual_backend"] == "QNN-Graph"
        }
        batches = sorted(set(cpu) & set(qnn))

        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- Shape: `n={n}`, `d={d}`")
        lines.append("- Metric: `cosine`")
        lines.append("- Backends: `cpu`, `qnn`")
        if batches:
            best_batch = max(batches, key=lambda batch: cpu[batch] / qnn[batch] if qnn[batch] > 0 else 0.0)
            best_speedup = cpu[best_batch] / qnn[best_batch]
            singleton_batch = min(batches)
            singleton_speedup = cpu[singleton_batch] / qnn[singleton_batch]
            lines.append(f"- QNN speedup at `q_batch={singleton_batch}`: `{singleton_speedup:.2f}x`")
            lines.append(f"- Best observed QNN speedup: `{best_speedup:.2f}x` at `q_batch={best_batch}`")
        lines.append("")
        lines.append(f"![{title} latency](bench_charts/{case}_latency.svg)")
        lines.append("")
        if batches:
            lines.append(f"![{title} speedup](bench_charts/{case}_speedup.svg)")
            lines.append("")
        lines.append("batch | cpu p50 ms | qnn p50 ms | cpu/qnn speedup")
        lines.append("---|---:|---:|---:")
        for batch in batches:
            speedup = cpu[batch] / qnn[batch] if qnn[batch] > 0 else 0.0
            lines.append(f"{batch} | {cpu[batch]:.2f} | {qnn[batch]:.2f} | {speedup:.2f}x")
        lines.append("")

    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run benchmark sweeps and emit CSV + SVG charts.")
    parser.add_argument("--out", default="benchmarks", help="Output directory (default: benchmarks)")
    parser.add_argument("--bench", default=None, help="Path to tachann-bench binary")
    parser.add_argument("--release", action="store_true", help="Use target/release/tachann-bench")
    parser.add_argument("--case", action="append", help="Case like 1024x128 or 1024x128:5 (iters)")
    parser.add_argument("--batches", default="1,2,4,8,16,32", help="Comma-separated q_batch values")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=None, help="Override iters for all cases")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="cosine", help="cosine | l2 | ip")
    parser.add_argument("--max-batch", type=int, default=4096, help="Max batch for distance()")
    parser.add_argument(
        "--backends",
        default="cpu",
        help="Comma-separated requested backends to run: cpu, ort, qnn, auto",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "bench_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if args.bench:
        bench = Path(args.bench)
    else:
        bench = Path("target/release/tachann-bench" if args.release else "target/debug/tachann-bench")

    if not bench.exists():
        raise SystemExit(f"bench binary not found: {bench}")

    if args.case:
        cases = [parse_case(c) for c in args.case]
    else:
        cases = [
            {"name": "n1024_d128", "n": 1024, "d": 128, "iters": 5},
            {"name": "n8192_d128", "n": 8192, "d": 128, "iters": 3},
            {"name": "n1024_d768", "n": 1024, "d": 768, "iters": 5},
            {"name": "n8192_d768", "n": 8192, "d": 768, "iters": 2},
        ]

    batches = [int(x.strip()) for x in args.batches.split(",") if x.strip()]
    requested_backends = [x.strip().lower() for x in args.backends.split(",") if x.strip()]

    rows = []
    for case in cases:
        for batch in batches:
            for requested_backend in requested_backends:
                iters = args.iters if args.iters is not None else case["iters"]
                cmd = [
                    str(bench),
                    "--backend", requested_backend,
                    "--n", str(case["n"]),
                    "--d", str(case["d"]),
                    "--iters", str(iters),
                    "--warmup", str(args.warmup),
                    "--q-batch", str(batch),
                    "--seed", str(args.seed),
                    "--metric", args.metric,
                    "--batch", str(args.max_batch),
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                found = 0
                for line in proc.stdout.splitlines():
                    s = line.strip()
                    if re.match(r"^(CPU|ORT-CPU|ORT-QNN|QNN-Graph)\s", s):
                        parts = s.split()
                        if len(parts) < 8:
                            continue
                        try:
                            p50 = float(parts[1])
                            rows_s = float(parts[6])
                            gops = float(parts[7])
                        except Exception:
                            continue
                        rows.append({
                            "case": case["name"],
                            "n": case["n"],
                            "d": case["d"],
                            "b": batch,
                            "requested_backend": requested_backend,
                            "actual_backend": parts[0],
                            "p50_ms": p50,
                            "rows_s": rows_s,
                            "gops": gops,
                            "iters": iters,
                            "exit": proc.returncode,
                        })
                        found += 1
                if found == 0:
                    print(
                        f"WARN: no results parsed for case={case['name']} b={batch} requested={requested_backend} rc={proc.returncode}"
                    )
                if proc.returncode != 0:
                    print(
                        f"WARN: run failed rc={proc.returncode} case={case['name']} b={batch} requested={requested_backend}"
                    )

    out_csv = out_dir / "bench_results.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "case", "n", "d", "b", "requested_backend", "actual_backend", "p50_ms", "rows_s", "gops", "iters", "exit",
        ])
        for r in rows:
            w.writerow([
                r["case"], r["n"], r["d"], r["b"], r["requested_backend"], r["actual_backend"],
                f"{r['p50_ms']:.3f}", f"{r['rows_s']:.0f}", f"{r['gops']:.3f}", r["iters"], r["exit"],
            ])

    by = {}
    for r in rows:
        by.setdefault((r["case"], r["b"]), []).append(r)

    out_md = out_dir / "bench_summary.md"
    with out_md.open("w") as f:
        f.write("# Benchmark Summary (Best p50)\n\n")
        f.write("case | batch | requested_backend | actual_backend | p50_ms\n")
        f.write("---|---:|---|---|---:\n")
        for key in sorted(by.keys()):
            items = by[key]
            best = min(items, key=lambda x: x["p50_ms"])
            f.write(f"{key[0]} | {key[1]} | {best['requested_backend']} | {best['actual_backend']} | {best['p50_ms']:.3f}\n")

    for case in sorted({r["case"] for r in rows}):
        case_rows = [r for r in rows if r["case"] == case]
        if case_rows:
            write_latency_svg(case_rows, charts_dir / f"{case}_latency.svg")
            write_speedup_svg(case_rows, charts_dir / f"{case}_speedup.svg")

    write_report(rows, out_dir / "REPORT.md")
    print(f"Wrote {out_csv}, {out_md}, and {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
