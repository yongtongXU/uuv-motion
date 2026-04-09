import argparse
import csv
import json
import math
import multiprocessing as mp
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_INIT = ROOT / "config" / "init.json"
DEFAULT_PATH = ROOT / "path.json"
DEFAULT_OUT_DIR = ROOT / "trajectories"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


def get_sample_rate(init_cfg: Dict) -> float:
    scene = init_cfg.get("scene", {})
    hz = scene.get("sample_rate_hz")
    if hz is not None:
        hz = float(hz)
        if hz > 0:
            return hz

    dt = scene.get("time_step")
    if dt is not None:
        dt = float(dt)
        if dt > 0:
            return 1.0 / dt

    return 50.0


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return cleaned or "uuv"


def iter_segment_samples(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    speed_m_s: float,
    dt: float,
) -> Iterable[Tuple[float, float, float, float]]:
    seg_dist = distance(start, end)
    if seg_dist == 0:
        return

    duration = seg_dist / speed_m_s
    steps = max(1, int(math.ceil(duration / dt)))
    yaw_deg = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))

    for k in range(1, steps + 1):
        alpha = k / steps
        x = start[0] + (end[0] - start[0]) * alpha
        y = start[1] + (end[1] - start[1]) * alpha
        z = start[2] + (end[2] - start[2]) * alpha
        yield x, y, z, yaw_deg


def estimate_rows(points: List[Tuple[float, float, float]], speed_m_s: float, dt: float) -> int:
    if not points:
        return 0
    count = 1
    for i in range(len(points) - 1):
        seg_dist = distance(points[i], points[i + 1])
        if seg_dist == 0:
            continue
        steps = max(1, int(math.ceil((seg_dist / speed_m_s) / dt)))
        count += steps
    return count


def simulate_single_uuv_to_csv(
    uuv: Dict,
    speed_m_s: float,
    sample_rate_hz: float,
    out_path: Path,
    flush_rows: int,
    max_rows_per_uuv: int,
) -> None:
    uuv_id = str(uuv.get("id", "uuv"))
    wps = uuv.get("waypoints", [])
    if not wps:
        return

    points: List[Tuple[float, float, float]] = [(float(p["x"]), float(p["y"]), float(p["z"])) for p in wps]
    dt = 1.0 / sample_rate_hz
    row_count = 0
    stopped_early = False

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["uuv_id", "step_idx", "t", "x", "y", "z", "yaw_deg", "target_wp_idx"])

        t = 0.0
        step_idx = 0
        if len(points) > 1:
            first_yaw = math.degrees(math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]))
        else:
            first_yaw = 0.0

        writer.writerow(
            [
                uuv_id,
                step_idx,
                round(t, 3),
                round(points[0][0], 3),
                round(points[0][1], 3),
                round(points[0][2], 3),
                round(first_yaw, 6),
                0,
            ]
        )
        row_count += 1

        for wp_idx in range(len(points) - 1):
            if max_rows_per_uuv > 0 and row_count >= max_rows_per_uuv:
                stopped_early = True
                break
            start = points[wp_idx]
            end = points[wp_idx + 1]
            for x, y, z, yaw_deg in iter_segment_samples(start, end, speed_m_s, dt):
                step_idx += 1
                t += dt
                writer.writerow(
                    [
                        uuv_id,
                        step_idx,
                        round(t, 3),
                        round(x, 3),
                        round(y, 3),
                        round(z, 3),
                        round(yaw_deg, 6),
                        wp_idx + 1,
                    ]
                )
                row_count += 1
                if row_count % flush_rows == 0:
                    f.flush()
                if max_rows_per_uuv > 0 and row_count >= max_rows_per_uuv:
                    stopped_early = True
                    break
            if stopped_early:
                break
        f.flush()


def simulate_multi_process(
    init_path: Path,
    path_path: Path,
    out_dir: Path,
    flush_rows: int = 200000,
    max_rows_per_uuv: int = 0,
) -> None:
    init_cfg = load_json(init_path)
    path_cfg = load_json(path_path)

    sample_rate_hz = get_sample_rate(init_cfg)
    speed_m_s = float(init_cfg.get("uuv_template", {}).get("max_speed_m_s", 1.0))
    if speed_m_s <= 0:
        raise ValueError("init.json: uuv_template.max_speed_m_s must be > 0")

    paths: List[Dict] = path_cfg.get("paths", [])
    if not paths:
        raise ValueError("path.json: missing paths")

    out_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / sample_rate_hz

    est_total_rows = 0
    for uuv in paths:
        points = [(float(p["x"]), float(p["y"]), float(p["z"])) for p in uuv.get("waypoints", [])]
        est_total_rows += estimate_rows(points, speed_m_s, dt)

    print(
        f"start simulation | process_count={len(paths)} | estimated_total_rows={est_total_rows} | "
        f"sample_rate_hz={sample_rate_hz}"
    )

    processes: List[Tuple[str, Path, mp.Process]] = []

    for uuv in paths:
        uuv_id = str(uuv.get("id", "uuv"))
        out_path = out_dir / f"trajectory_{sanitize_filename(uuv_id)}.csv"
        p = mp.Process(
            target=simulate_single_uuv_to_csv,
            args=(
                uuv,
                speed_m_s,
                sample_rate_hz,
                out_path,
                max(1, flush_rows),
                max_rows_per_uuv,
            ),
            name=f"sim_{uuv_id}",
        )
        p.start()
        processes.append((uuv_id, out_path, p))

    failed = 0
    for _uid, _out, p in processes:
        p.join()
        if p.exitcode != 0:
            failed += 1

    results: List[Dict] = []
    for uuv_id, out_path, p in processes:
        rows = 0
        status = "error"
        if out_path.exists():
            with out_path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = max(0, sum(1 for _ in f) - 1)
            status = "done" if p.exitcode == 0 else "error"
        results.append({"uuv_id": uuv_id, "rows": rows, "status": status, "file": str(out_path)})

    results.sort(key=lambda x: x["uuv_id"])
    total_rows = sum(int(r["rows"]) for r in results)
    for r in results:
        print(f"[{r['status']}] {r['uuv_id']} rows={r['rows']} file={r['file']}")

    print(
        f"all done | uuv_count={len(paths)} | total_rows={total_rows} | failed_processes={failed} | out_dir={out_dir}"
    )

    if failed:
        raise RuntimeError(f"{failed} worker process(es) failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multiprocess UUV trajectory simulation to per-UUV CSV files.")
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT, help="init.json path")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH, help="path.json path")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="output directory for per-UUV csv files")
    parser.add_argument("--flush-rows", type=int, default=200000, help="flush file buffer every N rows")
    parser.add_argument(
        "--max-rows-per-uuv",
        type=int,
        default=0,
        help="debug limit for each uuv csv (0 means no limit)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulate_multi_process(
        init_path=args.init,
        path_path=args.path,
        out_dir=args.out_dir,
        flush_rows=args.flush_rows,
        max_rows_per_uuv=args.max_rows_per_uuv,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
