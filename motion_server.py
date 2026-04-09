import argparse
import csv
import json
import math
import multiprocessing as mp
import re
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_INIT = ROOT / "config" / "init.json"
DEFAULT_PATH = ROOT / "path.json"
DEFAULT_OUT_DIR = ROOT / "trajectories"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def wrap_deg(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def clamp_abs(v: float, lim: float) -> float:
    return clamp(v, -abs(lim), abs(lim))


def body_to_world_velocity(
    u: float,
    v: float,
    w: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> Tuple[float, float, float]:
    rr = math.radians(roll_deg)
    pr = math.radians(pitch_deg)
    yr = math.radians(yaw_deg)

    cr, sr = math.cos(rr), math.sin(rr)
    cp, sp = math.cos(pr), math.sin(pr)
    cy, sy = math.cos(yr), math.sin(yr)

    r11 = cy * cp
    r12 = cy * sp * sr - sy * cr
    r13 = cy * sp * cr + sy * sr
    r21 = sy * cp
    r22 = sy * sp * sr + cy * cr
    r23 = sy * sp * cr - cy * sr
    r31 = -sp
    r32 = cp * sr
    r33 = cp * cr

    vx = r11 * u + r12 * v + r13 * w
    vy = r21 * u + r22 * v + r23 * w
    vz = r31 * u + r32 * v + r33 * w
    return vx, vy, vz


def world_to_body_velocity(
    vx: float,
    vy: float,
    vz: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> Tuple[float, float, float]:
    rr = math.radians(roll_deg)
    pr = math.radians(pitch_deg)
    yr = math.radians(yaw_deg)

    cr, sr = math.cos(rr), math.sin(rr)
    cp, sp = math.cos(pr), math.sin(pr)
    cy, sy = math.cos(yr), math.sin(yr)

    r11 = cy * cp
    r12 = cy * sp * sr - sy * cr
    r13 = cy * sp * cr + sy * sr
    r21 = sy * cp
    r22 = sy * sp * sr + cy * cr
    r23 = sy * sp * cr - cy * sr
    r31 = -sp
    r32 = cp * sr
    r33 = cp * cr

    # body = R^T * world
    u = r11 * vx + r21 * vy + r31 * vz
    v = r12 * vx + r22 * vy + r32 * vz
    w = r13 * vx + r23 * vy + r33 * vz
    return u, v, w


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
    init_state: Dict,
    max_speed_m_s: float,
    max_accel_m_s2: float,
    max_yaw_rate_deg_s: float,
    max_pitch_rate_deg_s: float,
    max_roll_rate_deg_s: float,
    max_yaw_accel_deg_s2: float,
    max_pitch_accel_deg_s2: float,
    max_roll_accel_deg_s2: float,
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
    pose = init_state.get("pose", {})
    twist = init_state.get("twist", {})

    x = float(points[0][0])
    y = float(points[0][1])
    z = float(points[0][2])
    roll_deg = float(pose.get("roll", 0.0))
    pitch_deg = float(pose.get("pitch", 0.0))
    yaw_deg = float(pose.get("yaw", 0.0))
    u_m_s = float(twist.get("u", 0.0))
    v_m_s = float(twist.get("v", 0.0))
    w_m_s = float(twist.get("w", 0.0))
    p_deg_s = float(twist.get("p", 0.0))
    q_deg_s = float(twist.get("q", 0.0))
    r_deg_s = float(twist.get("r", 0.0))

    max_roll_deg = 35.0

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "uuv_id",
                "step_idx",
                "t",
                "x",
                "y",
                "z",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "u_m_s",
                "v_m_s",
                "w_m_s",
                "p_deg_s",
                "q_deg_s",
                "r_deg_s",
                "du_m_s2",
                "dv_m_s2",
                "dw_m_s2",
                "dp_deg_s2",
                "dq_deg_s2",
                "dr_deg_s2",
                "target_wp_idx",
            ]
        )

        t = 0.0
        step_idx = 0

        writer.writerow(
            [
                uuv_id,
                step_idx,
                round(t, 3),
                round(x, 3),
                round(y, 3),
                round(z, 3),
                round(roll_deg, 6),
                round(pitch_deg, 6),
                round(yaw_deg, 6),
                round(u_m_s, 6),
                round(v_m_s, 6),
                round(w_m_s, 6),
                round(p_deg_s, 6),
                round(q_deg_s, 6),
                round(r_deg_s, 6),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
            ]
        )
        row_count += 1

        for wp_idx in range(len(points) - 1):
            if max_rows_per_uuv > 0 and row_count >= max_rows_per_uuv:
                stopped_early = True
                break

            tx, ty, tz = points[wp_idx + 1]
            seg_dist = distance((x, y, z), (tx, ty, tz))
            max_steps = max(300, int(math.ceil(seg_dist / max(0.1, max_speed_m_s * dt))) * 40)
            seg_step = 0

            while True:
                dx = tx - x
                dy = ty - y
                dz = tz - z
                horiz_dist = math.hypot(dx, dy)
                dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist_3d <= 0.05:
                    x, y, z = tx, ty, tz
                    break
                if seg_step >= max_steps:
                    x, y, z = tx, ty, tz
                    u_m_s = 0.0
                    v_m_s = 0.0
                    w_m_s = 0.0
                    p_deg_s = 0.0
                    q_deg_s = 0.0
                    r_deg_s = 0.0
                    break

                if horiz_dist > 1e-9:
                    desired_yaw_deg = math.degrees(math.atan2(dy, dx))
                else:
                    desired_yaw_deg = yaw_deg
                desired_pitch_deg = math.degrees(math.atan2(-dz, max(1e-9, horiz_dist)))
                desired_pitch_deg = clamp(desired_pitch_deg, -60.0, 60.0)

                yaw_err = wrap_deg(desired_yaw_deg - yaw_deg)
                pitch_err = desired_pitch_deg - pitch_deg

                desired_r_deg_s = clamp_abs(yaw_err / dt, max_yaw_rate_deg_s)
                desired_q_deg_s = clamp_abs(pitch_err / dt, max_pitch_rate_deg_s)
                if abs(u_m_s) > 1e-3:
                    desired_roll_deg = math.degrees(
                        math.atan((u_m_s * math.radians(r_deg_s)) / 9.81)
                    )
                else:
                    desired_roll_deg = 0.0
                desired_roll_deg = clamp(desired_roll_deg, -max_roll_deg, max_roll_deg)
                roll_err = wrap_deg(desired_roll_deg - roll_deg)
                desired_p_deg_s = clamp_abs(roll_err / dt, max_roll_rate_deg_s)

                prev_u, prev_v, prev_w = u_m_s, v_m_s, w_m_s
                prev_p, prev_q, prev_r = p_deg_s, q_deg_s, r_deg_s

                r_deg_s += clamp(
                    desired_r_deg_s - r_deg_s,
                    -max_yaw_accel_deg_s2 * dt,
                    max_yaw_accel_deg_s2 * dt,
                )
                q_deg_s += clamp(
                    desired_q_deg_s - q_deg_s,
                    -max_pitch_accel_deg_s2 * dt,
                    max_pitch_accel_deg_s2 * dt,
                )
                p_deg_s += clamp(
                    desired_p_deg_s - p_deg_s,
                    -max_roll_accel_deg_s2 * dt,
                    max_roll_accel_deg_s2 * dt,
                )
                r_deg_s = clamp_abs(r_deg_s, max_yaw_rate_deg_s)
                q_deg_s = clamp_abs(q_deg_s, max_pitch_rate_deg_s)
                p_deg_s = clamp_abs(p_deg_s, max_roll_rate_deg_s)

                roll_deg = wrap_deg(roll_deg + p_deg_s * dt)
                pitch_deg = clamp(pitch_deg + q_deg_s * dt, -85.0, 85.0)
                yaw_deg = wrap_deg(yaw_deg + r_deg_s * dt)

                desired_speed = min(max_speed_m_s, dist_3d / dt)
                unit_x, unit_y, unit_z = dx / dist_3d, dy / dist_3d, dz / dist_3d
                vx_cmd, vy_cmd, vz_cmd = (
                    desired_speed * unit_x,
                    desired_speed * unit_y,
                    desired_speed * unit_z,
                )
                u_cmd, v_cmd, w_cmd = world_to_body_velocity(
                    vx_cmd, vy_cmd, vz_cmd, roll_deg, pitch_deg, yaw_deg
                )

                u_m_s += clamp(u_cmd - u_m_s, -max_accel_m_s2 * dt, max_accel_m_s2 * dt)
                v_m_s += clamp(v_cmd - v_m_s, -max_accel_m_s2 * dt, max_accel_m_s2 * dt)
                w_m_s += clamp(w_cmd - w_m_s, -max_accel_m_s2 * dt, max_accel_m_s2 * dt)
                speed_norm = math.sqrt(u_m_s * u_m_s + v_m_s * v_m_s + w_m_s * w_m_s)
                if speed_norm > max_speed_m_s and speed_norm > 1e-9:
                    scale = max_speed_m_s / speed_norm
                    u_m_s *= scale
                    v_m_s *= scale
                    w_m_s *= scale

                vx, vy, vz = body_to_world_velocity(
                    u_m_s, v_m_s, w_m_s, roll_deg, pitch_deg, yaw_deg
                )
                x += vx * dt
                y += vy * dt
                z += vz * dt

                if distance((x, y, z), (tx, ty, tz)) <= 0.05:
                    x, y, z = tx, ty, tz

                step_idx += 1
                t += dt
                du = (u_m_s - prev_u) / dt
                dv = (v_m_s - prev_v) / dt
                dw = (w_m_s - prev_w) / dt
                dp = (p_deg_s - prev_p) / dt
                dq = (q_deg_s - prev_q) / dt
                dr = (r_deg_s - prev_r) / dt
                writer.writerow(
                    [
                        uuv_id,
                        step_idx,
                        round(t, 3),
                        round(x, 3),
                        round(y, 3),
                        round(z, 3),
                        round(roll_deg, 6),
                        round(pitch_deg, 6),
                        round(yaw_deg, 6),
                        round(u_m_s, 6),
                        round(v_m_s, 6),
                        round(w_m_s, 6),
                        round(p_deg_s, 6),
                        round(q_deg_s, 6),
                        round(r_deg_s, 6),
                        round(du, 6),
                        round(dv, 6),
                        round(dw, 6),
                        round(dp, 6),
                        round(dq, 6),
                        round(dr, 6),
                        wp_idx + 1,
                    ]
                )
                row_count += 1
                seg_step += 1
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
    template = init_cfg.get("uuv_template", {})
    max_speed_m_s = float(template.get("max_speed_m_s", 1.0))
    if max_speed_m_s <= 0:
        raise ValueError("init.json: uuv_template.max_speed_m_s must be > 0")
    max_accel_m_s2 = float(template.get("max_accel_m_s2", 0.6))
    if max_accel_m_s2 <= 0:
        raise ValueError("init.json: uuv_template.max_accel_m_s2 must be > 0")

    max_yaw_rate_rad_s = float(template.get("max_yaw_rate_rad_s", 0.35))
    max_yaw_rate_deg_s = abs(math.degrees(max_yaw_rate_rad_s))
    if max_yaw_rate_deg_s <= 0:
        raise ValueError("init.json: uuv_template.max_yaw_rate_rad_s must be > 0")
    max_pitch_rate_rad_s = float(template.get("max_pitch_rate_rad_s", max_yaw_rate_rad_s))
    max_pitch_rate_deg_s = abs(math.degrees(max_pitch_rate_rad_s))
    if max_pitch_rate_deg_s <= 0:
        raise ValueError("init.json: uuv_template.max_pitch_rate_rad_s must be > 0")
    if "max_roll_rate_rad_s" in template:
        max_roll_rate_deg_s = abs(math.degrees(float(template.get("max_roll_rate_rad_s", 0.0))))
    else:
        max_roll_rate_deg_s = max_pitch_rate_deg_s
    if max_roll_rate_deg_s <= 0:
        raise ValueError("init.json: uuv_template.max_roll_rate_rad_s must be > 0 when provided")

    if "max_yaw_accel_deg_s2" in template:
        max_yaw_accel_deg_s2 = abs(float(template.get("max_yaw_accel_deg_s2", 0.0)))
    elif "max_yaw_accel_rad_s2" in template:
        max_yaw_accel_deg_s2 = abs(math.degrees(float(template.get("max_yaw_accel_rad_s2", 0.0))))
    else:
        # Fallback when no explicit yaw-accel is configured.
        max_yaw_accel_deg_s2 = max(30.0, 2.0 * max_yaw_rate_deg_s)
    if "max_pitch_accel_deg_s2" in template:
        max_pitch_accel_deg_s2 = abs(float(template.get("max_pitch_accel_deg_s2", 0.0)))
    elif "max_pitch_accel_rad_s2" in template:
        max_pitch_accel_deg_s2 = abs(math.degrees(float(template.get("max_pitch_accel_rad_s2", 0.0))))
    else:
        max_pitch_accel_deg_s2 = max(20.0, 2.0 * max_pitch_rate_deg_s)
    if "max_roll_accel_deg_s2" in template:
        max_roll_accel_deg_s2 = abs(float(template.get("max_roll_accel_deg_s2", 0.0)))
    elif "max_roll_accel_rad_s2" in template:
        max_roll_accel_deg_s2 = abs(math.degrees(float(template.get("max_roll_accel_rad_s2", 0.0))))
    else:
        max_roll_accel_deg_s2 = max(20.0, 2.0 * max_roll_rate_deg_s)

    paths: List[Dict] = path_cfg.get("paths", [])
    if not paths:
        raise ValueError("path.json: missing paths")

    out_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / sample_rate_hz
    init_uuv_map = {str(u.get("id", "")): u for u in init_cfg.get("uuvs", [])}

    est_total_rows = 0
    for uuv in paths:
        points = [(float(p["x"]), float(p["y"]), float(p["z"])) for p in uuv.get("waypoints", [])]
        est_total_rows += estimate_rows(points, max_speed_m_s, dt)

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
                init_uuv_map.get(uuv_id, {}),
                max_speed_m_s,
                max_accel_m_s2,
                max_yaw_rate_deg_s,
                max_pitch_rate_deg_s,
                max_roll_rate_deg_s,
                max_yaw_accel_deg_s2,
                max_pitch_accel_deg_s2,
                max_roll_accel_deg_s2,
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
