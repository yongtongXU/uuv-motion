import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
INIT_PATH = CONFIG_DIR / "init.json"
PATH_PATH = ROOT / "path.json"


def load_init(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_track_centers(y_min: float, y_max: float, sensor_width: float, max_lines: int) -> Tuple[List[float], int, int]:
    band_width = y_max - y_min
    required_lines = max(1, int(math.ceil(band_width / sensor_width)))
    lines = min(required_lines, max_lines)
    step = band_width / lines
    centers = [y_min + (i + 0.5) * step for i in range(lines)]
    return centers, required_lines, lines


def build_comb_waypoints(
    start_x: float,
    start_y: float,
    start_z: float,
    x_min: float,
    x_max: float,
    y_tracks: List[float],
) -> List[Dict]:
    if not y_tracks:
        return [{"x": round(start_x, 3), "y": round(start_y, 3), "z": round(start_z, 3)}]

    wps: List[Tuple[float, float, float]] = []
    wps.append((start_x, start_y, start_z))
    wps.append((x_min, y_tracks[0], start_z))

    current_x = x_min
    for i, y in enumerate(y_tracks):
        if i > 0:
            wps.append((current_x, y, start_z))
        target_x = x_max if current_x == x_min else x_min
        wps.append((target_x, y, start_z))
        current_x = target_x

    compact: List[Tuple[float, float, float]] = []
    for p in wps:
        if not compact or p != compact[-1]:
            compact.append(p)

    # Return to own start point after finishing coverage.
    if compact[-1] != (start_x, start_y, start_z):
        compact.append((start_x, start_y, start_z))

    return [{"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)} for x, y, z in compact]


def _plan_comb_no_obstacle(cfg: Dict, planner_cfg: Dict, obstacles: List[Dict]) -> Dict:
    area = cfg.get("area", {})
    area_length = float(area.get("length", 0))
    area_width = float(area.get("width", 0))
    if area_length <= 0 or area_width <= 0:
        raise ValueError("init.json: area.length and area.width must be > 0")

    uuvs = cfg.get("uuvs", [])
    if not uuvs:
        raise ValueError("init.json: missing uuvs")

    template = cfg.get("uuv_template", {})
    sensor_width = float(template.get("sonser", template.get("sensor", 100.0)))
    if sensor_width <= 0:
        raise ValueError("init.json: uuv_template.sonser must be > 0")

    max_waypoints_per_uuv = 100
    max_lines = max(1, (max_waypoints_per_uuv - 2) // 2)

    x_min = 0.0
    x_max = area_length
    y_global_min = 0.0
    band_width = area_width / len(uuvs)

    paths = []
    capped_count = 0

    for idx, uuv in enumerate(uuvs):
        uid = uuv.get("id", f"uuv_{idx+1}")
        pose = uuv.get("pose", {})
        sx = float(pose.get("x", 0.0))
        sy = float(pose.get("y", 0.0))
        sz = float(pose.get("z", -20.0))

        y_min = y_global_min + idx * band_width
        y_max = y_global_min + (idx + 1) * band_width

        tracks, required_lines, used_lines = build_track_centers(
            y_min=y_min,
            y_max=y_max,
            sensor_width=sensor_width,
            max_lines=max_lines,
        )
        if required_lines > used_lines:
            capped_count += 1

        waypoints = build_comb_waypoints(
            start_x=sx,
            start_y=sy,
            start_z=sz,
            x_min=x_min,
            x_max=x_max,
            y_tracks=tracks,
        )

        paths.append(
            {
                "id": uid,
                "sensor_width": sensor_width,
                "band": {"y_min": round(y_min, 3), "y_max": round(y_max, 3)},
                "line_count": used_lines,
                "required_line_count": required_lines,
                "waypoint_count": len(waypoints),
                "waypoints": waypoints,
            }
        )

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "planner": {
            "type": planner_cfg.get("type", "comb_no_obstacle"),
            "sensor_width": sensor_width,
            "max_waypoints_per_uuv": max_waypoints_per_uuv,
            "obstacle_count": len(obstacles),
        },
        "area": {"length": area_length, "width": area_width},
        "summary": {
            "uuv_count": len(uuvs),
            "total_waypoints": sum(p["waypoint_count"] for p in paths),
            "max_waypoints_in_single_uuv": max(p["waypoint_count"] for p in paths) if paths else 0,
            "capped_uuv_count": capped_count,
        },
        "paths": paths,
    }


def _plan_comb_with_obstacle(cfg: Dict, planner_cfg: Dict, obstacles: List[Dict]) -> Dict:
    # Placeholder for future obstacle-aware algorithm.
    raise NotImplementedError(
        "planner.type='comb_with_obstacle' is reserved for future obstacle-aware planning. "
        "Please implement obstacle processing with cfg['environment']['obstacles']."
    )


def plan_paths() -> Dict:
    cfg = load_init(INIT_PATH)
    planner_cfg = cfg.get("planner", {})
    planner_type = planner_cfg.get("type", "comb_no_obstacle")
    environment = cfg.get("environment", {})
    obstacles = environment.get("obstacles", [])
    if not isinstance(obstacles, list):
        raise ValueError("init.json: environment.obstacles must be a list")

    planners = {
        "comb_no_obstacle": _plan_comb_no_obstacle,
        "comb_with_obstacle": _plan_comb_with_obstacle,
    }
    planner_fn = planners.get(planner_type)
    if planner_fn is None:
        supported = ", ".join(sorted(planners.keys()))
        raise ValueError(f"Unsupported planner.type='{planner_type}'. Supported: {supported}")
    return planner_fn(cfg, planner_cfg, obstacles)


def main() -> None:
    result = plan_paths()
    with PATH_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(
        f"Planned {result['summary']['uuv_count']} UUVs, "
        f"total_waypoints={result['summary']['total_waypoints']}, "
        f"max_per_uuv={result['summary']['max_waypoints_in_single_uuv']} -> {PATH_PATH.name}"
    )


if __name__ == "__main__":
    main()
