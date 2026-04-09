import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_PATH_JSON = ROOT / "path.json"
DEFAULT_OUT = ROOT / "planned_paths.png"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def extract_xy(waypoints: List[Dict]) -> Tuple[List[float], List[float]]:
    xs = [float(p["x"]) for p in waypoints]
    ys = [float(p["y"]) for p in waypoints]
    return xs, ys


def draw_planned_paths(path_json: Path, save_to: Path, show_window: bool) -> None:
    try:
        import matplotlib
        if not show_window:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("matplotlib is required. Install it with: pip install matplotlib") from e

    cfg = load_json(path_json)
    paths = cfg.get("paths", [])
    if not paths:
        raise ValueError(f"{path_json} has no 'paths'")

    area = cfg.get("area", {})
    area_length = float(area.get("length", 0))
    area_width = float(area.get("width", 0))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    cmap = plt.get_cmap("tab10")

    for idx, p in enumerate(paths):
        uuv_id = str(p.get("id", f"uuv_{idx+1}"))
        wps = p.get("waypoints", [])
        if len(wps) < 1:
            continue
        xs, ys = extract_xy(wps)
        color = cmap(idx % 10)
        ax.plot(xs, ys, "-", lw=1.8, color=color, label=f"{uuv_id} ({len(wps)} wp)")
        ax.scatter(xs[0], ys[0], marker="o", s=35, color=color)
        ax.scatter(xs[-1], ys[-1], marker="x", s=45, color=color)

    if area_length > 0 and area_width > 0:
        ax.set_xlim(0, area_length)
        ax.set_ylim(0, area_width)

    ax.set_title("Planned Coverage Paths (All UUVs, NED)")
    ax.set_xlabel("X North (m)")
    ax.set_ylabel("Y East (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    save_to.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_to)
    print(f"saved figure: {save_to}")

    if show_window:
        try:
            plt.show()
        except Exception:
            print("interactive window failed; image already saved.")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show planned paths for all UUVs from path.json.")
    parser.add_argument("--path-json", type=Path, default=DEFAULT_PATH_JSON, help="path.json file")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output image path")
    parser.add_argument("--show", action="store_true", help="open interactive window after saving image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draw_planned_paths(path_json=args.path_json, save_to=args.out, show_window=args.show)


if __name__ == "__main__":
    main()
