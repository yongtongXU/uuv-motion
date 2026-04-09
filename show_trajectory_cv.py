import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
DEFAULT_INIT = ROOT / "config" / "init.json"
DEFAULT_TRAJ_DIR = ROOT / "trajectories"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def color_palette() -> List[Tuple[int, int, int]]:
    return [
        (66, 135, 245),
        (52, 199, 89),
        (245, 130, 48),
        (220, 53, 69),
        (171, 71, 188),
        (38, 198, 218),
        (255, 193, 7),
        (0, 121, 107),
    ]


def world_to_pixel(
    x: float,
    y: float,
    x_min: float,
    y_min: float,
    scale: float,
    draw_x0: int,
    draw_y1: int,
) -> Tuple[int, int]:
    px = int(draw_x0 + (x - x_min) * scale)
    py = int(draw_y1 - (y - y_min) * scale)
    return px, py


class CsvTrackReader:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.fp = csv_path.open("r", encoding="utf-8-sig", newline="")
        self.reader = csv.DictReader(self.fp)
        self.finished = False
        self._next_point: Optional[Tuple[float, float, float, float]] = None
        self._prime()

    def _prime(self) -> None:
        self._next_point = self._read_one()

    def _read_one(self) -> Optional[Tuple[float, float, float, float]]:
        if self.finished:
            return None
        try:
            row = next(self.reader)
        except StopIteration:
            self.finished = True
            return None
        return (float(row["t"]), float(row["x"]), float(row["y"]), float(row["z"]))

    def peek_point(self) -> Optional[Tuple[float, float, float, float]]:
        return self._next_point

    def pop_point(self) -> Optional[Tuple[float, float, float, float]]:
        p = self._next_point
        if p is None:
            return None
        self._next_point = self._read_one()
        return p

    def close(self) -> None:
        self.fp.close()


def build_readers(traj_dir: Path) -> List[Tuple[str, CsvTrackReader]]:
    files = sorted(traj_dir.glob("trajectory_*.csv"))
    readers: List[Tuple[str, CsvTrackReader]] = []
    for f in files:
        uuv_id = f.stem.replace("trajectory_", "")
        readers.append((uuv_id, CsvTrackReader(f)))
    return readers


def show_trajectories_cv(
    init_path: Path,
    traj_dir: Path,
    width: int,
    height: int,
    tail_length: int,
    playback_stride: int,
    display_fps: float,
    speedup: float,
    hold_final: bool,
) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        raise RuntimeError("opencv-python and numpy are required: pip install opencv-python numpy") from e

    init_cfg = load_json(init_path)
    area = init_cfg.get("area", {})
    area_length = float(area.get("length", 0))
    area_width = float(area.get("width", 0))
    if area_length <= 0 or area_width <= 0:
        raise ValueError("init.json area.length and area.width must be > 0")

    x_min, x_max = 0.0, area_length
    y_min, y_max = 0.0, area_width
    pad = 60

    if width <= 0 or height <= 0:
        ratio = area_length / max(1e-9, area_width)
        base = 900
        if ratio >= 1.0:
            width = base
            height = max(400, int(round(base / ratio)))
        else:
            height = base
            width = max(400, int(round(base * ratio)))

    usable_w = max(10, width - 2 * pad)
    usable_h = max(10, height - 2 * pad)
    scale_x = usable_w / max(1e-9, x_max - x_min)
    scale_y = usable_h / max(1e-9, y_max - y_min)
    scale = min(scale_x, scale_y)
    draw_w = int((x_max - x_min) * scale)
    draw_h = int((y_max - y_min) * scale)
    draw_x0 = (width - draw_w) // 2
    draw_y0 = (height - draw_h) // 2
    draw_x1 = draw_x0 + draw_w
    draw_y1 = draw_y0 + draw_h

    readers = build_readers(traj_dir)
    if not readers:
        raise ValueError(f"no trajectory_*.csv found in {traj_dir}")

    if playback_stride > 0:
        print(
            f"playback config | mode=stride | stride={playback_stride} points/frame | "
            f"display_fps={display_fps:.1f}"
        )
    else:
        print(
            f"playback config | mode=time | display_fps={display_fps:.1f} | speedup={speedup:.1f}x"
        )

    colors = color_palette()
    trails: Dict[str, Deque[Tuple[int, int]]] = {
        uuv_id: deque(maxlen=tail_length) if tail_length > 0 else deque() for uuv_id, _r in readers
    }
    last_pos: Dict[str, Optional[Tuple[int, int]]] = {uuv_id: None for uuv_id, _r in readers}
    latest_t: Dict[str, float] = {uuv_id: 0.0 for uuv_id, _r in readers}
    z_vals: Dict[str, float] = {uuv_id: 0.0 for uuv_id, _r in readers}
    latest_pos: Dict[str, Optional[Tuple[int, int]]] = {uuv_id: None for uuv_id, _r in readers}

    canvas_bg = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.rectangle(canvas_bg, (draw_x0, draw_y0), (draw_x1, draw_y1), (80, 80, 80), 1)
    trail_canvas = canvas_bg.copy()

    def redraw_tail_mode() -> None:
        nonlocal trail_canvas
        trail_canvas = canvas_bg.copy()
        for i, (uuv_id, _reader) in enumerate(readers):
            c = colors[i % len(colors)]
            pts = list(trails[uuv_id])
            if len(pts) >= 2:
                cv2.polylines(trail_canvas, [np.array(pts, dtype=np.int32)], False, c, 2, lineType=cv2.LINE_AA)

    frame_idx = 0
    sim_t = 0.0
    last_canvas = None
    while True:
        active = 0
        for i, (uuv_id, reader) in enumerate(readers):
            got_any = False
            if playback_stride > 0:
                for _ in range(playback_stride):
                    p = reader.pop_point()
                    if p is None:
                        break
                    got_any = True
                    t, x, y, z = p
                    latest_t[uuv_id] = t
                    z_vals[uuv_id] = z
                    px, py = world_to_pixel(x, y, x_min, y_min, scale, draw_x0, draw_y1)
                    latest_pos[uuv_id] = (px, py)
                    if tail_length == 0:
                        prev = last_pos[uuv_id]
                        if prev is not None and prev != (px, py):
                            cv2.line(trail_canvas, prev, (px, py), colors[i % len(colors)], 2, lineType=cv2.LINE_AA)
                        last_pos[uuv_id] = (px, py)
                    else:
                        trails[uuv_id].append((px, py))
            else:
                # Time-based playback: consume all points with t <= current virtual simulation time.
                while True:
                    p = reader.peek_point()
                    if p is None or p[0] > sim_t:
                        break
                    t, x, y, z = reader.pop_point()
                    got_any = True
                    latest_t[uuv_id] = t
                    z_vals[uuv_id] = z
                    px, py = world_to_pixel(x, y, x_min, y_min, scale, draw_x0, draw_y1)
                    latest_pos[uuv_id] = (px, py)
                    prev = last_pos[uuv_id]
                    if tail_length == 0:
                        if prev is not None and prev != (px, py):
                            cv2.line(trail_canvas, prev, (px, py), colors[i % len(colors)], 2, lineType=cv2.LINE_AA)
                        last_pos[uuv_id] = (px, py)
                    else:
                        trails[uuv_id].append((px, py))
            if got_any:
                active += 1

        if active == 0:
            if any(r.peek_point() is not None for _uid, r in readers):
                # no point crossed current sim_t this frame; advance time and continue
                sim_t += max(1.0, speedup) / max(1.0, display_fps)
                continue
            break

        if tail_length > 0:
            redraw_tail_mode()
            for uuv_id in latest_pos:
                pts = trails[uuv_id]
                latest_pos[uuv_id] = pts[-1] if pts else latest_pos[uuv_id]

        canvas = trail_canvas.copy()

        for i, (uuv_id, _reader) in enumerate(readers):
            c = colors[i % len(colors)]
            if latest_pos[uuv_id] is not None:
                cv2.circle(canvas, latest_pos[uuv_id], 4, c, -1, lineType=cv2.LINE_AA)

            cv2.putText(
                canvas,
                f"{uuv_id} t={latest_t[uuv_id]:.1f}s z={z_vals[uuv_id]:.1f}",
                (12, 28 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                c,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            "Trajectory Playback NED (X=North, Y=East, ESC to exit)",
            (12, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"speedup={speedup:.1f}x stride={playback_stride}",
            (width - 340, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("UUV Trajectories", canvas)
        last_canvas = canvas
        frame_idx += 1
        wait_ms = max(1, int(1000.0 / max(1.0, display_fps)))
        key = cv2.waitKey(wait_ms)
        sim_t += max(1.0, speedup) / max(1.0, display_fps)
        if key in (27, ord("q"), ord("Q")):
            break

    if hold_final and last_canvas is not None:
        final_canvas = last_canvas.copy()
        cv2.putText(
            final_canvas,
            "Completed - press any key to close",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("UUV Trajectories", final_canvas)
        cv2.waitKey(0)

    for _uuv_id, r in readers:
        r.close()
    cv2.destroyAllWindows()
    print(f"playback ended, frames={frame_idx}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV playback for all UUV trajectory CSVs.")
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT, help="init.json path")
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR, help="directory of trajectory csv files")
    parser.add_argument("--width", type=int, default=0, help="window width (0 means auto from scene ratio)")
    parser.add_argument("--height", type=int, default=0, help="window height (0 means auto from scene ratio)")
    parser.add_argument(
        "--tail-length",
        type=int,
        default=0,
        help="visible trail length per UUV (0 means keep full trajectory)",
    )
    parser.add_argument(
        "--playback-stride",
        type=int,
        default=0,
        help="read N points each frame per UUV (0 means time-based playback by t)",
    )
    parser.add_argument("--display-fps", type=float, default=30.0, help="render fps")
    parser.add_argument("--speedup", type=float, default=120.0, help="simulation speed multiplier")
    parser.add_argument("--no-hold-final", action="store_true", help="close immediately when playback ends")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_trajectories_cv(
        init_path=args.init,
        traj_dir=args.traj_dir,
        width=args.width,
        height=args.height,
        tail_length=max(0, args.tail_length),
        playback_stride=max(0, args.playback_stride),
        display_fps=max(1.0, args.display_fps),
        speedup=max(1.0, args.speedup),
        hold_final=not args.no_hold_final,
    )


if __name__ == "__main__":
    main()
