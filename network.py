import argparse
import csv
import json
import math
import multiprocessing as mp
import re
import socket
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_INIT = ROOT / "config" / "init.json"
DEFAULT_PATH = ROOT / "path.json"
DEFAULT_FORMAT = ROOT / "config" / "network_format.json"
DEFAULT_ENDPOINTS = ROOT / "config" / "network_endpoints.json"
DEFAULT_TRAJ_DIR = ROOT / "trajectories"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_endpoint_config(path: Path) -> Dict[str, Tuple[str, int]]:
    if not path.exists():
        return {}
    raw = load_json(path)
    out: Dict[str, Tuple[str, int]] = {}
    endpoints = raw.get("endpoints", {})
    for uuv_id, conf in endpoints.items():
        try:
            out[str(uuv_id)] = (str(conf["ip"]), int(conf["port"]))
        except Exception:
            continue
    return out


def save_endpoint_config(path: Path, endpoint_map: Dict[str, Tuple[str, int]]) -> None:
    data = {
        "schema_version": "1.0",
        "endpoints": {uid: {"ip": ip, "port": int(port)} for uid, (ip, port) in endpoint_map.items()},
    }
    save_json(path, data)


def test_tcp_connection(host: str, port: int, timeout_s: float = 1.5) -> Tuple[bool, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect((host, int(port)))
        return True, "ok"
    except Exception as e:
        return False, str(e)
    finally:
        sock.close()


def get_sample_rate_hz(init_cfg: Dict) -> float:
    scene = init_cfg.get("scene", {})
    hz = scene.get("sample_rate_hz")
    if hz is not None and float(hz) > 0:
        return float(hz)
    dt = scene.get("time_step")
    if dt is not None and float(dt) > 0:
        return 1.0 / float(dt)
    return 50.0


def waypoint_to_tuple(p: Dict) -> Tuple[float, float, float]:
    return float(p["x"]), float(p["y"]), float(p["z"])


def calc_yaw(curr: Tuple[float, float, float], nxt: Tuple[float, float, float]) -> float:
    dx = nxt[0] - curr[0]
    dy = nxt[1] - curr[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.atan2(dy, dx)


def default_format_config() -> Dict:
    return {
        "separator": ",",
        "fields": ["$", "uuv_id", "x", "y", "z", "pitch", "roll", "yaw", "time", "*", "&&"],
        "value_formats": {
            "x": "{:.3f}",
            "y": "{:.3f}",
            "z": "{:.3f}",
            "pitch": "{:.6f}",
            "roll": "{:.6f}",
            "yaw": "{:.6f}",
            "u": "{:.6f}",
            "v": "{:.6f}",
            "w": "{:.6f}",
            "p": "{:.6f}",
            "q": "{:.6f}",
            "r": "{:.6f}",
            "time": "{:.3f}",
        },
        "regex_rules": [],
    }


def load_format_config(path: Optional[Path]) -> Dict:
    if path is None:
        return default_format_config()
    p = Path(path)
    if not p.exists():
        return default_format_config()
    cfg = load_json(p)
    base = default_format_config()
    base.update(cfg)
    return base


class MessageFormatter:
    def __init__(self, cfg: Dict):
        self.separator = str(cfg.get("separator", ","))
        self.fields = list(cfg.get("fields", []))
        if not self.fields:
            self.fields = default_format_config()["fields"]
        self.value_formats = dict(cfg.get("value_formats", {}))
        self.regex_rules = list(cfg.get("regex_rules", []))
        self.constants = dict(cfg.get("constants", {}))
        self.auto_length = bool(cfg.get("auto_length", False))
        self.length_field = str(cfg.get("length_field", "Length"))
        self.length_head_prefix = str(cfg.get("length_head_prefix", "$"))
        self.length_end_marker = str(cfg.get("length_end_marker", "*"))

    def _fmt(self, key: str, val) -> str:
        if key in self.value_formats:
            pattern = self.value_formats[key]
            try:
                return pattern.format(val)
            except Exception:
                return str(val)
        return str(val)

    def format_message(self, values: Dict) -> str:
        out_tokens: List[str] = []
        merged = dict(self.constants)
        merged.update(values)

        for token in self.fields:
            t = str(token)
            if t in merged:
                out_tokens.append(self._fmt(t, merged[t]))
                continue

            match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(.*)$", t)
            if match:
                key = match.group(1)
                suffix = match.group(2)
                if key in merged:
                    out_tokens.append(self._fmt(key, merged[key]) + suffix)
                    continue

            out_tokens.append(t)

        msg = self.separator.join(out_tokens)
        if self.auto_length and self.length_field in self.fields:
            # Compute protocol length from first '$' to first '*' (inclusive).
            s = msg
            i0 = s.find(self.length_head_prefix)
            i1 = s.find(self.length_end_marker, i0 if i0 >= 0 else 0)
            if i0 >= 0 and i1 >= i0:
                n = i1 - i0 + 1
                length_value = str(n)
                # Replace only once to avoid touching payload accidentally.
                old = f"{self.separator}{self.length_field}{self.separator}"
                new = f"{self.separator}{length_value}{self.separator}"
                if old in msg:
                    msg = msg.replace(old, new, 1)
                elif msg.startswith(f"{self.length_field}{self.separator}"):
                    msg = msg.replace(f"{self.length_field}{self.separator}", f"{length_value}{self.separator}", 1)

        for rule in self.regex_rules:
            pat = rule.get("pattern")
            repl = rule.get("repl", "")
            if pat:
                msg = re.sub(str(pat), str(repl), msg)
        return msg


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def send_single_uuv_from_csv(
    uuv_id: str,
    csv_path: Path,
    host: str,
    port: int,
    max_frames: int,
    format_cfg: Optional[Dict] = None,
    progress_queue: Optional[mp.Queue] = None,
    stop_event: Optional[mp.Event] = None,
    pause_event: Optional[mp.Event] = None,
    connect_timeout_s: float = 2.0,
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(connect_timeout_s)
    addr = (host, port)
    formatter = MessageFormatter(format_cfg or default_format_config())

    try:
        total = count_csv_rows(csv_path)
    except Exception as e:
        if progress_queue is not None:
            progress_queue.put({"type": "uuv_error", "uuv_id": uuv_id, "error": f"count csv failed: {e}"})
        return

    try:
        sock.connect(addr)
    except Exception as e:
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "uuv_error",
                    "uuv_id": uuv_id,
                    "port": port,
                    "host": host,
                    "error": f"connect failed: {e}",
                }
            )
        sock.close()
        return

    if progress_queue is not None:
        progress_queue.put(
            {"type": "uuv_start", "uuv_id": uuv_id, "port": port, "host": host, "total": total, "status": "待命"}
        )

    sent = 0
    paused_reported = False
    t0 = time.perf_counter()
    prev_t: Optional[float] = None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if stop_event is not None and stop_event.is_set():
                break
            if max_frames > 0 and sent >= max_frames:
                break

            while pause_event is not None and pause_event.is_set():
                if (not paused_reported) and progress_queue is not None:
                    progress_queue.put(
                        {
                            "type": "uuv_paused",
                            "uuv_id": uuv_id,
                            "sent": sent,
                            "total": total,
                            "status": "暂停发送",
                        }
                    )
                paused_reported = True
                time.sleep(0.05)
                if stop_event is not None and stop_event.is_set():
                    break
            if stop_event is not None and stop_event.is_set():
                break
            if paused_reported and progress_queue is not None:
                progress_queue.put(
                    {
                        "type": "uuv_resumed",
                        "uuv_id": uuv_id,
                        "sent": sent,
                        "total": total,
                        "status": "正在发送",
                    }
                )
            paused_reported = False

            x = float(row.get("x", 0.0))
            y = float(row.get("y", 0.0))
            z = float(row.get("z", 0.0))
            yaw = float(row.get("yaw_deg", row.get("yaw", 0.0)))
            sim_t = float(row.get("t", 0.0))
            u_val = float(row.get("u_m_s", row.get("u", 0.0)))
            r_val = float(row.get("r_deg_s", row.get("r", 0.0)))
            dr_val = float(row.get("dr_deg_s2", row.get("dr", 0.0)))

            digits = "".join(ch for ch in uuv_id if ch.isdigit())
            node_index = max(1, int(digits or "1"))

            values = {
                "uuv_id": uuv_id,
                "Head": "$RS",
                "Length": "Length",
                "node_desc": f"U-3-{node_index}-J",
                "x": x,
                "y": y,
                "z": z,
                "pitch": 0.0,
                "roll": 0.0,
                "yaw": yaw,
                "u": u_val,
                "v": 0.0,
                "w": 0.0,
                "p": 0.0,
                "q": 0.0,
                "r": r_val,
                "du": 0.0,
                "dv": 0.0,
                "dw": 0.0,
                "dp": 0.0,
                "dq": 0.0,
                "dr": dr_val,
                "lat": 0.0,
                "lon": 0.0,
                "Spare_1": 1,
                "Spare_2": 2,
                "Spare_3": 3,
                "Time": time.strftime("%H:%M:%S", time.gmtime(max(0.0, sim_t))),
                "End": "*",
                "time": sim_t,
                "sent": sent,
                "total": total,
                "host": host,
                "port": port,
            }
            msg = formatter.format_message(values)
            try:
                sock.sendall(msg.encode("utf-8"))
            except Exception as e:
                if progress_queue is not None:
                    progress_queue.put(
                        {
                            "type": "uuv_error",
                            "uuv_id": uuv_id,
                            "port": port,
                            "host": host,
                            "error": f"send failed: {e}",
                        }
                    )
                break

            sent += 1
            if progress_queue is not None:
                progress_queue.put(
                    {
                        "type": "uuv_progress",
                        "uuv_id": uuv_id,
                        "sent": sent,
                        "total": total,
                        "x": x,
                        "y": y,
                        "z": z,
                        "yaw": yaw,
                        "t": sim_t,
                        "status": "正在发送",
                    }
                )

            if prev_t is not None:
                dt = sim_t - prev_t
                if dt > 0:
                    time.sleep(dt)
            prev_t = sim_t

    sock.close()
    elapsed = time.perf_counter() - t0
    if progress_queue is not None:
        progress_queue.put(
            {
                "type": "uuv_done",
                "uuv_id": uuv_id,
                "sent": sent,
                "total": total,
                "elapsed": elapsed,
                "status": "发送结束",
            }
        )


def send_path_points(
    init_path: Path,
    path_path: Path,
    host: str,
    base_port: int,
    max_frames: int,
    endpoint_map: Optional[Dict[str, Tuple[str, int]]] = None,
    format_config_path: Optional[Path] = DEFAULT_FORMAT,
    connect_timeout_s: float = 2.0,
    progress_queue: Optional[mp.Queue] = None,
    stop_event: Optional[mp.Event] = None,
    pause_event: Optional[mp.Event] = None,
    source: str = "trajectory",
    traj_dir: Path = DEFAULT_TRAJ_DIR,
) -> None:
    _ = load_json(init_path)
    path_cfg = load_json(path_path)
    format_cfg = load_format_config(format_config_path)

    paths: List[Dict] = path_cfg.get("paths", [])
    if not paths:
        raise ValueError("path.json has no paths")

    workers: List[Tuple[str, str, int, Path]] = []
    for i, p in enumerate(paths):
        uuv_id = str(p.get("id", f"uuv_{i+1}"))
        dst_host = host
        dst_port = base_port + i
        if endpoint_map and uuv_id in endpoint_map:
            dst_host, dst_port = endpoint_map[uuv_id]

        if source == "trajectory":
            csv_path = traj_dir / f"trajectory_{uuv_id}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"trajectory file not found for {uuv_id}: {csv_path}")
        else:
            raise ValueError("only source='trajectory' is supported")

        workers.append((uuv_id, dst_host, int(dst_port), csv_path))

    print(
        f"start tcp send | mode=multiprocess | source={source} | uuv_count={len(workers)} | "
        f"host={host} | base_port={base_port}"
    )
    if progress_queue is not None:
        progress_queue.put(
            {
                "type": "all_start",
                "uuv_count": len(workers),
                "host": host,
                "base_port": base_port,
            }
        )

    procs: List[Tuple[str, int, mp.Process]] = []
    for uuv_id, dst_host, port, csv_path in workers:
        proc = mp.Process(
            target=send_single_uuv_from_csv,
            args=(
                uuv_id,
                csv_path,
                dst_host,
                port,
                max_frames,
                format_cfg,
                progress_queue,
                stop_event,
                pause_event,
                connect_timeout_s,
            ),
            name=f"tcp_{uuv_id}",
        )
        proc.start()
        procs.append((uuv_id, port, proc))

    failed = 0
    for uuv_id, port, proc in procs:
        proc.join()
        if proc.exitcode != 0:
            failed += 1
            if progress_queue is not None:
                progress_queue.put(
                    {"type": "uuv_error", "uuv_id": uuv_id, "port": port, "exitcode": proc.exitcode}
                )

    if progress_queue is not None:
        progress_queue.put({"type": "all_done", "worker_count": len(procs), "failed_processes": failed})
    if failed:
        raise RuntimeError(f"{failed} sender process(es) failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TCP trajectory point sender for all UUVs (multiprocess).")
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT, help="init.json path")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH, help="path.json path")
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR, help="trajectory csv directory")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="receiver host")
    parser.add_argument("--base-port", type=int, default=5000, help="base TCP port; uuv_i uses base_port+i")
    parser.add_argument("--max-frames", type=int, default=0, help="debug max frames to send (0 means full)")
    parser.add_argument("--connect-timeout", type=float, default=2.0, help="TCP connect timeout seconds")
    parser.add_argument(
        "--format-config",
        type=Path,
        default=DEFAULT_FORMAT,
        help="message format config json path",
    )
    parser.add_argument(
        "--endpoint-config",
        type=Path,
        default=DEFAULT_ENDPOINTS,
        help="endpoint json path (uuv_id -> ip/port)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint_map = load_endpoint_config(args.endpoint_config)
    send_path_points(
        init_path=args.init,
        path_path=args.path,
        host=args.host,
        base_port=args.base_port,
        max_frames=max(0, args.max_frames),
        endpoint_map=endpoint_map,
        format_config_path=args.format_config,
        connect_timeout_s=max(0.1, args.connect_timeout),
        source="trajectory",
        traj_dir=args.traj_dir,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
