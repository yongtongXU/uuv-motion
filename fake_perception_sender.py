import argparse
import json
import math
import queue
import random
import socket
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_INIT = ROOT / "config" / "init.json"


@dataclass
class FakeTarget:
    target_type: int
    lon: float
    lat: float
    depth: float
    bearing_deg: float
    speed: float
    confidence: float
    heading_rate: float


DEFAULT_TARGETS = [
    FakeTarget(144, 120.15802, 22.66370, 98.0, 87.4, 0.60, 0.80, 1.8),
    FakeTarget(141, 120.15962, 22.66364, 99.0, -31.0, 1.20, 0.76, -1.2),
    FakeTarget(144, 120.16595, 22.66352, 86.0, -33.0, 0.85, 0.72, 1.0),
]

EventCallback = Optional[Callable[[Dict], None]]


def load_origin(init_path: Path = DEFAULT_INIT) -> Tuple[float, float]:
    try:
        with init_path.open("r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
        origin = cfg.get("scene", {}).get("origin", {})
        return float(origin.get("longitude", 0.0)), float(origin.get("latitude", 0.0))
    except Exception:
        return 0.0, 0.0


def lon_lat_to_xy(lon: float, lat: float, origin_lon: float, origin_lat: float) -> Tuple[float, float]:
    x = (float(lat) - origin_lat) * 110_540.0
    denom = 111_320.0 * max(0.01, math.cos(math.radians(origin_lat)))
    y = (float(lon) - origin_lon) * denom
    return x, y


def compute_length(message: str) -> str:
    i0 = message.find("$")
    i1 = message.find("*", i0 if i0 >= 0 else 0)
    if i0 < 0 or i1 < i0:
        return message
    return message.replace(",Length,", f",{i1 - i0 + 1},", 1)


def now_time_token() -> str:
    now = time.time()
    return time.strftime("%H:%M:%S", time.localtime(now)) + f":{int((now % 1) * 100):02d}"


def log_tail_token() -> str:
    now = time.time()
    return time.strftime("%H-%M-%S", time.localtime(now)) + f".{int((now % 1) * 1_000_000):06d}"


def make_er_frame(
    targets: List[FakeTarget],
    rng: random.Random,
    uuv_index: int,
    tick: int,
    noise_m: float,
    max_targets: int,
) -> str:
    count = min(max_targets, len(targets))
    visible = targets[:count]
    tokens = ["$ER", "Length", str(len(visible))]
    noise_lat_scale = noise_m / 110_540.0

    for target_id, target in enumerate(visible):
        lat_rad = math.radians(target.lat)
        noise_lon_scale = noise_m / max(1.0, 111_320.0 * math.cos(lat_rad))
        phase = tick * 0.2 + uuv_index * 0.35 + target_id
        drift_lon = math.cos(phase) * noise_lon_scale * 0.35
        drift_lat = math.sin(phase) * noise_lat_scale * 0.35
        lon = target.lon + drift_lon + rng.uniform(-noise_lon_scale, noise_lon_scale)
        lat = target.lat + drift_lat + rng.uniform(-noise_lat_scale, noise_lat_scale)
        bearing = target.bearing_deg + target.heading_rate * tick + rng.uniform(-2.5, 2.5)
        speed = max(0.0, target.speed + rng.uniform(-0.08, 0.08))
        confidence = min(0.99, max(0.1, target.confidence + rng.uniform(-0.05, 0.05)))

        tokens.extend(
            [
                str(target_id),
                str(target.target_type),
                f"{lon:.6f}",
                f"{lat:.6f}",
                f"{target.depth:.1f}",
                f"{bearing:.1f}",
                "0",
                f"{speed:.12g}",
                f"{confidence:.8g}",
                now_time_token(),
                "0",
            ]
        )

    tokens.append(now_time_token() + "*")
    return compute_length(",".join(tokens)) + ":" + log_tail_token() + "\n"


def connect_with_retry(host: str, port: int, timeout_s: float, retry_s: float) -> socket.socket:
    deadline = time.time() + timeout_s
    last_error: Optional[BaseException] = None
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
            sock.settimeout(0.2)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
            time.sleep(retry_s)
    raise ConnectionError(f"connect failed {host}:{port}: {last_error}")


def parse_er_points(frame: str) -> List[Tuple[float, float, int]]:
    text = frame.strip()
    if text.startswith("$"):
        text = text[1:]
    tokens = [t.strip().rstrip("*") for t in text.split(",")]
    if len(tokens) < 4 or tokens[0] != "ER":
        return []
    try:
        count = int(float(tokens[2]))
    except ValueError:
        return []
    out: List[Tuple[float, float, int]] = []
    idx = 3
    for _ in range(count):
        if idx + 11 > len(tokens):
            break
        try:
            target_id = int(float(tokens[idx]))
            lon = float(tokens[idx + 2])
            lat = float(tokens[idx + 3])
        except ValueError:
            idx += 11
            continue
        out.append((lon, lat, target_id))
        idx += 11
    return out


def parse_re_state(message: str, origin_lon: float, origin_lat: float) -> Dict[str, float]:
    text = message.strip()
    if text.startswith("$"):
        text = text[1:]
    tokens = [t.strip().rstrip("*") for t in text.split(",")]
    if len(tokens) < 16 or tokens[0] != "RE":
        return {}
    try:
        lon = float(tokens[7])
        lat = float(tokens[8])
        x, y = lon_lat_to_xy(lon, lat, origin_lon, origin_lat)
        return {
            "lon": lon,
            "lat": lat,
            "x": x,
            "y": y,
            "z": float(tokens[9]),
            "roll": float(tokens[10]),
            "pitch": float(tokens[11]),
            "yaw": float(tokens[12]),
            "speed": float(tokens[13]),
            "sim_time": float(tokens[14]),
            "status_code": float(tokens[15]),
        }
    except ValueError:
        return {}


def drain_re(
    sock: socket.socket,
    stop_event: threading.Event,
    name: str,
    verbose: bool,
    event_callback: EventCallback = None,
    origin_lon: float = 0.0,
    origin_lat: float = 0.0,
) -> None:
    buf = ""
    while not stop_event.is_set():
        try:
            data = sock.recv(4096)
        except socket.timeout:
            continue
        except OSError:
            break
        if not data:
            break
        buf += data.decode("utf-8", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if line:
                if event_callback is not None:
                    event_callback(
                        {
                            "type": "re",
                            "uuv": name,
                            "message": line,
                            "state": parse_re_state(line, origin_lon, origin_lat),
                        }
                    )
                if verbose:
                    print(f"[{name}][RE] {line}")


def run_sender(
    host: str,
    base_port: int,
    uuv_count: int,
    interval_s: float,
    duration_s: float,
    noise_m: float,
    max_targets: int,
    seed: int,
    verbose: bool,
    event_callback: EventCallback = None,
    external_stop_event: Optional[threading.Event] = None,
    origin_lon: float = 0.0,
    origin_lat: float = 0.0,
) -> None:
    rng = random.Random(seed)
    clients: List[Tuple[str, socket.socket]] = []
    stop_event = threading.Event()
    reader_threads: List[threading.Thread] = []

    try:
        for i in range(uuv_count):
            name = f"uuv_{i + 1}"
            port = base_port + i
            sock = connect_with_retry(host, port, timeout_s=5.0, retry_s=0.2)
            clients.append((name, sock))
            t = threading.Thread(
                target=drain_re,
                args=(sock, stop_event, name, verbose, event_callback, origin_lon, origin_lat),
                daemon=True,
            )
            t.start()
            reader_threads.append(t)
            print(f"[connect] {name} -> {host}:{port}")
            if event_callback is not None:
                event_callback({"type": "connect", "uuv": name, "host": host, "port": port})

        tick = 0
        started = time.time()
        while (duration_s <= 0 or time.time() - started < duration_s) and not (
            external_stop_event is not None and external_stop_event.is_set()
        ):
            for uuv_idx, (name, sock) in enumerate(clients, start=1):
                frame = make_er_frame(
                    targets=DEFAULT_TARGETS,
                    rng=rng,
                    uuv_index=uuv_idx,
                    tick=tick,
                    noise_m=noise_m,
                    max_targets=max_targets,
                )
                try:
                    sock.sendall(frame.encode("utf-8"))
                except OSError as exc:
                    print(f"[send][fail] {name}: {exc}", file=sys.stderr)
                    if event_callback is not None:
                        event_callback({"type": "error", "uuv": name, "message": str(exc)})
                    continue
                if event_callback is not None:
                    event_callback(
                        {
                            "type": "er",
                            "uuv": name,
                            "tick": tick,
                            "message": frame.strip(),
                            "points": parse_er_points(frame),
                        }
                    )
                if verbose:
                    print(f"[{name}][ER] {frame.strip()}")
            tick += 1
            time.sleep(max(0.05, interval_s))
    finally:
        stop_event.set()
        for _name, sock in clients:
            try:
                sock.close()
            except Exception:
                pass
        if event_callback is not None:
            event_callback({"type": "stopped"})


class FakePerceptionGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Fake Perception Sender")
        self.root.geometry("1280x780")
        self.event_q: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.stop_requested = False
        self.running = False
        self.stop_event = threading.Event()
        self.stats: Dict[str, Dict[str, int]] = {}
        self.row_status: Dict[str, Dict[str, str]] = {}
        self.uuv_states: Dict[str, Dict[str, float]] = {}
        self.last_points: List[Tuple[str, float, float, int]] = []
        self.colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
        self.origin_lon, self.origin_lat = load_origin()

        self.host_var = tk.StringVar(value="127.0.0.1")
        self.base_port_var = tk.StringVar(value="6001")
        self.uuv_count_var = tk.StringVar(value="6")
        self.interval_var = tk.StringVar(value="1.0")
        self.duration_var = tk.StringVar(value="0")
        self.noise_var = tk.StringVar(value="20")
        self.targets_var = tk.StringVar(value="3")
        self.seed_var = tk.StringVar(value="7")
        self.status_var = tk.StringVar(value="待命")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll_events()

    def _build_ui(self) -> None:
        ctrl = ttk.LabelFrame(self.root, text="假感知发送设置")
        ctrl.pack(fill=tk.X, padx=8, pady=8)
        fields = [
            ("Host", self.host_var, 14),
            ("Base Port", self.base_port_var, 8),
            ("UUV Count", self.uuv_count_var, 8),
            ("Interval(s)", self.interval_var, 8),
            ("Duration(s)", self.duration_var, 8),
            ("Noise(m)", self.noise_var, 8),
            ("Targets", self.targets_var, 8),
            ("Seed", self.seed_var, 8),
        ]
        for i, (label, var, width) in enumerate(fields):
            ttk.Label(ctrl, text=label).grid(row=0, column=i * 2, padx=4, pady=4, sticky="w")
            ttk.Entry(ctrl, textvariable=var, width=width).grid(row=0, column=i * 2 + 1, padx=4, pady=4)
        ttk.Button(ctrl, text="启动发送", command=self.start).grid(row=1, column=0, padx=4, pady=4)
        ttk.Button(ctrl, text="停止发送", command=self.stop).grid(row=1, column=1, padx=4, pady=4)
        ttk.Label(ctrl, textvariable=self.status_var).grid(row=1, column=2, columnspan=4, sticky="w")

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=2)

        table_frame = ttk.LabelFrame(left, text="UUV连接与发送统计")
        table_frame.pack(fill=tk.BOTH, expand=True)
        cols = (
            "uuv",
            "conn_state",
            "send_state",
            "recv_state",
            "sent_er",
            "recv_re",
            "lon",
            "lat",
            "x",
            "y",
            "z",
            "yaw",
            "sim_t",
            "status",
            "last",
        )
        self.table = ttk.Treeview(table_frame, columns=cols, show="headings", height=12)
        for c, w in [
            ("uuv", 70),
            ("conn_state", 80),
            ("send_state", 80),
            ("recv_state", 80),
            ("sent_er", 70),
            ("recv_re", 70),
            ("lon", 95),
            ("lat", 95),
            ("x", 80),
            ("y", 80),
            ("z", 65),
            ("yaw", 70),
            ("sim_t", 70),
            ("status", 60),
            ("last", 90),
        ]:
            self.table.heading(c, text=c)
            self.table.column(c, width=w, anchor=tk.CENTER)
        xscroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.table.xview)
        self.table.configure(xscrollcommand=xscroll.set)
        self.table.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))
        xscroll.pack(fill=tk.X, padx=4, pady=(0, 4))

        uuv_canvas_frame = ttk.LabelFrame(left, text="UUV位置(x/y)")
        uuv_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.uuv_canvas = tk.Canvas(uuv_canvas_frame, bg="#101820", height=220, highlightthickness=0)
        self.uuv_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        canvas_frame = ttk.LabelFrame(right, text="目标观测散点")
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.canvas = tk.Canvas(canvas_frame, bg="#111", height=420, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        log_frame = ttk.LabelFrame(right, text="日志")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=10, wrap="none")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def log(self, line: str) -> None:
        self.log_text.insert(tk.END, line + "\n")
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 200:
            self.log_text.delete("1.0", f"{lines - 199}.0")
        self.log_text.see(tk.END)

    def start(self) -> None:
        if self.running:
            return
        try:
            host = self.host_var.get().strip() or "127.0.0.1"
            base_port = int(self.base_port_var.get().strip())
            uuv_count = max(1, int(self.uuv_count_var.get().strip()))
            interval_s = max(0.05, float(self.interval_var.get().strip()))
            duration_s = max(0.0, float(self.duration_var.get().strip()))
            noise_m = max(0.0, float(self.noise_var.get().strip()))
            max_targets = max(1, int(self.targets_var.get().strip()))
            seed = int(self.seed_var.get().strip())
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        self.stats.clear()
        self.row_status.clear()
        self.uuv_states.clear()
        self.table.delete(*self.table.get_children())
        for i in range(uuv_count):
            uid = f"uuv_{i + 1}"
            self.stats[uid] = {"sent_er": 0, "recv_re": 0}
            self.row_status[uid] = {"conn": "连接中", "send": "待发送", "recv": "待接收"}
            self.table.insert(
                "",
                tk.END,
                iid=uid,
                values=(uid, "连接中", "待发送", "待接收", 0, 0, "-", "-", "-", "-", "-", "-", "-", "-", "-"),
            )
        self.last_points.clear()
        self.draw_points()
        self.draw_uuvs()

        self.running = True
        self.stop_event.clear()
        self.status_var.set("运行中")
        self.worker = threading.Thread(
            target=self._worker,
            args=(host, base_port, uuv_count, interval_s, duration_s, noise_m, max_targets, seed),
            daemon=True,
        )
        self.worker.start()

    def _worker(
        self,
        host: str,
        base_port: int,
        uuv_count: int,
        interval_s: float,
        duration_s: float,
        noise_m: float,
        max_targets: int,
        seed: int,
    ) -> None:
        try:
            run_sender(
                host=host,
                base_port=base_port,
                uuv_count=uuv_count,
                interval_s=interval_s,
                duration_s=duration_s,
                noise_m=noise_m,
                max_targets=max_targets,
                seed=seed,
                verbose=False,
                event_callback=lambda ev: self.event_q.put(ev),
                external_stop_event=self.stop_event,
                origin_lon=self.origin_lon,
                origin_lat=self.origin_lat,
            )
        except Exception as exc:
            self.event_q.put({"type": "fatal", "message": str(exc)})

    def stop(self) -> None:
        self.stop_event.set()
        self.status_var.set("停止中")
        self.log("[stop] stop requested")

    def on_close(self) -> None:
        self.stop_event.set()
        self.root.destroy()

    def _poll_events(self) -> None:
        while True:
            try:
                ev = self.event_q.get_nowait()
            except queue.Empty:
                break
            et = ev.get("type")
            uid = str(ev.get("uuv", ""))
            if et == "connect":
                self._set_row(uid, conn_state="已连接")
                self.log(f"[connect] {uid} -> {ev.get('host')}:{ev.get('port')}")
            elif et == "er":
                stats = self.stats.setdefault(uid, {"sent_er": 0, "recv_re": 0})
                stats["sent_er"] += 1
                self._set_row(uid, send_state="发送中", last=f"tick={ev.get('tick')}")
                self.last_points = [
                    p for p in self.last_points if p[0] != uid
                ] + [(uid, lon, lat, target_id) for lon, lat, target_id in ev.get("points", [])]
                self.draw_points()
            elif et == "re":
                stats = self.stats.setdefault(uid, {"sent_er": 0, "recv_re": 0})
                stats["recv_re"] += 1
                state = ev.get("state", {}) or {}
                if state:
                    self.uuv_states[uid] = state
                    self.draw_uuvs()
                self._set_row(uid, recv_state="收RE")
            elif et == "error":
                self._set_row(uid, conn_state="异常", send_state="异常")
                self.log(f"[error] {uid} {ev.get('message')}")
            elif et == "fatal":
                self.running = False
                self.status_var.set("异常")
                self.log(f"[fatal] {ev.get('message')}")
            elif et == "stopped":
                self.running = False
                self.status_var.set("已停止")
                for row_uid in self.row_status:
                    self._set_row(row_uid, conn_state="已断开", send_state="已停止")
                self.log("[stopped] sender stopped")
        self.root.after(100, self._poll_events)

    def _set_row(
        self,
        uid: str,
        conn_state: Optional[str] = None,
        send_state: Optional[str] = None,
        recv_state: Optional[str] = None,
        last: Optional[str] = None,
    ) -> None:
        if not self.table.exists(uid):
            return
        stats = self.stats.setdefault(uid, {"sent_er": 0, "recv_re": 0})
        status = self.row_status.setdefault(uid, {"conn": "-", "send": "-", "recv": "-"})
        if conn_state is not None:
            status["conn"] = conn_state
        if send_state is not None:
            status["send"] = send_state
        if recv_state is not None:
            status["recv"] = recv_state
        vals = list(self.table.item(uid, "values"))
        vals[1] = status["conn"]
        vals[2] = status["send"]
        vals[3] = status["recv"]
        vals[4] = stats["sent_er"]
        vals[5] = stats["recv_re"]
        uuv_state = self.uuv_states.get(uid, {})
        if uuv_state:
            vals[6] = f"{float(uuv_state.get('lon', 0.0)):.6f}"
            vals[7] = f"{float(uuv_state.get('lat', 0.0)):.6f}"
            vals[8] = f"{float(uuv_state.get('x', 0.0)):.2f}"
            vals[9] = f"{float(uuv_state.get('y', 0.0)):.2f}"
            vals[10] = f"{float(uuv_state.get('z', 0.0)):.2f}"
            vals[11] = f"{float(uuv_state.get('yaw', 0.0)):.2f}"
            vals[12] = f"{float(uuv_state.get('sim_time', 0.0)):.2f}"
            vals[13] = f"{float(uuv_state.get('status_code', 0.0)):.0f}"
        if last is not None:
            vals[14] = last
        self.table.item(uid, values=tuple(vals))

    def draw_points(self) -> None:
        self.canvas.delete("all")
        if not self.last_points:
            self.canvas.create_text(20, 20, text="等待ER发送", fill="#ddd", anchor="w")
            return
        w = max(200, self.canvas.winfo_width())
        h = max(200, self.canvas.winfo_height())
        lons = [p[1] for p in self.last_points]
        lats = [p[2] for p in self.last_points]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        pad = 32
        lon_span = max(1e-6, max_lon - min_lon)
        lat_span = max(1e-6, max_lat - min_lat)
        for uid, lon, lat, target_id in self.last_points:
            uuv_idx = int("".join(ch for ch in uid if ch.isdigit()) or "1") - 1
            color = self.colors[uuv_idx % len(self.colors)]
            x = pad + (lon - min_lon) / lon_span * (w - pad * 2)
            y = h - pad - (lat - min_lat) / lat_span * (h - pad * 2)
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline="")
            self.canvas.create_text(x + 7, y, text=f"{uid}/T{target_id}", fill=color, anchor="w")

    def draw_uuvs(self) -> None:
        self.uuv_canvas.delete("all")
        if not self.uuv_states:
            self.uuv_canvas.create_text(20, 20, text="等待RE中的UUV状态", fill="#ddd", anchor="w")
            return
        w = max(200, self.uuv_canvas.winfo_width())
        h = max(160, self.uuv_canvas.winfo_height())
        points = [(uid, s.get("x", 0.0), s.get("y", 0.0)) for uid, s in self.uuv_states.items()]
        xs = [float(p[1]) for p in points]
        ys = [float(p[2]) for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad = 28
        x_span = max(1.0, max_x - min_x)
        y_span = max(1.0, max_y - min_y)
        self.uuv_canvas.create_text(
            12,
            12,
            text=f"origin lon/lat={self.origin_lon:.6f},{self.origin_lat:.6f}",
            fill="#aab",
            anchor="w",
        )
        for uid, x_m, y_m in points:
            uuv_idx = int("".join(ch for ch in uid if ch.isdigit()) or "1") - 1
            color = self.colors[uuv_idx % len(self.colors)]
            px = pad + (float(y_m) - min_y) / y_span * (w - pad * 2)
            py = h - pad - (float(x_m) - min_x) / x_span * (h - pad * 2)
            self.uuv_canvas.create_oval(px - 6, py - 6, px + 6, py + 6, fill=color, outline="")
            self.uuv_canvas.create_text(px + 8, py, text=f"{uid} x={x_m:.1f} y={y_m:.1f}", fill=color, anchor="w")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fake perception TCP client that sends ER frames to perception servers.")
    parser.add_argument("--gui", action="store_true", help="open graphical fake sender")
    parser.add_argument("--host", default="127.0.0.1", help="perception TCP server host")
    parser.add_argument("--base-port", type=int, default=6001, help="uuv_1 TCP server port; others use base+i")
    parser.add_argument("--uuv-count", type=int, default=6, help="number of fake perception clients")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between ER sends")
    parser.add_argument("--duration", type=float, default=0.0, help="run seconds; 0 means until Ctrl+C")
    parser.add_argument("--noise-m", type=float, default=20.0, help="position noise radius in meters")
    parser.add_argument("--targets", type=int, default=3, help="number of fake targets per ER frame")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--verbose", action="store_true", help="print sent ER and received RE messages")
    return parser.parse_args()


def run_gui() -> None:
    root = tk.Tk()
    FakePerceptionGui(root)
    root.mainloop()


def main() -> int:
    args = parse_args()
    if args.gui:
        run_gui()
        return 0
    try:
        run_sender(
            host=args.host,
            base_port=args.base_port,
            uuv_count=max(1, args.uuv_count),
            interval_s=max(0.05, args.interval),
            duration_s=max(0.0, args.duration),
            noise_m=max(0.0, args.noise_m),
            max_targets=max(1, args.targets),
            seed=args.seed,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\nstopped")
    except Exception as exc:
        print(f"fake perception failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
