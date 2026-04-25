import json
import math
import queue
import socket
import threading
import time
import tkinter as tk
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_INIT = ROOT / "config" / "init.json"
DEFAULT_PERCEPTION_CFG = ROOT / "config" / "perception_endpoints.json"
DEFAULT_LOG_DIR = ROOT / "logs" / "perception"


@dataclass
class TargetObservation:
    uuv_id: str
    source_port: int
    target_id: int
    target_type: int
    lon: float
    lat: float
    depth: float
    bearing: float
    speed: float
    confidence: float
    target_time: str
    recv_ts: float


@dataclass
class ClusteredTarget:
    target_id: int
    target_type: int
    lon: float
    lat: float
    depth: float
    bearing: float
    speed: float
    confidence: float
    target_time: str
    source_count: int
    source_uuvs: List[str]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def init_origin(init_path: Path = DEFAULT_INIT) -> Tuple[float, float]:
    try:
        scene = load_json(init_path).get("scene", {})
        origin = scene.get("origin", {})
        return float(origin.get("longitude", 0.0)), float(origin.get("latitude", 0.0))
    except Exception:
        return 0.0, 0.0


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(str(value).strip().rstrip("*"))
    except Exception:
        return None


def _safe_int(value: str) -> Optional[int]:
    try:
        return int(float(str(value).strip().rstrip("*")))
    except Exception:
        return None


def _decode_delimiter(value: str) -> str:
    if value == "\\n":
        return "\n"
    if value == "\\r\\n":
        return "\r\n"
    return value


def parse_er_frame(frame: str, uuv_id: str, source_port: int, recv_ts: Optional[float] = None) -> List[TargetObservation]:
    # print("er recv")
    text = frame.strip()
    # print(text)
    if not text:
        return []
    if text.endswith("&&"):
        text = text[:-2]
    if text.startswith("$"):
        text = text[1:]
    tokens = [t.strip() for t in text.split(",")]
    if len(tokens) < 4 or tokens[0].upper() != "ER":
        return []

    count = _safe_int(tokens[2])
    if count is None or count <= 0:
        return []

    out: List[TargetObservation] = []
    now = time.time() if recv_ts is None else recv_ts
    idx = 3
    record_len = 11
    for _ in range(count):
        if idx + record_len > len(tokens):
            break
        rec = tokens[idx : idx + record_len]
        target_id = _safe_int(rec[0])
        target_type = _safe_int(rec[1])
        lon = _safe_float(rec[2])
        lat = _safe_float(rec[3])
        depth = _safe_float(rec[4])
        bearing = _safe_float(rec[5])
        speed = _safe_float(rec[7])
        confidence = _safe_float(rec[8])
        target_time = rec[9]
        if None not in (target_id, target_type, lon, lat, depth, bearing, speed, confidence):
            out.append(
                TargetObservation(
                    uuv_id=uuv_id,
                    source_port=source_port,
                    target_id=int(target_id),
                    target_type=int(target_type),
                    lon=float(lon),
                    lat=float(lat),
                    depth=float(depth),
                    bearing=float(bearing),
                    speed=float(speed),
                    confidence=float(confidence),
                    target_time=target_time,
                    recv_ts=now,
                )
            )
        idx += record_len
    return out


def _meters_xy(points: List[TargetObservation]) -> List[Tuple[float, float]]:
    if not points:
        return []
    lat0 = math.radians(sum(p.lat for p in points) / len(points))
    lon0 = sum(p.lon for p in points) / len(points)
    base_lat = sum(p.lat for p in points) / len(points)
    xy = []
    for p in points:
        x = (p.lon - lon0) * 111_320.0 * math.cos(lat0)
        y = (p.lat - base_lat) * 110_540.0
        xy.append((x, y))
    return xy


def dbscan_labels(points: List[TargetObservation], eps_m: float, min_samples: int) -> List[int]:
    if not points:
        return []
    xy = _meters_xy(points)
    labels = [-99] * len(points)
    cluster_id = 0

    def neighbors(i: int) -> List[int]:
        x0, y0 = xy[i]
        return [
            j
            for j, (x1, y1) in enumerate(xy)
            if math.hypot(x0 - x1, y0 - y1) <= eps_m
        ]

    for i in range(len(points)):
        if labels[i] != -99:
            continue
        seed = neighbors(i)
        if len(seed) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        k = 0
        while k < len(seed):
            j = seed[k]
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] != -99:
                k += 1
                continue
            labels[j] = cluster_id
            expanded = neighbors(j)
            if len(expanded) >= min_samples:
                for n in expanded:
                    if n not in seed:
                        seed.append(n)
            k += 1
        cluster_id += 1
    return labels


def cluster_observations(points: List[TargetObservation], eps_m: float, min_samples: int) -> List[ClusteredTarget]:
    labels = dbscan_labels(points, eps_m=max(0.1, eps_m), min_samples=max(1, min_samples))
    groups: Dict[int, List[TargetObservation]] = {}
    singleton_label = max([l for l in labels if l >= 0], default=-1) + 1
    for label, point in zip(labels, points):
        if label < 0:
            label = singleton_label
            singleton_label += 1
        groups.setdefault(label, []).append(point)

    clusters: List[ClusteredTarget] = []
    for target_id, group in enumerate(groups.values(), start=1):
        latest = max(group, key=lambda p: p.recv_ts)
        target_type = Counter(p.target_type for p in group).most_common(1)[0][0]
        source_uuvs = sorted({p.uuv_id for p in group})
        clusters.append(
            ClusteredTarget(
                target_id=target_id,
                target_type=target_type,
                lon=sum(p.lon for p in group) / len(group),
                lat=sum(p.lat for p in group) / len(group),
                depth=sum(p.depth for p in group) / len(group),
                bearing=sum(p.bearing for p in group) / len(group),
                speed=sum(p.speed for p in group) / len(group),
                confidence=sum(p.confidence for p in group) / len(group),
                target_time=latest.target_time,
                source_count=len(group),
                source_uuvs=source_uuvs,
            )
        )
    clusters.sort(key=lambda c: c.target_id)
    return clusters


def _compute_length(message: str) -> str:
    i0 = message.find("$")
    i1 = message.find("*", i0 if i0 >= 0 else 0)
    if i0 < 0 or i1 < i0:
        return message
    return message.replace(",Length,", f",{i1 - i0 + 1},", 1)


def format_tt_message(clusters: List[ClusteredTarget]) -> str:
    now = time.time()
    stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1_000_000):06d}"
    tail = time.strftime("%H-%M-%S", time.localtime(now)) + f".{int((now % 1) * 1_000_000):06d}"
    tokens: List[str] = ["$TT", "Length", str(len(clusters))]
    for c in clusters:
        tokens.extend(
            [
                str(c.target_id),
                str(c.target_type),
                f"{c.lon:.6f}",
                f"{c.lat:.6f}",
                f"{c.depth:.6f}",
                f"{c.bearing:.1f}",
                "0",
                f"{c.speed:.12g}",
                f"{c.confidence:.8g}",
                c.target_time,
                "0",
            ]
        )
    tokens.extend([stamp, "*&&"])
    return _compute_length(",".join(tokens))


UuvStateProvider = Callable[[str], Dict[str, float]]


def format_re_message(
    uuv_index: int,
    lon: float,
    lat: float,
    depth: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    speed: float = 0.0,
    sim_time: float = 0.0,
    status_code: int = 0,
) -> str:
    now = time.time()
    hhmmss = time.strftime("%H:%M:%S", time.localtime(now))
    stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1_000_000):06d}"
    tail = time.strftime("%H-%M-%S", time.localtime(now)) + f".{int((now % 1) * 1_000_000):06d}"
    tokens = [
        "$RE",
        "Length",
        str(uuv_index),
        "1",
        f"{int(lon)}",
        f"{int(lat)}",
        hhmmss,
        f"{lon:.6f}",
        f"{lat:.6f}",
        f"{depth:.3f}",
        f"{roll:.3f}",
        f"{pitch:.3f}",
        f"{yaw:.3f}",
        f"{speed:.3f}",
        f"{sim_time:.3f}",
        str(int(status_code)),
        stamp,
        "*&&",
    ]
    return _compute_length(",".join(tokens))


class PerceptionHub:
    def __init__(self, event_q: queue.Queue, uuv_state_provider: Optional[UuvStateProvider] = None):
        self.event_q = event_q
        self.uuv_state_provider = uuv_state_provider
        self._stop_event = threading.Event()
        self._sockets: List[socket.socket] = []
        self._threads: List[threading.Thread] = []
        self._lock = threading.Lock()
        self._observations: List[TargetObservation] = []
        self.running = False
        self.udp_host = "127.0.0.1"
        self.udp_port = 7000
        self.eps_m = 80.0
        self.min_samples = 1
        self.window_s = 5.0
        self.udp_interval_s = 1.0
        self.re_lon = 0.0
        self.re_lat = 0.0

    def start(
        self,
        endpoints: List[Tuple[str, str, int]],
        delimiter: str,
        udp_host: str,
        udp_port: int,
        eps_m: float,
        min_samples: int,
        window_s: float,
        udp_interval_s: float,
        re_lon: float,
        re_lat: float,
    ) -> None:
        self.stop()
        self._stop_event.clear()
        self.running = True
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.eps_m = eps_m
        self.min_samples = min_samples
        self.window_s = window_s
        self.udp_interval_s = udp_interval_s
        self.re_lon = re_lon
        self.re_lat = re_lat
        for uuv_id, host, port in endpoints:
            thread = threading.Thread(target=self._serve_port, args=(uuv_id, host, port, delimiter), daemon=True)
            thread.start()
            self._threads.append(thread)
        sender = threading.Thread(target=self._udp_loop, daemon=True)
        sender.start()
        self._threads.append(sender)

    def stop(self) -> None:
        self._stop_event.set()
        for sock in self._sockets:
            try:
                sock.close()
            except Exception:
                pass
        self._sockets.clear()
        self._threads.clear()
        self.running = False

    def _serve_port(self, uuv_id: str, host: str, port: int, delimiter: str) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((host, port))
            srv.listen(8)
            srv.settimeout(0.5)
            self._sockets.append(srv)
            self.event_q.put({"type": "listen", "uuv_id": uuv_id, "host": host, "port": port})
        except Exception as exc:
            self.event_q.put({"type": "error", "uuv_id": uuv_id, "port": port, "error": f"bind failed: {exc}"})
            try:
                srv.close()
            except Exception:
                pass
            return

        while not self._stop_event.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.event_q.put({"type": "connect", "uuv_id": uuv_id, "port": port, "peer": f"{addr[0]}:{addr[1]}"})
            thread = threading.Thread(target=self._handle_client, args=(uuv_id, port, conn, addr, delimiter), daemon=True)
            thread.start()
            self._threads.append(thread)
        self.event_q.put({"type": "stopped", "uuv_id": uuv_id, "port": port})

    def _handle_client(self, uuv_id: str, port: int, conn: socket.socket, addr, delimiter: str) -> None:
        peer = f"{addr[0]}:{addr[1]}"
        local_stop = threading.Event()
        re_thread = threading.Thread(target=self._re_loop, args=(uuv_id, conn, local_stop), daemon=True)
        re_thread.start()
        buf = ""
        try:
            while not self._stop_event.is_set():
                data = conn.recv(8192)
                if not data:
                    break
                buf += data.decode("utf-8", errors="replace")
                frames, buf = self._split_frames(buf, delimiter)
                for frame in frames:
                    observations = parse_er_frame(frame, uuv_id=uuv_id, source_port=port)
                    if observations:
                        with self._lock:
                            self._observations.extend(observations)
                        self.event_q.put(
                            {
                                "type": "er",
                                "uuv_id": uuv_id,
                                "port": port,
                                "peer": peer,
                                "count": len(observations),
                                "raw": frame,
                                "recv_ts": time.time(),
                            }
                        )
                    else:
                        self.event_q.put(
                            {"type": "raw", "uuv_id": uuv_id, "port": port, "peer": peer, "raw": frame}
                        )
        except Exception as exc:
            self.event_q.put({"type": "error", "uuv_id": uuv_id, "port": port, "peer": peer, "error": str(exc)})
        finally:
            local_stop.set()
            try:
                conn.close()
            except Exception:
                pass
            self.event_q.put({"type": "disconnect", "uuv_id": uuv_id, "port": port, "peer": peer})

    def _split_frames(self, buf: str, delimiter: str) -> Tuple[List[str], str]:
        frames: List[str] = []
        while True:
            start = buf.find("$")
            if start < 0:
                if len(buf) > 65536:
                    frames.append(buf)
                    buf = ""
                break
            if start > 0:
                buf = buf[start:]
            end = buf.find("&&", 1)
            if end < 0:
                if len(buf) > 65536:
                    frames.append(buf)
                    buf = ""
                break
            frames.append(buf[: end + 2])
            buf = buf[end + 2 :]
        return frames, buf

    def _re_loop(self, uuv_id: str, conn: socket.socket, local_stop: threading.Event) -> None:
        digits = "".join(ch for ch in uuv_id if ch.isdigit())
        uuv_index = max(1, int(digits or "1"))
        while not self._stop_event.is_set() and not local_stop.is_set():
            state = self._re_state_for_uuv(uuv_id)
            msg = format_re_message(
                uuv_index=uuv_index,
                lon=float(state.get("lon", self.re_lon)),
                lat=float(state.get("lat", self.re_lat)),
                depth=float(state.get("z", 0.0)),
                roll=float(state.get("roll", 0.0)),
                pitch=float(state.get("pitch", 0.0)),
                yaw=float(state.get("yaw", 0.0)),
                speed=float(state.get("speed", 0.0)),
                sim_time=float(state.get("t", 0.0)),
                status_code=int(state.get("status_code", 0)),
            )
            try:
                conn.sendall((msg).encode("utf-8"))
                self.event_q.put({"type": "re", "uuv_id": uuv_id, "message": msg})
            except Exception as exc:
                self.event_q.put({"type": "error", "uuv_id": uuv_id, "error": f"RE send failed: {exc}"})
                break
            local_stop.wait(1.0)

    def _re_state_for_uuv(self, uuv_id: str) -> Dict[str, float]:
        if self.uuv_state_provider is None:
            return {"lon": self.re_lon, "lat": self.re_lat}
        try:
            state = self.uuv_state_provider(uuv_id)
        except Exception as exc:
            self.event_q.put({"type": "error", "uuv_id": uuv_id, "error": f"UUV state link failed: {exc}"})
            return {"lon": self.re_lon, "lat": self.re_lat}
        if not state:
            return {"lon": self.re_lon, "lat": self.re_lat}
        return state

    def _recent_observations(self) -> List[TargetObservation]:
        cutoff = time.time() - max(0.1, self.window_s)
        with self._lock:
            self._observations = [p for p in self._observations if p.recv_ts >= cutoff]
            return list(self._observations)

    def _udp_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            while not self._stop_event.is_set():
                points = self._recent_observations()
                clusters = cluster_observations(points, self.eps_m, self.min_samples)
                msg = format_tt_message(clusters)
                try:
                    sock.sendto(msg.encode("utf-8"), (self.udp_host, int(self.udp_port)))
                    self.event_q.put({"type": "tt", "count": len(clusters), "message": msg})
                except Exception as exc:
                    self.event_q.put({"type": "error", "error": f"UDP send failed: {exc}"})
                self._stop_event.wait(max(0.1, self.udp_interval_s))
        finally:
            sock.close()


class PerceptionPage(ttk.Frame):
    def __init__(
        self,
        parent,
        init_path: Path = DEFAULT_INIT,
        uuv_state_provider: Optional[UuvStateProvider] = None,
    ):
        super().__init__(parent)
        origin_lon, origin_lat = init_origin(init_path)
        self.event_q: queue.Queue = queue.Queue()
        self.hub = PerceptionHub(self.event_q, uuv_state_provider=uuv_state_provider)
        self.max_log_lines = 200
        self.log_dir = DEFAULT_LOG_DIR
        self.log_files: Dict[str, Path] = {}
        self.running = False
        self.port_rows: Dict[str, str] = {}
        self.uuv_stats: Dict[str, Dict[str, int]] = {}

        self.bind_host_var = tk.StringVar(value="127.0.0.1")
        self.base_port_var = tk.StringVar(value="6001")
        self.delimiter_var = tk.StringVar(value="&&")
        self.udp_host_var = tk.StringVar(value="127.0.0.1")
        self.udp_port_var = tk.StringVar(value="7000")
        self.eps_var = tk.StringVar(value="80")
        self.min_samples_var = tk.StringVar(value="1")
        self.window_var = tk.StringVar(value="5")
        self.udp_interval_var = tk.StringVar(value="1")
        self.re_lon_var = tk.StringVar(value=str(origin_lon))
        self.re_lat_var = tk.StringVar(value=str(origin_lat))
        self.status_var = tk.StringVar(value="待命")
        self.last_tt_var = tk.StringVar(value="TT目标数: 0")
        self.endpoint_uuv_var = tk.StringVar(value="uuv_1")
        self.endpoint_host_var = tk.StringVar(value="127.0.0.1")
        self.endpoint_port_var = tk.StringVar(value="6001")

        self._build_ui()
        self._init_endpoint_rows()
        self._load_config(silent=True)
        self._poll_events()

    def _build_ui(self) -> None:
        ctrl = ttk.LabelFrame(self, text="感知接收与目标汇总")
        ctrl.pack(fill=tk.X, padx=8, pady=(8, 6))
        labels = [
            ("Bind Host", self.bind_host_var, 14),
            ("Base Port", self.base_port_var, 8),
            ("Delimiter", self.delimiter_var, 8),
            ("UDP Host", self.udp_host_var, 14),
            ("UDP Port", self.udp_port_var, 8),
            ("DBSCAN eps(m)", self.eps_var, 8),
            ("Min Samples", self.min_samples_var, 8),
            ("Window(s)", self.window_var, 8),
            ("UDP Interval(s)", self.udp_interval_var, 8),
        ]
        for i, (label, var, width) in enumerate(labels):
            ttk.Label(ctrl, text=label).grid(row=i // 5, column=(i % 5) * 2, padx=4, pady=4, sticky="w")
            ttk.Entry(ctrl, textvariable=var, width=width).grid(row=i // 5, column=(i % 5) * 2 + 1, padx=4, pady=4)
        ttk.Label(ctrl, text="RE Lon").grid(row=2, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(ctrl, textvariable=self.re_lon_var, width=14).grid(row=2, column=1, padx=4, pady=4)
        ttk.Label(ctrl, text="RE Lat").grid(row=2, column=2, padx=4, pady=4, sticky="w")
        ttk.Entry(ctrl, textvariable=self.re_lat_var, width=14).grid(row=2, column=3, padx=4, pady=4)

        ttk.Button(ctrl, text="按默认生成端口", command=self.fill_default_ports).grid(row=2, column=4, padx=6, pady=4)
        ttk.Button(ctrl, text="保存配置", command=self.save_config).grid(row=2, column=5, padx=6, pady=4)
        ttk.Button(ctrl, text="启动感知页", command=self.start).grid(row=2, column=6, padx=6, pady=4)
        ttk.Button(ctrl, text="停止感知页", command=self.stop).grid(row=2, column=7, padx=6, pady=4)
        ttk.Label(ctrl, textvariable=self.status_var).grid(row=2, column=8, padx=6, pady=4, sticky="w")
        ttk.Label(ctrl, textvariable=self.last_tt_var).grid(row=2, column=9, padx=6, pady=4, sticky="w")

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=2)

        endpoint_frame = ttk.LabelFrame(left, text="TCP Server端口映射")
        endpoint_frame.pack(fill=tk.BOTH, expand=True)
        cols = ("uuv", "host", "port", "conn", "er_frames", "targets", "last_recv", "state")
        self.endpoint_tree = ttk.Treeview(endpoint_frame, columns=cols, show="headings", height=10)
        for c, w in [
            ("uuv", 80),
            ("host", 130),
            ("port", 80),
            ("conn", 60),
            ("er_frames", 80),
            ("targets", 70),
            ("last_recv", 110),
            ("state", 90),
        ]:
            self.endpoint_tree.heading(c, text=c)
            self.endpoint_tree.column(c, width=w, anchor=tk.CENTER)
        self.endpoint_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.endpoint_tree.bind("<<TreeviewSelect>>", self.on_endpoint_select)

        edit = ttk.Frame(endpoint_frame)
        edit.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(edit, text="uuv").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Entry(edit, textvariable=self.endpoint_uuv_var, width=9).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(edit, text="host").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Entry(edit, textvariable=self.endpoint_host_var, width=13).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(edit, text="port").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Entry(edit, textvariable=self.endpoint_port_var, width=8).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(edit, text="更新选中端口", command=self.apply_selected_endpoint).pack(side=tk.LEFT, padx=4)
        ttk.Button(edit, text="应用默认Host/Base", command=self.fill_default_ports).pack(side=tk.LEFT, padx=4)

        target_frame = ttk.LabelFrame(right, text="聚类后的Target汇总")
        target_frame.pack(fill=tk.X, pady=(0, 6))
        t_cols = ("id", "type", "lon", "lat", "depth", "bearing", "speed", "conf", "sources")
        self.target_tree = ttk.Treeview(target_frame, columns=t_cols, show="headings", height=8)
        for c, w in [
            ("id", 50),
            ("type", 60),
            ("lon", 110),
            ("lat", 110),
            ("depth", 90),
            ("bearing", 80),
            ("speed", 80),
            ("conf", 70),
            ("sources", 160),
        ]:
            self.target_tree.heading(c, text=c)
            self.target_tree.column(c, width=w, anchor=tk.CENTER)
        self.target_tree.pack(fill=tk.X, padx=4, pady=4)

        log_frame = ttk.LabelFrame(right, text="日志")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=12, wrap="none")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _init_endpoint_rows(self) -> None:
        for i in range(6):
            uid = f"uuv_{i + 1}"
            item = self.endpoint_tree.insert(
                "", tk.END, values=(uid, self.bind_host_var.get(), 6001 + i, 0, 0, 0, "-", "待命")
            )
            self.port_rows[uid] = item
            self.uuv_stats[uid] = {"conn": 0, "er_frames": 0, "targets": 0}

    def _refresh_port_rows(self) -> None:
        self.port_rows.clear()
        for item in self.endpoint_tree.get_children():
            uid = str(self.endpoint_tree.item(item, "values")[0])
            self.port_rows[uid] = item
            self.uuv_stats.setdefault(uid, {"conn": 0, "er_frames": 0, "targets": 0})

    def _append_log(self, line: str) -> None:
        self._write_log_file("system", line)
        self._append_log_ui(line)

    def _append_log_ui(self, line: str) -> None:
        self.log_text.insert(tk.END, line + "\n")
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > self.max_log_lines:
            self.log_text.delete("1.0", f"{lines - self.max_log_lines + 1}.0")
        self.log_text.see(tk.END)

    def _log_path(self, category: str) -> Path:
        if category not in self.log_files:
            day = time.strftime("%Y-%m-%d")
            self.log_files[category] = self.log_dir / day / f"{category}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        return self.log_files[category]

    def _write_log_file(self, category: str, line: str) -> None:
        try:
            path = self._log_path(category)
            path.parent.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with path.open("a", encoding="utf-8") as f:
                f.write(f"{stamp} {line}\n")
        except Exception:
            pass

    def _log_event(self, category: str, line: str) -> None:
        self._write_log_file(category, line)
        self._append_log_ui(line)

    def fill_default_ports(self) -> None:
        host = self.bind_host_var.get().strip() or "127.0.0.1"
        try:
            base_port = int(self.base_port_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Base Port 必须为整数")
            return
        for i, item in enumerate(self.endpoint_tree.get_children()):
            vals = list(self.endpoint_tree.item(item, "values"))
            vals[1] = host
            vals[2] = base_port + i
            self.endpoint_tree.item(item, values=tuple(vals))
        self._refresh_port_rows()

    def on_endpoint_select(self, _event=None) -> None:
        sel = self.endpoint_tree.selection()
        if not sel:
            return
        vals = self.endpoint_tree.item(sel[0], "values")
        self.endpoint_uuv_var.set(str(vals[0]))
        self.endpoint_host_var.set(str(vals[1]))
        self.endpoint_port_var.set(str(vals[2]))

    def apply_selected_endpoint(self) -> None:
        sel = self.endpoint_tree.selection()
        if not sel:
            messagebox.showwarning("No Selection", "请先选择一个端口映射行")
            return
        try:
            uid = self.endpoint_uuv_var.get().strip() or "uuv_1"
            host = self.endpoint_host_var.get().strip() or "127.0.0.1"
            port = int(self.endpoint_port_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "port 必须为整数")
            return
        vals = list(self.endpoint_tree.item(sel[0], "values"))
        vals[0] = uid
        vals[1] = host
        vals[2] = port
        self.endpoint_tree.item(sel[0], values=tuple(vals))
        self._refresh_port_rows()

    def _endpoints_from_ui(self) -> List[Tuple[str, str, int]]:
        endpoints = []
        for item in self.endpoint_tree.get_children():
            uid, host, port, *_ = self.endpoint_tree.item(item, "values")
            endpoints.append((str(uid), str(host), int(port)))
        return endpoints

    def _load_config(self, silent: bool = False) -> None:
        if not DEFAULT_PERCEPTION_CFG.exists():
            return
        try:
            cfg = load_json(DEFAULT_PERCEPTION_CFG)
            self.udp_host_var.set(str(cfg.get("udp_host", self.udp_host_var.get())))
            self.udp_port_var.set(str(cfg.get("udp_port", self.udp_port_var.get())))
            self.eps_var.set(str(cfg.get("eps_m", self.eps_var.get())))
            self.min_samples_var.set(str(cfg.get("min_samples", self.min_samples_var.get())))
            self.window_var.set(str(cfg.get("window_s", self.window_var.get())))
            self.udp_interval_var.set(str(cfg.get("udp_interval_s", self.udp_interval_var.get())))
            self.re_lon_var.set(str(cfg.get("re_lon", self.re_lon_var.get())))
            self.re_lat_var.set(str(cfg.get("re_lat", self.re_lat_var.get())))
            for ep, item in zip(cfg.get("endpoints", []), self.endpoint_tree.get_children()):
                vals = list(self.endpoint_tree.item(item, "values"))
                vals[0] = str(ep.get("uuv_id", vals[0]))
                vals[1] = str(ep.get("host", vals[1]))
                vals[2] = int(ep.get("port", vals[2]))
                self.endpoint_tree.item(item, values=tuple(vals))
            self._refresh_port_rows()
        except Exception as exc:
            if not silent:
                messagebox.showerror("Config Error", str(exc))

    def save_config(self) -> None:
        try:
            data = {
                "schema_version": "1.0",
                "udp_host": self.udp_host_var.get().strip(),
                "udp_port": int(self.udp_port_var.get().strip()),
                "eps_m": float(self.eps_var.get().strip()),
                "min_samples": int(self.min_samples_var.get().strip()),
                "window_s": float(self.window_var.get().strip()),
                "udp_interval_s": float(self.udp_interval_var.get().strip()),
                "re_lon": float(self.re_lon_var.get().strip()),
                "re_lat": float(self.re_lat_var.get().strip()),
                "endpoints": [
                    {"uuv_id": uid, "host": host, "port": port}
                    for uid, host, port in self._endpoints_from_ui()
                ],
            }
            save_json(DEFAULT_PERCEPTION_CFG, data)
            self._append_log(f"[config] saved {DEFAULT_PERCEPTION_CFG}")
        except Exception as exc:
            messagebox.showerror("Config Error", str(exc))

    def start(self) -> None:
        if self.running:
            return
        try:
            self._refresh_port_rows()
            endpoints = self._endpoints_from_ui()
            delimiter = _decode_delimiter(self.delimiter_var.get().strip() or "&&")
            udp_host = self.udp_host_var.get().strip() or "127.0.0.1"
            udp_port = int(self.udp_port_var.get().strip())
            eps_m = float(self.eps_var.get().strip())
            min_samples = int(self.min_samples_var.get().strip())
            window_s = float(self.window_var.get().strip())
            udp_interval_s = float(self.udp_interval_var.get().strip())
            re_lon = float(self.re_lon_var.get().strip())
            re_lat = float(self.re_lat_var.get().strip())
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        self.hub.start(
            endpoints=endpoints,
            delimiter=delimiter,
            udp_host=udp_host,
            udp_port=udp_port,
            eps_m=eps_m,
            min_samples=min_samples,
            window_s=window_s,
            udp_interval_s=udp_interval_s,
            re_lon=re_lon,
            re_lat=re_lat,
        )
        self.running = True
        self.status_var.set("运行中")
        self._append_log(f"[start] tcp={[(u, h, p) for u, h, p in endpoints]} udp={udp_host}:{udp_port}")

    def stop(self) -> None:
        self.hub.stop()
        self.running = False
        self.status_var.set("已停止")
        self._append_log("[stop] perception page stopped")

    def _set_row(self, uid: str, state: Optional[str] = None, last_recv: Optional[str] = None) -> None:
        item = self.port_rows.get(uid)
        if item is None:
            return
        stats = self.uuv_stats.setdefault(uid, {"conn": 0, "er_frames": 0, "targets": 0})
        vals = list(self.endpoint_tree.item(item, "values"))
        vals[3] = stats["conn"]
        vals[4] = stats["er_frames"]
        vals[5] = stats["targets"]
        if last_recv is not None:
            vals[6] = last_recv
        if state is not None:
            vals[7] = state
        self.endpoint_tree.item(item, values=tuple(vals))

    def _update_targets_from_message(self, message: str) -> None:
        self.target_tree.delete(*self.target_tree.get_children())
        tokens = [t.strip() for t in message.split(",")]
        if len(tokens) < 4 or tokens[0] != "$TT":
            return
        count = _safe_int(tokens[2]) or 0
        idx = 3
        for _ in range(count):
            if idx + 11 > len(tokens):
                break
            rec = tokens[idx : idx + 11]
            self.target_tree.insert(
                "",
                tk.END,
                values=(
                    rec[0],
                    rec[1],
                    rec[2],
                    rec[3],
                    f"{float(rec[4]):.2f}",
                    f"{float(rec[5]):.1f}",
                    f"{float(rec[7]):.2f}",
                    f"{float(rec[8]):.2f}",
                    "-",
                ),
            )
            idx += 11

    def _poll_events(self) -> None:
        while True:
            try:
                ev = self.event_q.get_nowait()
            except queue.Empty:
                break
            et = ev.get("type")
            uid = str(ev.get("uuv_id", ""))
            if et == "listen":
                self._set_row(uid, state="监听中")
                self._log_event("connection", f"[listen] {uid} {ev.get('host')}:{ev.get('port')}")
            elif et == "connect":
                stats = self.uuv_stats.setdefault(uid, {"conn": 0, "er_frames": 0, "targets": 0})
                stats["conn"] += 1
                self._set_row(uid, state="已连接")
                self._log_event("connection", f"[connect] {uid} port={ev.get('port')} peer={ev.get('peer')}")
            elif et == "disconnect":
                stats = self.uuv_stats.setdefault(uid, {"conn": 0, "er_frames": 0, "targets": 0})
                stats["conn"] = max(0, stats["conn"] - 1)
                self._set_row(uid, state="监听中" if self.running else "待命")
                self._log_event("connection", f"[disconnect] {uid} peer={ev.get('peer')}")
            elif et == "er":
                stats = self.uuv_stats.setdefault(uid, {"conn": 0, "er_frames": 0, "targets": 0})
                stats["er_frames"] += 1
                stats["targets"] += int(ev.get("count", 0))
                last_recv = time.strftime("%H:%M:%S", time.localtime(float(ev.get("recv_ts", time.time()))))
                self._set_row(uid, state="接收中", last_recv=last_recv)
                raw = str(ev.get("raw", "")).replace("\r", "").replace("\n", "")
                self._log_event("ER", f"[ER] {uid} targets={ev.get('count')} raw={raw}")
            elif et == "re":
                msg = str(ev.get("message", ""))
                self._log_event("RE", f"[RE] {uid} {msg}")
            elif et == "tt":
                count = int(ev.get("count", 0))
                msg = str(ev.get("message", ""))
                self.last_tt_var.set(f"TT目标数: {count}")
                self._update_targets_from_message(msg)
                self._log_event("TT", f"[TT->UDP] count={count} msg={msg}")
            elif et == "raw":
                raw = str(ev.get("raw", "")).replace("\r", "").replace("\n", "")
                self._log_event("raw", f"[RAW] {uid} {raw}")
            elif et == "error":
                self._log_event("error", f"[ERROR] {uid} {ev.get('error')}")
            elif et == "stopped":
                self._set_row(uid, state="已停止")
        self.after(100, self._poll_events)
