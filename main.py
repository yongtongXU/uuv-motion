import argparse
import json
import multiprocessing as mp
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Tuple

from motion_server import simulate_multi_process
from network import (
    DEFAULT_ENDPOINTS,
    load_endpoint_config,
    load_json,
    save_endpoint_config,
    send_path_points,
    test_tcp_connection,
)
from planning import INIT_PATH, PATH_PATH, plan_paths


ROOT = Path(__file__).resolve().parent
DEFAULT_INIT = INIT_PATH
DEFAULT_PATH = PATH_PATH
DEFAULT_TRAJ_DIR = ROOT / "trajectories"
DEFAULT_PLANNED_IMG = ROOT / "planned_paths.png"
DEFAULT_NET_FORMAT = ROOT / "config" / "network_format.json"
DEFAULT_ENDPOINT_CFG = DEFAULT_ENDPOINTS


def count_csv_points(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def run_plan(path_json: Path) -> Dict:
    result = plan_paths()
    with path_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def run_cli(args: argparse.Namespace) -> None:
    if not args.skip_plan:
        result = run_plan(path_json=args.path_json)
        print(
            f"[plan] uuv_count={result['summary']['uuv_count']} "
            f"total_waypoints={result['summary']['total_waypoints']} -> {args.path_json}"
        )

    if not args.skip_sim:
        simulate_multi_process(
            init_path=args.init,
            path_path=args.path_json,
            out_dir=args.traj_dir,
            flush_rows=max(1, args.flush_rows),
            max_rows_per_uuv=max(0, args.max_rows_per_uuv),
        )


class MissionGUI:
    def __init__(self, root: tk.Tk, init_path: Path, path_json: Path, traj_dir: Path):
        self.root = root
        self.init_path = init_path
        self.path_json = path_json
        self.traj_dir = traj_dir
        self.init_cfg: Dict = {}

        self.sending = False
        self.send_paused = False
        self.motion_running = False

        self.progress_queue: Optional[mp.Queue] = None
        self.send_stop_event: Optional[mp.Event] = None
        self.send_pause_event: Optional[mp.Event] = None
        self.send_thread: Optional[threading.Thread] = None
        self.motion_thread: Optional[threading.Thread] = None

        self.send_item_map: Dict[str, str] = {}
        self.endpoint_item_map: Dict[str, str] = {}

        self.area_len_var = tk.StringVar(value="10000")
        self.area_wid_var = tk.StringVar(value="10000")
        self.sample_hz_var = tk.StringVar(value="50")

        self.host_var = tk.StringVar(value="127.0.0.1")
        self.base_port_var = tk.StringVar(value="5000")
        self.max_frames_var = tk.StringVar(value="0")
        self.connect_timeout_var = tk.StringVar(value="2.0")
        self.endpoint_uuv_var = tk.StringVar()
        self.endpoint_ip_var = tk.StringVar(value="127.0.0.1")
        self.endpoint_port_var = tk.StringVar(value="5000")

        self.motion_flush_var = tk.StringVar(value="200000")
        self.motion_max_rows_var = tk.StringVar(value="0")

        self.uuv_edit_vars = {
            "id": tk.StringVar(),
            "x": tk.StringVar(),
            "y": tk.StringVar(),
            "z": tk.StringVar(),
            "roll": tk.StringVar(),
            "pitch": tk.StringVar(),
            "yaw": tk.StringVar(),
        }

        self.summary_var = tk.StringVar(value="发送状态统计: 待命=0, 正在发送=0, 暂停发送=0, 发送结束=0")

        self.root.title("UUV Mission GUI")
        self.root.geometry("1620x920")
        self._build_ui()
        self.load_init_to_ui()
        self.load_send_rows_from_path()
        self.draw_path_preview()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)

        ttk.Button(top, text="读取Init", command=self.load_init_to_ui).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="写入Init", command=self.save_ui_to_init).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="一键规划", command=self.on_plan).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="一键运行Motion", command=self.on_run_motion).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="网络连接", command=self.on_network_connect).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="一键发送", command=self.on_start_send).pack(side=tk.LEFT, padx=3)
        self.pause_btn = ttk.Button(top, text="暂停发送", command=self.on_pause_resume_send)
        self.pause_btn.pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="停止发送", command=self.on_stop_send).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="预览路径", command=self.draw_path_preview).pack(side=tk.LEFT, padx=3)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=1)

        cfg_frame = ttk.LabelFrame(left, text="场景设置(init.json)")
        cfg_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(cfg_frame, text="Area Length").grid(row=0, column=0, padx=5, pady=4, sticky="w")
        ttk.Entry(cfg_frame, width=12, textvariable=self.area_len_var).grid(row=0, column=1, padx=5, pady=4)
        ttk.Label(cfg_frame, text="Area Width").grid(row=0, column=2, padx=5, pady=4, sticky="w")
        ttk.Entry(cfg_frame, width=12, textvariable=self.area_wid_var).grid(row=0, column=3, padx=5, pady=4)
        ttk.Label(cfg_frame, text="Sample Hz").grid(row=0, column=4, padx=5, pady=4, sticky="w")
        ttk.Entry(cfg_frame, width=10, textvariable=self.sample_hz_var).grid(row=0, column=5, padx=5, pady=4)

        uuv_frame = ttk.LabelFrame(left, text="UUV初始状态(可编辑)")
        uuv_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("id", "x", "y", "z", "roll", "pitch", "yaw")
        self.uuv_tree = ttk.Treeview(uuv_frame, columns=cols, show="headings", height=14)
        for c, w in [("id", 80), ("x", 90), ("y", 90), ("z", 90), ("roll", 80), ("pitch", 80), ("yaw", 80)]:
            self.uuv_tree.heading(c, text=c)
            self.uuv_tree.column(c, width=w, anchor=tk.CENTER)
        self.uuv_tree.pack(fill=tk.X, padx=4, pady=4)
        self.uuv_tree.bind("<<TreeviewSelect>>", self.on_uuv_select)

        edit = ttk.Frame(uuv_frame)
        edit.pack(fill=tk.X, padx=4, pady=4)
        for i, k in enumerate(["id", "x", "y", "z", "roll", "pitch", "yaw"]):
            ttk.Label(edit, text=k).grid(row=0, column=i, padx=3, sticky="w")
            ttk.Entry(edit, width=9 if k != "id" else 10, textvariable=self.uuv_edit_vars[k]).grid(
                row=1, column=i, padx=3, pady=2
            )
        ttk.Button(edit, text="更新选中UUV", command=self.apply_selected_uuv_edit).grid(row=1, column=7, padx=8)

        preview_frame = ttk.LabelFrame(left, text="规划路径预览")
        preview_frame.pack(fill=tk.X, expand=False, pady=(6, 0))
        preview_frame.configure(height=430)
        preview_frame.pack_propagate(False)
        self.preview_canvas = tk.Canvas(preview_frame, bg="#111", width=780, height=420, highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        send_cfg = ttk.LabelFrame(right, text="网络发送默认设置")
        send_cfg.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(send_cfg, text="Host").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(send_cfg, width=14, textvariable=self.host_var).grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(send_cfg, text="Base Port").grid(row=0, column=2, padx=4, pady=4, sticky="w")
        ttk.Entry(send_cfg, width=8, textvariable=self.base_port_var).grid(row=0, column=3, padx=4, pady=4)
        ttk.Label(send_cfg, text="Max Frames").grid(row=0, column=4, padx=4, pady=4, sticky="w")
        ttk.Entry(send_cfg, width=8, textvariable=self.max_frames_var).grid(row=0, column=5, padx=4, pady=4)
        ttk.Label(send_cfg, text="Connect Timeout").grid(row=0, column=6, padx=4, pady=4, sticky="w")
        ttk.Entry(send_cfg, width=7, textvariable=self.connect_timeout_var).grid(row=0, column=7, padx=4, pady=4)

        ttk.Label(send_cfg, text="Motion Flush").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(send_cfg, width=14, textvariable=self.motion_flush_var).grid(row=1, column=1, padx=4, pady=4)
        ttk.Label(send_cfg, text="Motion Max Rows").grid(row=1, column=2, padx=4, pady=4, sticky="w")
        ttk.Entry(send_cfg, width=8, textvariable=self.motion_max_rows_var).grid(row=1, column=3, padx=4, pady=4)
        ttk.Button(send_cfg, text="按默认生成端点", command=self.fill_default_endpoints).grid(
            row=1, column=4, padx=6, pady=4
        )
        ttk.Button(send_cfg, text="读取端点JSON", command=self.load_endpoint_config_json).grid(
            row=1, column=5, padx=6, pady=4
        )
        ttk.Button(send_cfg, text="保存端点JSON", command=self.save_endpoint_config_json).grid(
            row=1, column=6, padx=6, pady=4
        )

        endpoint_frame = ttk.LabelFrame(right, text="每个Socket目标地址(可编辑)")
        endpoint_frame.pack(fill=tk.X, pady=(0, 6))
        e_cols = ("uuv", "ip", "port")
        self.endpoint_tree = ttk.Treeview(endpoint_frame, columns=e_cols, show="headings", height=7)
        for c, w in [("uuv", 90), ("ip", 170), ("port", 90)]:
            self.endpoint_tree.heading(c, text=c)
            self.endpoint_tree.column(c, width=w, anchor=tk.CENTER)
        self.endpoint_tree.pack(fill=tk.X, padx=4, pady=4)
        self.endpoint_tree.bind("<<TreeviewSelect>>", self.on_endpoint_select)

        endpoint_edit = ttk.Frame(endpoint_frame)
        endpoint_edit.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(endpoint_edit, text="uuv").grid(row=0, column=0, padx=3, sticky="w")
        ttk.Entry(endpoint_edit, width=10, textvariable=self.endpoint_uuv_var, state="readonly").grid(
            row=1, column=0, padx=3
        )
        ttk.Label(endpoint_edit, text="ip").grid(row=0, column=1, padx=3, sticky="w")
        ttk.Entry(endpoint_edit, width=18, textvariable=self.endpoint_ip_var).grid(row=1, column=1, padx=3)
        ttk.Label(endpoint_edit, text="port").grid(row=0, column=2, padx=3, sticky="w")
        ttk.Entry(endpoint_edit, width=10, textvariable=self.endpoint_port_var).grid(row=1, column=2, padx=3)
        ttk.Button(endpoint_edit, text="更新端点", command=self.apply_selected_endpoint_edit).grid(
            row=1, column=3, padx=6
        )

        status_frame = ttk.LabelFrame(right, text="发送状态与UUV状态")
        status_frame.pack(fill=tk.X, pady=(0, 6))
        s_cols = ("uuv", "status", "sent", "total", "x", "y", "z", "yaw", "t", "port")
        self.send_tree = ttk.Treeview(status_frame, columns=s_cols, show="headings", height=10)
        for c, w in [
            ("uuv", 80),
            ("status", 90),
            ("sent", 70),
            ("total", 70),
            ("x", 85),
            ("y", 85),
            ("z", 85),
            ("yaw", 95),
            ("t", 80),
            ("port", 70),
        ]:
            self.send_tree.heading(c, text=c)
            self.send_tree.column(c, width=w, anchor=tk.CENTER)
        self.send_tree.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(status_frame, textvariable=self.summary_var).pack(anchor="w", padx=4, pady=(0, 4))

        log_frame = ttk.LabelFrame(right, text="日志")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.log_text = tk.Text(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def log(self, msg: str) -> None:
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def load_init_to_ui(self) -> None:
        self.init_cfg = load_json(self.init_path)
        area = self.init_cfg.get("area", {})
        scene = self.init_cfg.get("scene", {})
        self.area_len_var.set(str(area.get("length", 10000)))
        self.area_wid_var.set(str(area.get("width", 10000)))
        self.sample_hz_var.set(str(scene.get("sample_rate_hz", 50)))

        self.uuv_tree.delete(*self.uuv_tree.get_children())
        for u in self.init_cfg.get("uuvs", []):
            pose = u.get("pose", {})
            self.uuv_tree.insert(
                "",
                tk.END,
                values=(
                    u.get("id", "uuv"),
                    pose.get("x", 0.0),
                    pose.get("y", 0.0),
                    pose.get("z", 0.0),
                    pose.get("roll", 0.0),
                    pose.get("pitch", 0.0),
                    pose.get("yaw", 0.0),
                ),
            )
        self.log("[init] loaded")
        self.load_send_rows_from_path()

    def save_ui_to_init(self) -> None:
        if not self.init_cfg:
            self.init_cfg = load_json(self.init_path)

        area_len = float(self.area_len_var.get())
        area_wid = float(self.area_wid_var.get())
        sample_hz = float(self.sample_hz_var.get())

        self.init_cfg.setdefault("area", {})["length"] = area_len
        self.init_cfg.setdefault("area", {})["width"] = area_wid
        self.init_cfg.setdefault("scene", {})["sample_rate_hz"] = sample_hz
        if sample_hz > 0:
            self.init_cfg.setdefault("scene", {})["time_step"] = round(1.0 / sample_hz, 8)

        old_by_id = {str(u.get("id")): u for u in self.init_cfg.get("uuvs", [])}
        new_uuvs: List[Dict] = []
        for item in self.uuv_tree.get_children():
            uid, x, y, z, roll, pitch, yaw = self.uuv_tree.item(item, "values")
            base = old_by_id.get(str(uid), {})
            pose = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
            }
            base["id"] = str(uid)
            base["pose"] = pose
            base.setdefault("twist", {"u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0, "q": 0.0, "r": 0.0})
            base.setdefault("health", {"battery": 1.0, "status": "ready"})
            new_uuvs.append(base)
        self.init_cfg["uuvs"] = new_uuvs

        with self.init_path.open("w", encoding="utf-8") as f:
            json.dump(self.init_cfg, f, ensure_ascii=False, indent=2)
        self.log("[init] saved")

    def on_uuv_select(self, _event=None) -> None:
        sel = self.uuv_tree.selection()
        if not sel:
            return
        vals = self.uuv_tree.item(sel[0], "values")
        for i, k in enumerate(["id", "x", "y", "z", "roll", "pitch", "yaw"]):
            self.uuv_edit_vars[k].set(str(vals[i]))

    def apply_selected_uuv_edit(self) -> None:
        sel = self.uuv_tree.selection()
        if not sel:
            messagebox.showwarning("No Selection", "请先选择一个UUV行")
            return
        vals = (
            self.uuv_edit_vars["id"].get().strip(),
            self.uuv_edit_vars["x"].get().strip(),
            self.uuv_edit_vars["y"].get().strip(),
            self.uuv_edit_vars["z"].get().strip(),
            self.uuv_edit_vars["roll"].get().strip(),
            self.uuv_edit_vars["pitch"].get().strip(),
            self.uuv_edit_vars["yaw"].get().strip(),
        )
        self.uuv_tree.item(sel[0], values=vals)

    def on_plan(self) -> None:
        try:
            self.save_ui_to_init()
            result = run_plan(self.path_json)
            self.log(
                f"[plan] uuv={result['summary']['uuv_count']} "
                f"total_wp={result['summary']['total_waypoints']}"
            )
            self.load_send_rows_from_path()
            self.draw_path_preview()
        except Exception as e:
            messagebox.showerror("Plan Error", str(e))

    def on_run_motion(self) -> None:
        if self.motion_running:
            self.log("[motion] already running")
            return

        try:
            flush_rows = int(self.motion_flush_var.get().strip())
            max_rows = int(self.motion_max_rows_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Motion参数必须为整数")
            return

        self.motion_running = True
        self.log("[motion] started")

        def worker() -> None:
            err = None
            try:
                simulate_multi_process(
                    init_path=self.init_path,
                    path_path=self.path_json,
                    out_dir=self.traj_dir,
                    flush_rows=max(1, flush_rows),
                    max_rows_per_uuv=max(0, max_rows),
                )
            except Exception as e:
                err = str(e)

            def done_cb() -> None:
                self.motion_running = False
                if err:
                    self.log(f"[motion] error: {err}")
                    messagebox.showerror("Motion Error", err)
                else:
                    self.log("[motion] finished")

            self.root.after(0, done_cb)

        self.motion_thread = threading.Thread(target=worker, daemon=True)
        self.motion_thread.start()

    def on_network_connect(self) -> None:
        try:
            endpoint_map = self.get_endpoint_map_from_ui()
            timeout_s = float(self.connect_timeout_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "connect timeout 必须为数字")
            return

        self.log("[connect] start tcp check...")

        def worker() -> None:
            ok_count = 0
            total = len(endpoint_map)
            for uid, (ip, port) in endpoint_map.items():
                ok, msg = test_tcp_connection(ip, int(port), timeout_s=max(0.1, timeout_s))
                if ok:
                    ok_count += 1
                    self.root.after(0, lambda u=uid, i=ip, p=port: self.log(f"[connect][ok] {u} -> {i}:{p}"))
                else:
                    self.root.after(
                        0, lambda u=uid, i=ip, p=port, m=msg: self.log(f"[connect][fail] {u} -> {i}:{p} | {m}")
                    )
            self.root.after(0, lambda: self.log(f"[connect] done {ok_count}/{total} reachable"))

        threading.Thread(target=worker, daemon=True).start()

    def on_start_send(self) -> None:
        if self.sending:
            self.log("[send] already running")
            return
        if not self.path_json.exists():
            messagebox.showwarning("Path Missing", "请先规划路径")
            return

        try:
            host = self.host_var.get().strip() or "127.0.0.1"
            base_port = int(self.base_port_var.get().strip())
            max_frames = int(self.max_frames_var.get().strip())
            endpoint_map = self.get_endpoint_map_from_ui()
            connect_timeout_s = float(self.connect_timeout_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "网络参数格式错误（base_port/max_frames为整数，timeout为数字）")
            return

        save_endpoint_config(self._endpoint_cfg_path(), endpoint_map)
        self.progress_queue = mp.Queue()
        self.send_stop_event = mp.Event()
        self.send_pause_event = mp.Event()
        self.sending = True
        self.send_paused = False
        self.pause_btn.config(text="暂停发送")
        self.log("[send] started")

        missing = []
        for uid in endpoint_map.keys():
            csv_path = self.traj_dir / f"trajectory_{uid}.csv"
            if not csv_path.exists():
                missing.append(str(csv_path))
        if missing:
            self.sending = False
            messagebox.showerror("Trajectory Missing", "未找到轨迹CSV，请先点击“一键运行Motion”。")
            self.log(f"[send] missing trajectory files: {len(missing)}")
            return

        for uid, item in self.send_item_map.items():
            vals = list(self.send_tree.item(item, "values"))
            vals[1] = "待命"
            vals[2] = 0
            if uid in endpoint_map:
                vals[9] = endpoint_map[uid][1]
            self.send_tree.item(item, values=tuple(vals))

        def worker() -> None:
            try:
                send_path_points(
                    init_path=self.init_path,
                    path_path=self.path_json,
                    host=host,
                    base_port=base_port,
                    max_frames=max(0, max_frames),
                    endpoint_map=endpoint_map,
                    format_config_path=DEFAULT_NET_FORMAT,
                    connect_timeout_s=max(0.1, connect_timeout_s),
                    progress_queue=self.progress_queue,
                    stop_event=self.send_stop_event,
                    pause_event=self.send_pause_event,
                    source="trajectory",
                    traj_dir=self.traj_dir,
                )
            except Exception as e:
                if self.progress_queue is not None:
                    self.progress_queue.put({"type": "fatal", "message": str(e)})

        self.send_thread = threading.Thread(target=worker, daemon=True)
        self.send_thread.start()
        self.root.after(50, self.poll_progress)

    def on_pause_resume_send(self) -> None:
        if not self.sending or self.send_pause_event is None:
            return
        if self.send_paused:
            self.send_pause_event.clear()
            self.send_paused = False
            self.pause_btn.config(text="暂停发送")
            self.log("[send] resumed")
        else:
            self.send_pause_event.set()
            self.send_paused = True
            self.pause_btn.config(text="恢复发送")
            self.log("[send] paused")

    def on_stop_send(self) -> None:
        if not self.sending or self.send_stop_event is None:
            return
        self.send_stop_event.set()
        self.log("[send] stop requested")

    def load_send_rows_from_path(self) -> None:
        self.send_tree.delete(*self.send_tree.get_children())
        self.send_item_map.clear()
        self.endpoint_tree.delete(*self.endpoint_tree.get_children())
        self.endpoint_item_map.clear()
        if not self.path_json.exists():
            self.update_summary()
            return

        cfg = load_json(self.path_json)
        base_port = int(self.base_port_var.get() or "5000")
        default_host = self.host_var.get().strip() or "127.0.0.1"
        for i, p in enumerate(cfg.get("paths", [])):
            uid = str(p.get("id", f"uuv_{i+1}"))
            csv_path = self.traj_dir / f"trajectory_{uid}.csv"
            csv_total = count_csv_points(csv_path)
            total = csv_total if csv_total > 0 else len(p.get("waypoints", []))
            port = base_port + i
            item = self.send_tree.insert(
                "",
                tk.END,
                values=(uid, "待命", 0, total, "-", "-", "-", "-", "-", port),
            )
            self.send_item_map[uid] = item
            endpoint_item = self.endpoint_tree.insert("", tk.END, values=(uid, default_host, port))
            self.endpoint_item_map[uid] = endpoint_item
        self.apply_endpoint_map(self._load_endpoint_cfg_file(silent=True))
        self.update_summary()

    def _endpoint_cfg_path(self) -> Path:
        return DEFAULT_ENDPOINT_CFG

    def _load_endpoint_cfg_file(self, silent: bool = False) -> Dict[str, Tuple[str, int]]:
        path = self._endpoint_cfg_path()
        try:
            return load_endpoint_config(path)
        except Exception as e:
            if not silent:
                messagebox.showerror("Endpoint JSON Error", str(e))
            return {}

    def apply_endpoint_map(self, endpoint_map: Dict[str, Tuple[str, int]]) -> None:
        if not endpoint_map:
            return
        for uid, item in self.endpoint_item_map.items():
            if uid in endpoint_map:
                ip, port = endpoint_map[uid]
                self.endpoint_tree.item(item, values=(uid, ip, int(port)))

    def load_endpoint_config_json(self) -> None:
        endpoint_map = self._load_endpoint_cfg_file(silent=False)
        self.apply_endpoint_map(endpoint_map)
        self.log(f"[endpoint] loaded from {self._endpoint_cfg_path()}")

    def save_endpoint_config_json(self) -> None:
        try:
            endpoint_map = self.get_endpoint_map_from_ui()
            save_endpoint_config(self._endpoint_cfg_path(), endpoint_map)
            self.log(f"[endpoint] saved to {self._endpoint_cfg_path()}")
        except Exception as e:
            messagebox.showerror("Endpoint JSON Error", str(e))

    def fill_default_endpoints(self) -> None:
        default_host = self.host_var.get().strip() or "127.0.0.1"
        try:
            base_port = int(self.base_port_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Base Port 必须为整数")
            return

        for i, item in enumerate(self.endpoint_tree.get_children()):
            vals = list(self.endpoint_tree.item(item, "values"))
            vals[1] = default_host
            vals[2] = base_port + i
            self.endpoint_tree.item(item, values=tuple(vals))
        self.log("[send] endpoint table reset to default host/base_port")

    def on_endpoint_select(self, _event=None) -> None:
        sel = self.endpoint_tree.selection()
        if not sel:
            return
        vals = self.endpoint_tree.item(sel[0], "values")
        self.endpoint_uuv_var.set(str(vals[0]))
        self.endpoint_ip_var.set(str(vals[1]))
        self.endpoint_port_var.set(str(vals[2]))

    def apply_selected_endpoint_edit(self) -> None:
        sel = self.endpoint_tree.selection()
        if not sel:
            messagebox.showwarning("No Selection", "请先选择一个端点行")
            return
        ip = self.endpoint_ip_var.get().strip() or "127.0.0.1"
        try:
            port = int(self.endpoint_port_var.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "port 必须为整数")
            return
        uid = self.endpoint_uuv_var.get().strip()
        self.endpoint_tree.item(sel[0], values=(uid, ip, port))

    def get_endpoint_map_from_ui(self) -> Dict[str, Tuple[str, int]]:
        endpoint_map: Dict[str, Tuple[str, int]] = {}
        for item in self.endpoint_tree.get_children():
            uid, ip, port = self.endpoint_tree.item(item, "values")
            endpoint_map[str(uid)] = (str(ip), int(port))
        return endpoint_map

    def update_summary(self) -> None:
        counts = {"待命": 0, "正在发送": 0, "暂停发送": 0, "发送结束": 0}
        for item in self.send_tree.get_children():
            st = str(self.send_tree.item(item, "values")[1])
            if st in counts:
                counts[st] += 1
        self.summary_var.set(
            f"发送状态统计: 待命={counts['待命']}, 正在发送={counts['正在发送']}, 暂停发送={counts['暂停发送']}, 发送结束={counts['发送结束']}"
        )

    def poll_progress(self) -> None:
        if self.progress_queue is None:
            return
        try:
            while True:
                evt = self.progress_queue.get_nowait()
                self.handle_event(evt)
        except queue.Empty:
            pass

        if self.sending:
            self.root.after(50, self.poll_progress)

    def handle_event(self, evt: Dict) -> None:
        et = evt.get("type")
        if et == "all_start":
            self.log(
                f"[send] workers={evt.get('uuv_count')} host={evt.get('host')} "
                f"base_port={evt.get('base_port')} hz={evt.get('hz')}"
            )
            return

        if et in {"uuv_start", "uuv_progress", "uuv_paused", "uuv_resumed", "uuv_done", "uuv_error"}:
            uid = str(evt.get("uuv_id", ""))
            item = self.send_item_map.get(uid)
            if item is None:
                return
            vals = list(self.send_tree.item(item, "values"))

            if et == "uuv_start":
                vals[1] = "待命"
                vals[3] = evt.get("total", vals[3])
                vals[9] = evt.get("port", vals[9])
            elif et == "uuv_progress":
                vals[1] = "正在发送"
                vals[2] = evt.get("sent", vals[2])
                vals[3] = evt.get("total", vals[3])
                vals[4] = f"{float(evt.get('x', 0.0)):.2f}"
                vals[5] = f"{float(evt.get('y', 0.0)):.2f}"
                vals[6] = f"{float(evt.get('z', 0.0)):.2f}"
                vals[7] = f"{float(evt.get('yaw', 0.0)):.4f}"
                vals[8] = f"{float(evt.get('t', 0.0)):.2f}"
            elif et == "uuv_paused":
                vals[1] = "暂停发送"
                vals[2] = evt.get("sent", vals[2])
                vals[3] = evt.get("total", vals[3])
            elif et == "uuv_resumed":
                vals[1] = "正在发送"
                vals[2] = evt.get("sent", vals[2])
                vals[3] = evt.get("total", vals[3])
            elif et == "uuv_done":
                vals[1] = "发送结束"
                vals[2] = evt.get("sent", vals[2])
                vals[3] = evt.get("total", vals[3])
            elif et == "uuv_error":
                vals[1] = "异常"

            self.send_tree.item(item, values=tuple(vals))
            self.update_summary()
            return

        if et == "all_done":
            self.sending = False
            self.send_paused = False
            self.pause_btn.config(text="暂停发送")
            self.log(f"[send] completed, failed={evt.get('failed_processes')}")
            self.update_summary()
            return

        if et == "fatal":
            self.sending = False
            self.send_paused = False
            self.pause_btn.config(text="暂停发送")
            msg = str(evt.get("message"))
            self.log(f"[send] fatal: {msg}")
            messagebox.showerror("Send Error", msg)

    def draw_path_preview(self) -> None:
        self.preview_canvas.delete("all")
        if not self.path_json.exists():
            return

        cfg = load_json(self.path_json)
        paths = cfg.get("paths", [])
        area = cfg.get("area", {})
        L = float(area.get("length", 1.0))
        W = float(area.get("width", 1.0))
        if L <= 0 or W <= 0:
            return

        cw = max(120, int(self.preview_canvas.winfo_width()))
        ch = max(120, int(self.preview_canvas.winfo_height()))
        pad = 24
        scale = min((cw - 2 * pad) / L, (ch - 2 * pad) / W)
        draw_w = L * scale
        draw_h = W * scale
        x0 = (cw - draw_w) / 2
        y0 = (ch - draw_h) / 2
        x1 = x0 + draw_w
        y1 = y0 + draw_h

        self.preview_canvas.create_rectangle(x0, y0, x1, y1, outline="#666")
        colors = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350", "#AB47BC", "#26C6DA", "#FDD835", "#26A69A"]

        def to_px(x: float, y: float) -> Tuple[float, float]:
            return x0 + x * scale, y1 - y * scale

        for i, p in enumerate(paths):
            uid = str(p.get("id", f"uuv_{i+1}"))
            wps = p.get("waypoints", [])
            if len(wps) < 1:
                continue
            c = colors[i % len(colors)]
            pts: List[float] = []
            for w in wps:
                px, py = to_px(float(w["x"]), float(w["y"]))
                pts.extend([px, py])
            if len(pts) >= 4:
                self.preview_canvas.create_line(*pts, fill=c, width=2)
            sx, sy = to_px(float(wps[0]["x"]), float(wps[0]["y"]))
            ex, ey = to_px(float(wps[-1]["x"]), float(wps[-1]["y"]))
            self.preview_canvas.create_oval(sx - 3, sy - 3, sx + 3, sy + 3, fill=c, outline="")
            self.preview_canvas.create_line(ex - 4, ey - 4, ex + 4, ey + 4, fill=c, width=2)
            self.preview_canvas.create_line(ex - 4, ey + 4, ex + 4, ey - 4, fill=c, width=2)
            self.preview_canvas.create_text(10, 12 + 16 * i, text=uid, fill=c, anchor="w", font=("Consolas", 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UUV main entry (GUI by default).")
    parser.add_argument("--cli", action="store_true", help="run legacy CLI pipeline instead of GUI")
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT, help="init.json path")
    parser.add_argument("--path-json", type=Path, default=DEFAULT_PATH, help="path.json output path")
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR, help="trajectory csv output directory")

    parser.add_argument("--skip-plan", action="store_true", help="skip planning stage")
    parser.add_argument("--skip-sim", action="store_true", help="skip simulation stage")
    parser.add_argument("--flush-rows", type=int, default=200000, help="simulation csv flush interval")
    parser.add_argument("--max-rows-per-uuv", type=int, default=0, help="debug limit rows per uuv (0 means unlimited)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cli:
        run_cli(args)
        return

    root = tk.Tk()
    MissionGUI(root=root, init_path=args.init, path_json=args.path_json, traj_dir=args.traj_dir)
    root.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    main()
