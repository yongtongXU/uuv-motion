import argparse
import queue
import re
import socket
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Dict, List, Optional, Tuple


DEFAULT_UUV_COUNT = 6
DEFAULT_BASE_PORT = 5000
DEFAULT_DELIMITER = "&&"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone TCP receiver for UUV telemetry.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="bind host")
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT, help="base listen port")
    parser.add_argument("--uuv-count", type=int, default=DEFAULT_UUV_COUNT, help="uuv count")
    parser.add_argument("--delimiter", type=str, default=DEFAULT_DELIMITER, help="frame delimiter")
    parser.add_argument("--cli", action="store_true", help="run in CLI mode")
    return parser.parse_args()


def _safe_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def parse_frame(frame: str) -> Dict[str, Optional[float]]:
    payload = frame.strip()
    if payload.endswith("&&"):
        payload = payload[:-2]
    if payload.startswith("$"):
        payload = payload[1:]
    payload = payload.strip()
    tokens = [t.strip() for t in payload.split(",") if t.strip()]

    out: Dict[str, Optional[float]] = {
        "uuv_id": None,
        "x": None,
        "y": None,
        "z": None,
        "yaw": None,
        "sim_time": None,
    }
    if not tokens:
        return out

    # Priority 1: explicit uuv_id token.
    for i, t in enumerate(tokens):
        if re.fullmatch(r"uuv[_-]?\d+", t, flags=re.IGNORECASE):
            num = re.sub(r"\D", "", t)
            if num:
                out["uuv_id"] = f"uuv_{int(num)}"
            # Expected sequence: uuv_id,x,y,z,pitch,roll,yaw,...,time
            if i + 3 < len(tokens):
                out["x"] = _safe_float(tokens[i + 1])
                out["y"] = _safe_float(tokens[i + 2])
                out["z"] = _safe_float(tokens[i + 3])
            if i + 6 < len(tokens):
                out["yaw"] = _safe_float(tokens[i + 6])
            if i + 1 < len(tokens):
                # Try from tail first: often time is near tail.
                out["sim_time"] = _safe_float(tokens[-1].rstrip("*"))
            return out

    # Priority 2: node_desc token, example: U-3-2-J -> uuv_2
    for t in tokens:
        m = re.fullmatch(r"[Uu]-\d+-(\d+)-[A-Za-z]", t)
        if m:
            out["uuv_id"] = f"uuv_{int(m.group(1))}"
            break

    # Fallback for RS-like format:
    # $RS,Length,6,1,node_desc,x,y,z,phi,theta,psi, ... ,Time,*
    if len(tokens) >= 11:
        out["x"] = _safe_float(tokens[5])
        out["y"] = _safe_float(tokens[6])
        out["z"] = _safe_float(tokens[7])
        out["yaw"] = _safe_float(tokens[10])
        out["sim_time"] = _safe_float(tokens[-1].rstrip("*"))
    return out


class TcpReceiver:
    def __init__(self, event_q: queue.Queue, delimiter: str):
        self.event_q = event_q
        self.delimiter = delimiter
        self._server_threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._sockets: List[socket.socket] = []

    def start(self, host: str, ports: List[int]) -> None:
        self.stop()
        self._stop_event.clear()
        for port in ports:
            t = threading.Thread(target=self._serve_port, args=(host, port), daemon=True)
            t.start()
            self._server_threads.append(t)

    def stop(self) -> None:
        self._stop_event.set()
        for s in self._sockets:
            try:
                s.close()
            except Exception:
                pass
        self._sockets.clear()
        self._server_threads.clear()

    def _serve_port(self, host: str, port: int) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((host, port))
            srv.listen(16)
            srv.settimeout(0.5)
            self._sockets.append(srv)
            self.event_q.put({"type": "port_started", "port": port})
        except Exception as e:
            self.event_q.put({"type": "error", "port": port, "error": f"bind/listen failed: {e}"})
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
            self.event_q.put({"type": "connected", "port": port, "peer": f"{addr[0]}:{addr[1]}"})
            t = threading.Thread(target=self._handle_client, args=(conn, addr, port), daemon=True)
            t.start()

        try:
            srv.close()
        except Exception:
            pass
        self.event_q.put({"type": "port_stopped", "port": port})

    def _handle_client(self, conn: socket.socket, addr, port: int) -> None:
        peer = f"{addr[0]}:{addr[1]}"
        buf = ""
        delim = self.delimiter
        try:
            while not self._stop_event.is_set():
                data = conn.recv(4096)
                if not data:
                    break
                text = data.decode("utf-8", errors="replace")
                buf += text
                while True:
                    idx = buf.find(delim)
                    if idx < 0:
                        break
                    frame = buf[: idx + len(delim)]
                    buf = buf[idx + len(delim) :]
                    parsed = parse_frame(frame)
                    self.event_q.put(
                        {
                            "type": "frame",
                            "port": port,
                            "peer": peer,
                            "raw": frame,
                            "parsed": parsed,
                            "recv_ts": time.time(),
                        }
                    )
        except Exception as e:
            self.event_q.put({"type": "error", "port": port, "peer": peer, "error": str(e)})
        finally:
            try:
                conn.close()
            except Exception:
                pass
            self.event_q.put({"type": "disconnected", "port": port, "peer": peer})


class ServerGui:
    def __init__(self, root: tk.Tk, host: str, base_port: int, uuv_count: int, delimiter: str):
        self.root = root
        self.root.title("UUV TCP Server Monitor")
        self.root.geometry("1280x760")

        self.host_var = tk.StringVar(value=host)
        self.base_port_var = tk.StringVar(value=str(base_port))
        self.delimiter_var = tk.StringVar(value=delimiter)
        self.status_var = tk.StringVar(value="待命")
        self.conn_count_var = tk.StringVar(value="0")

        self.event_q: queue.Queue = queue.Queue()
        self.receiver = TcpReceiver(event_q=self.event_q, delimiter=delimiter)
        self.running = False
        self.max_log_lines = 1200

        self.uuv_count = max(1, uuv_count)
        self.uuv_ids = [f"uuv_{i+1}" for i in range(self.uuv_count)]
        self.uuv_ports = {uid: base_port + i for i, uid in enumerate(self.uuv_ids)}
        self.row_by_uid: Dict[str, str] = {}
        self.conn_per_port: Dict[int, int] = {}

        self._build_ui()
        self._init_rows()
        self._poll_events()

    def _build_ui(self) -> None:
        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.pack(fill="x")

        ttk.Label(ctrl, text="Host").pack(side="left")
        ttk.Entry(ctrl, textvariable=self.host_var, width=16).pack(side="left", padx=(4, 10))
        ttk.Label(ctrl, text="Base Port").pack(side="left")
        ttk.Entry(ctrl, textvariable=self.base_port_var, width=8).pack(side="left", padx=(4, 10))
        ttk.Label(ctrl, text="Delimiter").pack(side="left")
        ttk.Entry(ctrl, textvariable=self.delimiter_var, width=8).pack(side="left", padx=(4, 10))

        ttk.Button(ctrl, text="Start", command=self.start_server).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Stop", command=self.stop_server).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Clear Log", command=self.clear_log).pack(side="left", padx=4)

        ttk.Label(ctrl, text="Server").pack(side="left", padx=(16, 4))
        ttk.Label(ctrl, textvariable=self.status_var).pack(side="left")
        ttk.Label(ctrl, text="Connections").pack(side="left", padx=(16, 4))
        ttk.Label(ctrl, textvariable=self.conn_count_var).pack(side="left")

        body = ttk.Panedwindow(self.root, orient="vertical")
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        top = ttk.Frame(body)
        bottom = ttk.Frame(body)
        body.add(top, weight=2)
        body.add(bottom, weight=1)

        cols = ("uuv_id", "listen_port", "conn", "frames", "x", "y", "z", "yaw", "sim_time", "last_recv", "state")
        self.table = ttk.Treeview(top, columns=cols, show="headings", height=16)
        for c in cols:
            self.table.heading(c, text=c)
            w = 95
            if c in {"uuv_id", "state"}:
                w = 110
            if c == "last_recv":
                w = 160
            self.table.column(c, width=w, anchor="center")
        yscroll = ttk.Scrollbar(top, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=yscroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        log_frame = ttk.LabelFrame(bottom, text="Receive Log", padding=6)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=10, wrap="none")
        log_y = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_y.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_y.pack(side="right", fill="y")

    def _init_rows(self) -> None:
        for uid in self.uuv_ids:
            row = self.table.insert(
                "",
                "end",
                values=(uid, self.uuv_ports[uid], 0, 0, "-", "-", "-", "-", "-", "-", "待命"),
            )
            self.row_by_uid[uid] = row

    def _fmt2(self, v: Optional[float]) -> str:
        if v is None:
            return "-"
        return f"{v:.2f}"

    def _update_conn_total(self) -> None:
        self.conn_count_var.set(str(sum(self.conn_per_port.values())))

    def _append_log(self, line: str) -> None:
        self.log_text.insert("end", line + "\n")
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > self.max_log_lines:
            remove_count = lines - self.max_log_lines
            self.log_text.delete("1.0", f"{remove_count + 1}.0")
        self.log_text.see("end")

    def clear_log(self) -> None:
        self.log_text.delete("1.0", "end")

    def _set_row_state(self, uuv_id: str, state: str, conn: Optional[int] = None) -> None:
        row = self.row_by_uid.get(uuv_id)
        if not row:
            return
        vals = list(self.table.item(row, "values"))
        if conn is not None:
            vals[2] = conn
        vals[10] = state
        self.table.item(row, values=vals)

    def start_server(self) -> None:
        if self.running:
            return
        try:
            base_port = int(self.base_port_var.get())
        except Exception:
            self._append_log("[ERROR] base port is invalid")
            return
        host = self.host_var.get().strip() or "0.0.0.0"
        delimiter = self.delimiter_var.get()
        if not delimiter:
            self._append_log("[ERROR] delimiter cannot be empty")
            return

        self.receiver.delimiter = delimiter
        self.uuv_ports = {uid: base_port + i for i, uid in enumerate(self.uuv_ids)}
        for uid, row in self.row_by_uid.items():
            vals = list(self.table.item(row, "values"))
            vals[1] = self.uuv_ports[uid]
            vals[2] = 0
            vals[3] = 0
            vals[10] = "待命"
            self.table.item(row, values=vals)

        self.conn_per_port = {p: 0 for p in self.uuv_ports.values()}
        self._update_conn_total()

        self.receiver.start(host=host, ports=list(self.uuv_ports.values()))
        self.running = True
        self.status_var.set("运行中")
        self._append_log(f"[INFO] server started: host={host}, ports={list(self.uuv_ports.values())}")

    def stop_server(self) -> None:
        if not self.running:
            return
        self.receiver.stop()
        self.running = False
        self.status_var.set("已停止")
        self._append_log("[INFO] server stopped")
        for uid in self.uuv_ids:
            self._set_row_state(uid, "待命", conn=0)
        self.conn_per_port.clear()
        self._update_conn_total()

    def _guess_uid(self, parsed_uid: Optional[str], port: int) -> Optional[str]:
        if parsed_uid and parsed_uid in self.row_by_uid:
            return parsed_uid
        for uid, p in self.uuv_ports.items():
            if p == port:
                return uid
        return None

    def _poll_events(self) -> None:
        while True:
            try:
                ev = self.event_q.get_nowait()
            except queue.Empty:
                break

            et = ev.get("type")
            if et == "connected":
                port = int(ev.get("port", -1))
                self.conn_per_port[port] = self.conn_per_port.get(port, 0) + 1
                self._update_conn_total()
                uid = self._guess_uid(None, port)
                if uid:
                    self._set_row_state(uid, "已连接", conn=self.conn_per_port[port])
                self._append_log(f"[CONNECT] port={port} peer={ev.get('peer')}")

            elif et == "disconnected":
                port = int(ev.get("port", -1))
                self.conn_per_port[port] = max(0, self.conn_per_port.get(port, 0) - 1)
                self._update_conn_total()
                uid = self._guess_uid(None, port)
                if uid:
                    state = "已连接" if self.conn_per_port[port] > 0 else "待命"
                    self._set_row_state(uid, state, conn=self.conn_per_port[port])
                self._append_log(f"[DISCONNECT] port={port} peer={ev.get('peer')}")

            elif et == "frame":
                port = int(ev.get("port", -1))
                parsed = ev.get("parsed", {}) or {}
                uid = self._guess_uid(parsed.get("uuv_id"), port)
                if uid:
                    row = self.row_by_uid[uid]
                    vals = list(self.table.item(row, "values"))
                    try:
                        vals[3] = int(vals[3]) + 1
                    except Exception:
                        vals[3] = 1
                    vals[4] = self._fmt2(parsed.get("x"))
                    vals[5] = self._fmt2(parsed.get("y"))
                    vals[6] = self._fmt2(parsed.get("z"))
                    vals[7] = self._fmt2(parsed.get("yaw"))
                    vals[8] = self._fmt2(parsed.get("sim_time"))
                    vals[9] = time.strftime("%H:%M:%S", time.localtime(ev.get("recv_ts", time.time())))
                    vals[10] = "接收中"
                    self.table.item(row, values=vals)
                raw = str(ev.get("raw", "")).replace("\r", "").replace("\n", "")
                self._append_log(f"[FRAME] port={port} uid={uid or '-'} msg={raw[:220]}")

            elif et == "error":
                self._append_log(
                    f"[ERROR] port={ev.get('port')} peer={ev.get('peer', '-')} error={ev.get('error')}"
                )

            elif et == "port_started":
                self._append_log(f"[INFO] listen started on port={ev.get('port')}")

            elif et == "port_stopped":
                self._append_log(f"[INFO] listen stopped on port={ev.get('port')}")

        self.root.after(50, self._poll_events)


def run_cli(host: str, base_port: int, uuv_count: int, delimiter: str) -> None:
    q: queue.Queue = queue.Queue()
    receiver = TcpReceiver(event_q=q, delimiter=delimiter)
    ports = [base_port + i for i in range(max(1, uuv_count))]
    receiver.start(host, ports)
    print(f"tcp server(cli) started: host={host}, ports={ports}, delimiter='{delimiter}'")
    try:
        while True:
            try:
                ev = q.get(timeout=0.5)
            except queue.Empty:
                continue
            print(ev)
    except KeyboardInterrupt:
        print("\nserver stopped")
    finally:
        receiver.stop()


def run_gui(host: str, base_port: int, uuv_count: int, delimiter: str) -> None:
    root = tk.Tk()
    app = ServerGui(root, host=host, base_port=base_port, uuv_count=uuv_count, delimiter=delimiter)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_server(), root.destroy()))
    root.mainloop()


def main() -> None:
    args = parse_args()
    if args.cli:
        run_cli(args.host, args.base_port, args.uuv_count, args.delimiter)
    else:
        run_gui(args.host, args.base_port, args.uuv_count, args.delimiter)


if __name__ == "__main__":
    main()
