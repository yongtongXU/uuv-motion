import argparse
import queue
import socket
import sys
import time
from typing import List, Tuple

from perception import PerceptionHub


ER_FRAME_NEAR_A = (
    "$ER,69,1,0,144,120.158020,22.663700,98.0,87.4,0,"
    "0.5938413672391344,0.8,23:41:44:19,0,23:41:46*&&"
)
ER_FRAME_NEAR_B = (
    "$ER,69,1,0,144,120.158030,22.663710,98.0,88.2,0,"
    "0.6500000000000000,0.82,23:41:45:19,0,23:41:47*&&"
)
ER_FRAME_FAR = (
    "$ER,69,1,0,144,120.165950,22.663520,86.0,-33.0,0,"
    "2.100844528140899,0.72418758,13:37:27:57,0,13:37:29*&&"
)


def read_until(sock: socket.socket, marker: str, timeout_s: float) -> str:
    deadline = time.time() + timeout_s
    chunks: List[str] = []
    while time.time() < deadline:
        try:
            data = sock.recv(4096)
        except socket.timeout:
            continue
        if not data:
            break
        chunks.append(data.decode("utf-8", errors="replace"))
        text = "".join(chunks)
        if marker in text:
            return text
    return "".join(chunks)


def wait_for_udp_tt(udp_sock: socket.socket, expected_count: int, timeout_s: float) -> str:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            data, _addr = udp_sock.recvfrom(65535)
        except socket.timeout:
            continue
        text = data.decode("utf-8", errors="replace")
        last = text
        tokens = [t.strip() for t in text.split(",")]
        if len(tokens) >= 3 and tokens[0] == "$TT" and tokens[2] == str(expected_count):
            return text
    raise AssertionError(f"did not receive $TT with count={expected_count}; last={last!r}")


def connect_client(host: str, port: int, timeout_s: float) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            sock.connect((host, port))
            return sock
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    sock.close()
    raise AssertionError(f"connect failed {host}:{port}: {last_error}")


def drain_events(event_q: queue.Queue) -> List[dict]:
    events = []
    while True:
        try:
            events.append(event_q.get_nowait())
        except queue.Empty:
            return events


def run_test(host: str, base_port: int, udp_port: int, verbose: bool) -> None:
    event_q: queue.Queue = queue.Queue()
    endpoints: List[Tuple[str, str, int]] = [(f"uuv_{i + 1}", host, base_port + i) for i in range(6)]
    hub = PerceptionHub(event_q)
    clients: List[socket.socket] = []
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        udp_sock.bind((host, udp_port))
        udp_sock.settimeout(0.25)
        hub.start(
            endpoints=endpoints,
            delimiter="&&",
            udp_host=host,
            udp_port=udp_port,
            eps_m=80.0,
            min_samples=1,
            window_s=3.0,
            udp_interval_s=0.2,
            re_lon=121.71685,
            re_lat=25.16955,
        )

        for _ in range(30):
            events = drain_events(event_q)
            if verbose:
                for ev in events:
                    print("[event]", ev)
            if sum(1 for ev in events if ev.get("type") == "listen") >= 6:
                break
            time.sleep(0.05)

        for _uid, ep_host, ep_port in endpoints:
            clients.append(connect_client(ep_host, ep_port, timeout_s=3.0))

        for i, sock in enumerate(clients, start=1):
            re_text = read_until(sock, "$RE,", timeout_s=2.0)
            assert "$RE," in re_text, f"uuv_{i} did not receive RE message"
            assert f",$RE," not in re_text, "unexpected malformed RE prefix"
            if verbose:
                print(f"[RE uuv_{i}]", re_text.strip().splitlines()[0])

        clients[0].sendall(ER_FRAME_NEAR_A.encode("utf-8"))
        clients[1].sendall(ER_FRAME_NEAR_B.encode("utf-8"))
        clients[2].sendall(ER_FRAME_FAR.encode("utf-8"))

        tt_text = wait_for_udp_tt(udp_sock, expected_count=2, timeout_s=3.0)
        if verbose:
            print("[TT]", tt_text)

        tokens = [t.strip() for t in tt_text.split(",")]
        assert tokens[0] == "$TT", "UDP message is not TT"
        assert tokens[2] == "2", "DBSCAN should merge two near observations and keep one far target"
        assert "120.158025" in tt_text, "near target longitude should be averaged"
        assert "120.165950" in tt_text, "far target should remain in TT output"

        print("perception test passed")
        print(f"tcp servers: {host}:{base_port}-{base_port + 5}")
        print(f"udp target:  {host}:{udp_port}")
    finally:
        for sock in clients:
            try:
                sock.close()
            except Exception:
                pass
        try:
            udp_sock.close()
        except Exception:
            pass
        hub.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for perception TCP/UDP aggregation.")
    parser.add_argument("--host", default="127.0.0.1", help="local bind host")
    parser.add_argument("--base-port", type=int, default=6601, help="test TCP base port")
    parser.add_argument("--udp-port", type=int, default=7601, help="test UDP receive port")
    parser.add_argument("--verbose", action="store_true", help="print RE/TT messages and hub events")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_test(host=args.host, base_port=args.base_port, udp_port=args.udp_port, verbose=args.verbose)
    except Exception as exc:
        print(f"perception test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
