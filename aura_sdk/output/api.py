"""
aura_sdk/output/api.py
───────────────────────
Language: Python
Why: Network I/O, file I/O, and async server logic are Python-native.
     The Rust core computes Q_HCI; this module broadcasts it.

Outputs
────────
  CSVLogger       — Write session data to CSV for analysis
  WebSocketServer — Stream Q_HCI in real time over WebSocket
  MQTTPublisher   — Publish to MQTT broker for IoT integrations
  ConsoleOutput   — Print telemetry to stdout (development)

All outputs implement the same OutputSink interface so the
pipeline can mix and match without code changes.
"""

from __future__ import annotations
import csv
import json
import time
import threading
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable


# ── Telemetry record ───────────────────────────────────────────────────────

@dataclass
class TelemetryRecord:
    """
    One frame of pipeline telemetry.
    Produced by the pipeline, consumed by output sinks.
    """
    frame_idx:      int
    timestamp_ms:   float
    fired:          bool
    q_hci:          Optional[float]
    state:          Optional[int]
    state_label:    Optional[str]
    alert:          bool
    power_uw:       float
    accumulated_uj: float
    energy_nj:      float
    idle_power_w:   float = 0.0   # always 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ── Abstract sink ──────────────────────────────────────────────────────────

class OutputSink(ABC):
    """Abstract output sink. All concrete outputs implement this."""

    @abstractmethod
    def open(self) -> None:
        """Initialise the sink."""

    @abstractmethod
    def write(self, record: TelemetryRecord) -> None:
        """Write one telemetry record."""

    @abstractmethod
    def close(self) -> None:
        """Flush and close the sink."""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()


# ── CSV Logger ─────────────────────────────────────────────────────────────

class CSVLogger(OutputSink):
    """
    Write pipeline telemetry to a CSV file.

    Creates a new timestamped file for each session.
    Suitable for offline analysis in Python/R/Excel.

    Usage
    ─────
        with CSVLogger("sessions/") as log:
            for record in pipeline:
                log.write(record)
    """

    FIELDS = [
        "frame_idx", "timestamp_ms", "fired", "q_hci", "state",
        "state_label", "alert", "power_uw", "accumulated_uj",
        "energy_nj", "idle_power_w",
    ]

    def __init__(self, directory: str = "sessions", prefix: str = "aura_session"):
        self.directory = Path(directory)
        self.prefix    = prefix
        self._file     = None
        self._writer   = None
        self._path     = None

    def open(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        ts          = time.strftime("%Y%m%d_%H%M%S")
        self._path  = self.directory / f"{self.prefix}_{ts}.csv"
        self._file  = open(self._path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def write(self, record: TelemetryRecord) -> None:
        if self._writer:
            self._writer.writerow({
                k: record.to_dict().get(k, "") for k in self.FIELDS
            })

    def close(self) -> None:
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def __repr__(self):
        return f"CSVLogger(path={self._path})"


# ── Console Output ─────────────────────────────────────────────────────────

class ConsoleOutput(OutputSink):
    """
    Print telemetry to stdout.
    Used during development and demos.
    """

    STATE_COLORS = {
        1: "\033[91m",   # red
        2: "\033[93m",   # yellow
        3: "\033[97m",   # white
        4: "\033[96m",   # cyan
        5: "\033[92m",   # green
    }
    RESET = "\033[0m"

    def __init__(self, every_n: int = 1, use_color: bool = True):
        self.every_n   = every_n
        self.use_color = use_color
        self._count    = 0

    def open(self)  -> None: pass
    def close(self) -> None: pass

    def write(self, record: TelemetryRecord) -> None:
        self._count += 1
        if self._count % self.every_n != 0:
            return

        if not record.fired:
            print(f"  [{record.frame_idx:4d}] · (accumulating energy...)")
            return

        col   = self.STATE_COLORS.get(record.state, "") if self.use_color else ""
        reset = self.RESET if self.use_color else ""
        alert = " ⚠ ALERT" if record.alert else ""
        print(
            f"  [{record.frame_idx:4d}] "
            f"{col}Q={record.q_hci:.4f}  "
            f"S={record.state}  {record.state_label}{reset}"
            f"  {record.power_uw:.1f}µW  "
            f"idle={record.idle_power_w}W"
            f"{alert}"
        )


# ── WebSocket Server ───────────────────────────────────────────────────────

class WebSocketServer(OutputSink):
    """
    Stream Q_HCI telemetry over WebSocket in real time.

    Clients (e.g. the React dashboard) connect to ws://localhost:PORT
    and receive JSON telemetry records at the pipeline framerate.

    Requires: pip install websockets

    Usage
    ─────
        server = WebSocketServer(port=8765)
        with server:
            for record in pipeline:
                server.write(record)
        # Dashboard connects to ws://localhost:8765
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host    = host
        self.port    = port
        self._queue: queue.Queue = queue.Queue(maxsize=256)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._clients: set = set()

    def open(self) -> None:
        try:
            import asyncio
            import websockets
        except ImportError:
            raise ImportError(
                "WebSocket server requires:\n"
                "  pip install websockets"
            )
        self._running = True
        self._thread  = threading.Thread(
            target=self._run_server, daemon=True
        )
        self._thread.start()
        print(f"  WebSocket server → ws://{self.host}:{self.port}")

    def _run_server(self) -> None:
        import asyncio
        import websockets

        async def handler(ws, path):
            self._clients.add(ws)
            try:
                await ws.wait_closed()
            finally:
                self._clients.discard(ws)

        async def broadcaster():
            while self._running:
                try:
                    record = self._queue.get_nowait()
                    if self._clients:
                        msg = record.to_json()
                        await asyncio.gather(
                            *[c.send(msg) for c in list(self._clients)],
                            return_exceptions=True,
                        )
                except queue.Empty:
                    pass
                await asyncio.sleep(0.001)

        async def main():
            async with websockets.serve(handler, self.host, self.port):
                await broadcaster()

        asyncio.run(main())

    def write(self, record: TelemetryRecord) -> None:
        if self._running:
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                pass   # Drop frame rather than block

    def close(self) -> None:
        self._running = False

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def __repr__(self):
        return f"WebSocketServer(ws://{self.host}:{self.port})"


# ── MQTT Publisher ─────────────────────────────────────────────────────────

class MQTTPublisher(OutputSink):
    """
    Publish Q_HCI telemetry to an MQTT broker.

    Suitable for IoT integrations — industrial SCADA systems,
    smart building management, medical device gateways.

    Requires: pip install paho-mqtt

    Topic structure:
        aura_m1/{device_id}/q_hci   → float
        aura_m1/{device_id}/state   → int 1-5
        aura_m1/{device_id}/alert   → bool
        aura_m1/{device_id}/telemetry → full JSON
    """

    def __init__(self, broker: str = "localhost", port: int = 1883,
                 device_id: str = "aura_m1_01"):
        self.broker    = broker
        self.port      = port
        self.device_id = device_id
        self._client   = None

    def open(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise ImportError("MQTT publisher requires: pip install paho-mqtt")
        self._client = mqtt.Client(client_id=self.device_id)
        self._client.connect(self.broker, self.port, keepalive=60)
        self._client.loop_start()
        print(f"  MQTT publisher → {self.broker}:{self.port} "
              f"(device={self.device_id})")

    def write(self, record: TelemetryRecord) -> None:
        if not self._client or not record.fired:
            return
        base = f"aura_m1/{self.device_id}"
        self._client.publish(f"{base}/q_hci",     str(record.q_hci))
        self._client.publish(f"{base}/state",     str(record.state))
        self._client.publish(f"{base}/alert",     str(int(record.alert)))
        self._client.publish(f"{base}/telemetry", record.to_json())

    def close(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()

    def __repr__(self):
        return f"MQTTPublisher(broker={self.broker}:{self.port})"


# ── Multi-sink broadcaster ─────────────────────────────────────────────────

class MultiSink(OutputSink):
    """
    Broadcasts to multiple sinks simultaneously.

    Usage
    ─────
        sink = MultiSink([
            CSVLogger("sessions/"),
            ConsoleOutput(every_n=10),
            WebSocketServer(port=8765),
        ])
        with sink:
            for record in pipeline:
                sink.write(record)
    """

    def __init__(self, sinks: list[OutputSink]):
        self.sinks = sinks

    def open(self) -> None:
        for s in self.sinks:
            s.open()

    def write(self, record: TelemetryRecord) -> None:
        for s in self.sinks:
            try:
                s.write(record)
            except Exception:
                pass   # One failing sink must not block others

    def close(self) -> None:
        for s in self.sinks:
            try:
                s.close()
            except Exception:
                pass

    def __repr__(self):
        return f"MultiSink([{', '.join(repr(s) for s in self.sinks)}])"
