#!/usr/bin/env python3
# gun_rx_class.py
"""
High-level, "nice" Python API for your ESP32-C3 PC receiver stream.

Protocol (from firmware):
  Serial baud: 1_000_000 bps (default here, configurable)
  Frame per event: 0xAA 0x55 + ShootPkt
  ShootPkt (little-endian, packed, total 8 bytes):
      uint16  seq
      uint32  t_us          # TX-side micros()
      uint8   buttonState   # bit0=trigger, bit2=A, bit3=B
      uint8   checksum      # two's complement of sum(payload[:-1])

This module exposes:
  - ButtonState dataclass (trig, A, B, raw)
  - GunEvent dataclass (seq, t_us, pc_time_ns, state, latency_ms, ema_latency_ms, edges)
  - GunReceiver class:
      rx = GunReceiver("COM5")                # or "/dev/ttyACM0"
      rx.open()
      ev = rx.read_event(timeout=1.0)         # blocking read of next event
      st = rx.get_buttons()                   # latest state (ButtonState)
      rx.is_pressed('T')                      # convenience: 'T','A','B'
      for ev in rx: ...                       # iteration yields events
      rx.close()

  Also supports context manager:
      with GunReceiver("/dev/ttyACM0") as rx:
          for ev in rx:
              print(ev)

Install dependency:
    pip install pyserial
"""
from __future__ import annotations

import struct
import time
import threading
import collections
from dataclasses import dataclass
from typing import Optional, Iterator, Deque, Tuple, Dict

try:
    import serial  # pyserial
except Exception as e:
    raise SystemExit("pyserial is required: pip install pyserial") from e


HEADER = b"\xAA\x55"
PKT_LEN = 8  # seq(2) + t_us(4) + buttonState(1) + checksum(1)
_STRUCT_PKT = struct.Struct("<H I B B")


@dataclass(frozen=True)
class ButtonState:
    trig: bool
    A: bool
    B: bool
    raw: int  # original byte

    def __getitem__(self, key: str) -> bool:
        key = key.upper()
        if key in ("T", "TRIGGER"):
            return self.trig
        if key == "A":
            return self.A
        if key == "B":
            return self.B
        raise KeyError(f"Unknown button '{key}', use 'T','A','B'")

    def as_tuple(self) -> Tuple[bool, bool, bool]:
        return (self.trig, self.A, self.B)


@dataclass(frozen=True)
class Edges:
    T: Optional[str]  # 'up', 'down', or None
    A: Optional[str]
    B: Optional[str]


@dataclass(frozen=True)
class GunEvent:
    seq: int
    t_us: int
    pc_time_ns: int
    state: ButtonState
    latency_ms: float
    ema_latency_ms: float
    edges: Edges


def _checksum_ok(payload: bytes) -> bool:
    # payload length is PKT_LEN; last byte is checksum so sum(payload) & 0xFF == 0
    return (sum(payload) & 0xFF) == 0


class GunReceiver:
    """High-level receiver for ESP32-C3 PC stream with convenience methods."""

    def __init__(self, port: str, baud: int = 1_000_000, *, read_chunk: int = 256, ema_alpha: float = 0.05):
        self.port = port
        self.baud = baud
        self.read_chunk = max(1, read_chunk)
        self._ema_alpha = float(ema_alpha)

        self._ser: Optional[serial.Serial] = None
        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._buf = bytearray()
        self._q: Deque[GunEvent] = collections.deque(maxlen=1024)
        self._cv = threading.Condition()

        self._base_offset_ns: Optional[int] = None
        self._ema_ms: Optional[float] = None
        self._last_raw_state: Optional[int] = None

    # ---------- Lifecycle ----------
    def open(self) -> None:
        if self._ser is not None:
            return
        self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        self._stop.clear()
        self._thr = threading.Thread(target=self._rx_loop, name="GunReceiverRX", daemon=True)
        self._thr.start()

    def close(self) -> None:
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=1.0)
            self._thr = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def __enter__(self) -> "GunReceiver":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------- Public API ----------
    def read_event(self, timeout: Optional[float] = None) -> Optional[GunEvent]:
        """Blocking read of next event. Returns None on timeout."""
        with self._cv:
            if not self._q:
                # Wait for new data
                if not self._cv.wait(timeout=timeout):
                    return None
            if not self._q:
                return None
            return self._q.popleft()

    def __iter__(self) -> Iterator[GunEvent]:
        """Iterate forever yielding events as they arrive."""
        while True:
            ev = self.read_event(timeout=None)
            if ev is None:
                # Shouldn't happen with None timeout, but keep it safe
                continue
            yield ev

    def get_buttons(self) -> Optional[ButtonState]:
        """Return the latest known button state (or None if not ready)."""
        with self._cv:
            if self._q:
                st = self._q[-1].state
                return st
            if self._last_raw_state is None:
                return None
            return self._decode_state(self._last_raw_state)

    def is_pressed(self, key: str) -> Optional[bool]:
        """Convenience: 'T', 'A', or 'B'. Returns None if no data yet."""
        st = self.get_buttons()
        if st is None:
            return None
        key = key.upper()
        if key in ("T", "TRIGGER"):
            return st.trig
        if key == "A":
            return st.A
        if key == "B":
            return st.B
        raise KeyError(f"Unknown button '{key}', use 'T','A','B'")

    # ---------- Internal ----------
    def _rx_loop(self) -> None:
        ser = self._ser
        if ser is None:
            return

        read = ser.read
        find = self._buf.find

        while not self._stop.is_set():
            try:
                chunk = read(ser.in_waiting or 1)
                if not chunk:
                    continue
                self._buf.extend(chunk)

                while True:
                    i = find(HEADER)
                    if i < 0:
                        # trim buffer to avoid unbounded growth
                        if len(self._buf) > 2048:
                            del self._buf[:-2]
                        break

                    if len(self._buf) - i < 2 + PKT_LEN:
                        # Need more bytes; remove garbage before header
                        if i > 0:
                            del self._buf[:i]
                        break

                    start = i + 2
                    end = start + PKT_LEN
                    payload = bytes(self._buf[start:end])
                    del self._buf[:end]

                    if not _checksum_ok(payload):
                        continue  # drop frame and continue

                    seq, t_us, raw_state, _ = _STRUCT_PKT.unpack(payload)

                    # Compute latency
                    now_ns = time.perf_counter_ns()
                    if self._base_offset_ns is None:
                        self._base_offset_ns = now_ns - (t_us * 1000)

                    latency_ns = now_ns - (self._base_offset_ns + t_us * 1000)
                    latency_ms = latency_ns / 1e6

                    if self._ema_ms is None:
                        self._ema_ms = latency_ms
                    else:
                        a = self._ema_alpha
                        self._ema_ms = self._ema_ms + a * (latency_ms - self._ema_ms)

                    state = self._decode_state(raw_state)
                    edges = self._edges_from(self._last_raw_state, raw_state)
                    self._last_raw_state = raw_state

                    ev = GunEvent(
                        seq=seq,
                        t_us=t_us,
                        pc_time_ns=now_ns,
                        state=state,
                        latency_ms=latency_ms,
                        ema_latency_ms=self._ema_ms,
                        edges=edges,
                    )

                    with self._cv:
                        self._q.append(ev)
                        self._cv.notify()

            except Exception:
                # On any parsing/serial error, brief pause and continue
                time.sleep(0.005)

    @staticmethod
    def _decode_state(raw: int) -> ButtonState:
        trig = bool((raw >> 0) & 0x1)
        A    = bool((raw >> 2) & 0x1)
        B    = bool((raw >> 3) & 0x1)
        return ButtonState(trig=trig, A=A, B=B, raw=raw & 0xFF)

    @staticmethod
    def _edges_from(prev_raw: Optional[int], curr_raw: int) -> Edges:
        if prev_raw is None:
            return Edges(T=None, A=None, B=None)

        def edge(bit: int) -> Optional[str]:
            p = (prev_raw >> bit) & 1
            c = (curr_raw >> bit) & 1
            if p == c:
                return None
            return "up" if c else "down"

        return Edges(
            T=edge(0),
            A=edge(2),
            B=edge(3),
        )


# ---------- CLI example ----------
if __name__ == "__main__":
    with GunReceiver("COM3", 1_000_000) as rx:
        print("Listening... (Ctrl+C to stop)")
        try:
            for ev in rx:
                t = time.strftime("%H:%M:%S")
                st = ev.state
                ed = ev.edges
                print(f"{t}  seq={ev.seq:5d}  state[T,A,B]=[{int(st.trig)},{int(st.A)},{int(st.B)}]"
                      f"  lat={ev.latency_ms:6.3f}  ema={ev.ema_latency_ms:6.3f}"
                      f"  edges[T,A,B]=[{ed.T},{ed.A},{ed.B}]")
        except KeyboardInterrupt:
            print("\nStopped.")