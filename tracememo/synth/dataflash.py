"""Write the synthetic flight as an ArduPilot DataFlash ``.bin`` log.

The message definitions mirror ArduCopter 4.x (``XKF1``, ``PSCN``/``PSCE``/``PSCD``, ``BCN``,
``EV``) so the ArduPilot adapter's default mapping is exercised on a realistic file layout.
Positions are written in ArduPilot's NED frame; the adapter converts back to ENU.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracememo.synth.drone import DroneSynthData

HEAD = b"\xa3\x95"
FMT_TYPE = 0x80
EV_ARMED = 10
EV_DISARMED = 11

# DataFlash format char -> struct code (subset used here)
STRUCT_CODES = {
    "B": "B",
    "b": "b",
    "c": "h",
    "C": "H",
    "e": "i",
    "f": "f",
    "Q": "Q",
    "n": "4s",
    "N": "16s",
    "Z": "64s",
}


@dataclass(frozen=True)
class MessageDef:
    """A DataFlash message definition (one ``FMT`` record)."""

    type: int
    name: str
    fmt: str
    columns: str

    @property
    def struct_fmt(self) -> str:
        """Little-endian struct format for the payload."""
        return "<" + "".join(STRUCT_CODES[c] for c in self.fmt)

    @property
    def length(self) -> int:
        """Total message length including the 3-byte header."""
        return 3 + struct.calcsize(self.struct_fmt)


# ArduCopter 4.x-style definitions.
XKF1 = MessageDef(
    0x10, "XKF1", "QBccCfffffffccce", "TimeUS,C,Roll,Pitch,Yaw,VN,VE,VD,dPD,PN,PE,PD,GX,GY,GZ,OH"
)
PSCN = MessageDef(0x11, "PSCN", "Qffffffff", "TimeUS,TPN,PN,DVN,TVN,VN,DAN,TAN,AN")
PSCE = MessageDef(0x12, "PSCE", "Qffffffff", "TimeUS,TPE,PE,DVE,TVE,VE,DAE,TAE,AE")
PSCD = MessageDef(0x13, "PSCD", "Qffffffff", "TimeUS,TPD,PD,DVD,TVD,VD,DAD,TAD,AD")
BCN = MessageDef(0x14, "BCN", "QBBfffffff", "TimeUS,Health,Cnt,D0,D1,D2,D3,PosX,PosY,PosZ")
EV = MessageDef(0x15, "EV", "QB", "TimeUS,Id")
MSG = MessageDef(0x16, "MSG", "QZ", "TimeUS,Message")
DEFS = [XKF1, PSCN, PSCE, PSCD, BCN, EV, MSG]


def fmt_record(d: MessageDef) -> bytes:
    """Encode the ``FMT`` record describing ``d``."""
    payload = struct.pack(
        "<BB4s16s64s", d.type, d.length, d.name.encode(), d.fmt.encode(), d.columns.encode()
    )
    return HEAD + bytes([FMT_TYPE]) + payload


def record(d: MessageDef, *values: object) -> bytes:
    """Encode one message of definition ``d``."""
    return HEAD + bytes([d.type]) + struct.pack(d.struct_fmt, *values)


def write_dataflash_bin(
    data: DroneSynthData, path: Path, arm_time_us: int = 5_000_000, prearm_s: float = 1.0
) -> Path:
    """Write ``data`` as a DataFlash ``.bin``; return the path.

    The log starts ``prearm_s`` before the ``EV`` ARMED event at ``arm_time_us`` with the
    vehicle sitting at the origin, so the adapter's arming-time normalization is exercised.
    A second EKF core (``C=1``) is written with a small offset to exercise instance filtering.
    """
    path = Path(path)
    out = bytearray()
    out += fmt_record(MessageDef(FMT_TYPE, "FMT", "BBnNZ", "Type,Length,Name,Format,Columns"))
    for d in DEFS:
        out += fmt_record(d)
    out += record(MSG, arm_time_us - int(prearm_s * 1e6), b"ArduCopter V4.5 (tracememo synthetic)")

    pose = data.pose
    nav = data.nav_target
    beacon = data.beacon
    dt_us = int(round(1e6 * float(np.median(np.diff(pose["t_s"])))))

    def us(t_s: float) -> int:
        return arm_time_us + int(round(t_s * 1e6))

    # Pre-arm samples: stationary at the origin, must be dropped by the adapter.
    n_pre = int(prearm_s * 1e6 / dt_us)
    for i in range(n_pre, 0, -1):
        t_us = arm_time_us - i * dt_us
        out += record(XKF1, t_us, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
    out += record(EV, arm_time_us, EV_ARMED)

    # Interleave by time: pose (XKF1 x2 cores), targets, beacon.
    events: list[tuple[int, int, bytes]] = []
    for row in pose.itertuples(index=False):
        t_us = us(row.t_s)
        pn, pe, pd_ = row.y_m, row.x_m, -row.z_m  # ENU -> NED
        roll_c = int(round(np.degrees(row.roll_rad) * 100))
        pitch_c = int(round(np.degrees(row.pitch_rad) * 100))
        yaw_c = int(round(np.degrees(row.yaw_rad) % 360.0 * 100))
        events.append(
            (
                t_us,
                0,
                record(
                    XKF1,
                    t_us,
                    0,
                    roll_c,
                    pitch_c,
                    yaw_c,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    pn,
                    pe,
                    pd_,
                    0,
                    0,
                    0,
                    0,
                ),
            )
        )
        events.append(
            (
                t_us,
                1,
                record(
                    XKF1,
                    t_us,
                    1,
                    roll_c,
                    pitch_c,
                    yaw_c,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    pn + 0.5,
                    pe + 0.5,
                    pd_,
                    0,
                    0,
                    0,
                    0,
                ),
            )
        )
    for row in nav.itertuples(index=False):
        t_us = us(row.t_s)
        events.append((t_us, 2, record(PSCN, t_us, row.y_m, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        events.append((t_us, 3, record(PSCE, t_us, row.x_m, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        events.append((t_us, 4, record(PSCD, t_us, -row.z_m, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    for row in beacon.itertuples(index=False):
        t_us = us(row.t_s)
        events.append(
            (
                t_us,
                5,
                record(
                    BCN, t_us, int(row.quality), 4, 0.0, 0.0, 0.0, 0.0, row.y_m, row.x_m, -row.z_m
                ),
            )
        )
    events.sort(key=lambda e: (e[0], e[1]))
    for _, _, rec in events:
        out += rec
    out += record(EV, us(float(pose["t_s"].iloc[-1])) + dt_us, EV_DISARMED)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path
