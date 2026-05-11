from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MidiNote:
    start_tick: int
    end_tick: int
    pitch: int
    velocity: int
    channel: int

    @property
    def duration_tick(self) -> int:
        return max(0, self.end_tick - self.start_tick)


@dataclass(frozen=True)
class MidiTempo:
    tick: int
    us_per_quarter: int


@dataclass(frozen=True)
class MidiParsed:
    ppq: int
    notes: List[MidiNote]
    tempos: List[MidiTempo]


class MidiParseError(ValueError):
    pass


def _read_u16_be(b: bytes, off: int) -> Tuple[int, int]:
    return int.from_bytes(b[off : off + 2], "big"), off + 2


def _read_u32_be(b: bytes, off: int) -> Tuple[int, int]:
    return int.from_bytes(b[off : off + 4], "big"), off + 4


def _read_vlq(b: bytes, off: int) -> Tuple[int, int]:
    """
    Variable Length Quantity (7-bit chunks).
    """
    val = 0
    while True:
        if off >= len(b):
            raise MidiParseError("unexpected EOF while reading VLQ")
        c = b[off]
        off += 1
        val = (val << 7) | (c & 0x7F)
        if (c & 0x80) == 0:
            break
    return val, off


def _parse_track(track: bytes, ppq: int) -> Tuple[List[MidiNote], List[MidiTempo]]:
    notes: List[MidiNote] = []
    tempos: List[MidiTempo] = []

    tick = 0
    running_status: Optional[int] = None
    active: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (ch,pitch) -> (start_tick, velocity)

    off = 0
    while off < len(track):
        delta, off = _read_vlq(track, off)
        tick += delta

        if off >= len(track):
            break

        status = track[off]
        if status < 0x80:
            # running status
            if running_status is None:
                raise MidiParseError("running status encountered without previous status")
            status = running_status
        else:
            off += 1
            running_status = status

        if status == 0xFF:
            # Meta event
            if off >= len(track):
                raise MidiParseError("unexpected EOF in meta event")
            meta_type = track[off]
            off += 1
            length, off = _read_vlq(track, off)
            data = track[off : off + length]
            off += length

            # Tempo: 0x51, 3 bytes (us per quarter)
            if meta_type == 0x51 and length == 3:
                us_per_quarter = int.from_bytes(data, "big")
                tempos.append(MidiTempo(tick=tick, us_per_quarter=us_per_quarter))
            # End of track: 0x2F
            if meta_type == 0x2F:
                break
            continue

        if status in (0xF0, 0xF7):
            # SysEx: length + data
            length, off = _read_vlq(track, off)
            off += length
            continue

        event_type = status & 0xF0
        channel = status & 0x0F

        def _read_data_byte() -> int:
            nonlocal off
            if off >= len(track):
                raise MidiParseError("unexpected EOF while reading data byte")
            v = track[off]
            off += 1
            return v

        if event_type in (0x80, 0x90):
            pitch = _read_data_byte()
            vel = _read_data_byte()
            key = (channel, pitch)

            if event_type == 0x90 and vel > 0:
                # note on
                active[key] = (tick, vel)
            else:
                # note off (0x80 or 0x90 vel=0)
                start = active.pop(key, None)
                if start is not None:
                    st, v0 = start
                    notes.append(MidiNote(start_tick=st, end_tick=tick, pitch=pitch, velocity=v0, channel=channel))
            continue

        # channel voice messages with fixed lengths
        if event_type in (0xA0, 0xB0, 0xE0):  # 2 data bytes
            _read_data_byte()
            _read_data_byte()
            continue
        if event_type in (0xC0, 0xD0):  # 1 data byte
            _read_data_byte()
            continue

        # Unknown status: best-effort stop
        break

    # close dangling notes (optional: ignore)
    return notes, tempos


def parse_midi_bytes(data: bytes) -> MidiParsed:
    """
    Minimal SMF(MIDI) parser for analysis:
    - Reads header to get PPQ
    - Parses tracks for note on/off and tempo meta
    """
    if len(data) < 14 or data[:4] != b"MThd":
        raise MidiParseError("invalid MIDI header (MThd)")

    header_len = int.from_bytes(data[4:8], "big")
    if header_len < 6:
        raise MidiParseError("invalid MIDI header length")

    fmt = int.from_bytes(data[8:10], "big")
    _ntrks = int.from_bytes(data[10:12], "big")
    division = int.from_bytes(data[12:14], "big")

    if division & 0x8000:
        # SMPTE timecode not supported in this minimal parser
        raise MidiParseError("SMPTE time division not supported")

    ppq = division
    # header chunk is 8 + header_len
    off = 8 + header_len

    all_notes: List[MidiNote] = []
    all_tempos: List[MidiTempo] = []

    # parse track chunks
    while off + 8 <= len(data):
        if data[off : off + 4] != b"MTrk":
            break
        track_len = int.from_bytes(data[off + 4 : off + 8], "big")
        off += 8
        track = data[off : off + track_len]
        off += track_len

        notes, tempos = _parse_track(track, ppq)
        all_notes.extend(notes)
        all_tempos.extend(tempos)

    # sort tempos by tick
    all_tempos.sort(key=lambda t: t.tick)

    # fmt is unused but kept for potential debugging
    _ = fmt
    return MidiParsed(ppq=ppq, notes=all_notes, tempos=all_tempos)


def ticks_to_seconds(ticks: int, ppq: int, tempos: List[MidiTempo]) -> float:
    """
    Convert ticks to seconds using tempo map.
    - If no tempo events, assume 120 BPM (500000 us/qn)
    """
    if ppq <= 0:
        return 0.0

    # default tempo
    if not tempos:
        us_per_quarter = 500_000
        return (ticks / ppq) * (us_per_quarter / 1_000_000.0)

    # piecewise integrate
    total_sec = 0.0
    prev_tick = 0
    prev_us_per_quarter = tempos[0].us_per_quarter

    for t in tempos[1:]:
        if t.tick >= ticks:
            break
        seg_ticks = max(0, t.tick - prev_tick)
        total_sec += (seg_ticks / ppq) * (prev_us_per_quarter / 1_000_000.0)
        prev_tick = t.tick
        prev_us_per_quarter = t.us_per_quarter

    # remaining
    seg_ticks = max(0, ticks - prev_tick)
    total_sec += (seg_ticks / ppq) * (prev_us_per_quarter / 1_000_000.0)
    return total_sec

