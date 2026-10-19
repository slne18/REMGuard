"""Read classic EDF headers without loading the signal samples."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path


@dataclass(frozen=True)
class EdfSignalHeader:
    index: int
    label: str
    physical_dimension: str
    physical_min: float
    physical_max: float
    digital_min: float
    digital_max: float
    prefiltering: str
    samples_per_record: int
    sampling_frequency_hz: float

    @property
    def physical_min_volts(self) -> float | None:
        return _to_volts(self.physical_min, self.physical_dimension)

    @property
    def physical_max_volts(self) -> float | None:
        return _to_volts(self.physical_max, self.physical_dimension)


@dataclass(frozen=True)
class EdfHeader:
    path: str
    start_date: str
    start_time: time
    start_datetime_naive: datetime
    n_records: int
    record_duration_sec: float
    duration_sec: float
    reserved: str
    signals: tuple[EdfSignalHeader, ...]

    @property
    def has_embedded_annotations(self) -> bool:
        """Classic EDF has an empty reserved field. EDF+ stores EDF+C or EDF+D there."""
        token = self.reserved.upper()
        return "EDF+C" in token or "EDF+D" in token


def read_edf_header(path: str | Path) -> EdfHeader:
    path = Path(path)
    with path.open("rb") as handle:
        fixed = handle.read(256)
        n_signals = int(_field(fixed, 252, 4))
        variable = handle.read(n_signals * 256)
    blob = fixed + variable
    record_duration = float(_field(fixed, 244, 8))
    n_records = int(_field(fixed, 236, 8))
    start_date = _field(fixed, 168, 8)
    start_clock = _field(fixed, 176, 8)
    hour, minute, second = (int(part) for part in start_clock.split("."))
    day, month, year = (int(part) for part in start_date.split("."))
    year += 2000 if year < 85 else 1900
    start_time = time(hour, minute, second)
    signals = _parse_signals(blob, n_signals, record_duration)
    duration = n_records * record_duration if n_records > 0 else float("nan")
    return EdfHeader(
        path=str(path),
        start_date=start_date,
        start_time=start_time,
        start_datetime_naive=datetime(year, month, day, hour, minute, second),
        n_records=n_records,
        record_duration_sec=record_duration,
        duration_sec=duration,
        reserved=_field(fixed, 192, 44),
        signals=tuple(signals),
    )


def _parse_signals(blob: bytes, n_signals: int, record_duration: float) -> list[EdfSignalHeader]:
    offset = 256
    labels = [_field(blob, offset + i * 16, 16) for i in range(n_signals)]
    offset += n_signals * 16
    offset += n_signals * 80  # transducer type
    dimensions = [_field(blob, offset + i * 8, 8) for i in range(n_signals)]
    offset += n_signals * 8
    physical_min = [_float_field(blob, offset + i * 8) for i in range(n_signals)]
    offset += n_signals * 8
    physical_max = [_float_field(blob, offset + i * 8) for i in range(n_signals)]
    offset += n_signals * 8
    digital_min = [_float_field(blob, offset + i * 8) for i in range(n_signals)]
    offset += n_signals * 8
    digital_max = [_float_field(blob, offset + i * 8) for i in range(n_signals)]
    offset += n_signals * 8
    prefiltering = [_field(blob, offset + i * 80, 80) for i in range(n_signals)]
    offset += n_signals * 80
    samples = [int(_field(blob, offset + i * 8, 8)) for i in range(n_signals)]
    signals: list[EdfSignalHeader] = []
    for index, label in enumerate(labels):
        signals.append(
            EdfSignalHeader(
                index=index,
                label=label,
                physical_dimension=dimensions[index],
                physical_min=physical_min[index],
                physical_max=physical_max[index],
                digital_min=digital_min[index],
                digital_max=digital_max[index],
                prefiltering=prefiltering[index],
                samples_per_record=samples[index],
                sampling_frequency_hz=samples[index] / record_duration,
            )
        )
    return signals


def _to_volts(value: float, dimension: str) -> float | None:
    unit = dimension.strip().lower().replace("µ", "u")
    if unit in {"v", "volt", "volts"}:
        return value
    if unit in {"mv"}:
        return value * 1e-3
    if unit in {"uv"}:
        return value * 1e-6
    return None


def _field(blob: bytes, start: int, width: int) -> str:
    return blob[start : start + width].decode("latin1", errors="replace").strip()


def _float_field(blob: bytes, start: int) -> float:
    text = _field(blob, start, 8)
    return float(text) if text else float("nan")
