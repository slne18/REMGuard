"""Extract REM-detector features for every CAP night.

The full database is about 40 GB and this machine does not have room to keep
every EDF. Each new recording is downloaded, checked, reduced to epoch
features, and then the EDF is deleted. The three proof-of-concept EDFs that
were already on disk are left in place. Score files are kept.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mne
import numpy as np

from remguard.data.epochs import build_epoch_timeline
from remguard.features import FEATURE_NAMES, extract_recording_features
from remguard.preprocessing import preprocess_recording

mne.set_log_level("ERROR")

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "features" / "nights"
KEEP_EDF = {"n12", "n6", "rbd1"}
S3 = "https://physionet-open.s3.amazonaws.com/capslpdb/1.0.0"
S3_HOST = "physionet-open.s3.amazonaws.com"
MIN_FREE_BYTES = 3 * 1024 ** 3
_HOST_IP: str | None = None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checksums = _checksums(RAW / "SHA256SUMS.txt")
    recording_ids = sorted(name[:-4] for name in checksums if name.endswith(".edf"))
    print(f"{len(recording_ids)} recordings in SHA256SUMS", flush=True)
    for index, recording_id in enumerate(recording_ids, start=1):
        target = OUT / f"{recording_id}.npz"
        if target.exists():
            print(f"[{index}/{len(recording_ids)}] {recording_id} already extracted", flush=True)
            continue
        print(f"[{index}/{len(recording_ids)}] {recording_id}", flush=True)
        downloaded: list[Path] = []
        try:
            edf, edf_new = _ensure(recording_id, f"{recording_id}.edf", checksums, downloaded)
            _ensure(recording_id, f"{recording_id}.txt", checksums, downloaded)
            reason, table = _features(edf, RAW / f"{recording_id}.txt")
            np.savez_compressed(
                target,
                recording_id=np.asarray(table.recording_id, dtype=str),
                onset_sec=np.asarray(table.onset_sec, dtype=np.float64),
                rem=np.asarray(table.rem, dtype=np.int8),
                values=np.asarray(table.values, dtype=np.float32),
                feature_names=np.asarray(FEATURE_NAMES),
                reason=np.asarray(reason),
            )
            print(f"  {table.n_epochs} epochs ({reason})", flush=True)
            (OUT / f"{recording_id}.error").unlink(missing_ok=True)
        except Exception as exc:
            (OUT / f"{recording_id}.error").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"  failed: {exc}", flush=True)
        finally:
            if target.exists():
                for path in downloaded:
                    if path.suffix == ".edf" and path.stem not in KEEP_EDF:
                        path.unlink(missing_ok=True)
        _write_status(recording_ids)
    print(f"Status: {OUT.parent / 'status.json'}", flush=True)


def _features(edf: Path, text: Path):
    from remguard.features import empty_table

    prepared = preprocess_recording(edf, text)
    if prepared.modality("eeg") is None:
        return "no_eeg", empty_table()
    if prepared.scoring is None or not prepared.scoring.events:
        return "no_scores", empty_table()
    timeline = build_epoch_timeline(prepared.scoring, _duration(prepared))
    table = extract_recording_features(prepared, timeline)
    if table.n_epochs == 0:
        return "no_labeled_epochs", table
    return "ok", table


def _duration(prepared) -> float:
    duration = prepared.header.duration_sec
    if duration == duration and duration > 0:
        return float(duration)
    eeg = prepared.modality("eeg")
    raw = eeg.processed if eeg is not None else prepared.modalities[0].original
    return float(raw.n_times / raw.info["sfreq"])


def _ensure(recording_id: str, filename: str, checksums: dict[str, str], downloaded: list[Path]) -> tuple[Path, bool]:
    path = RAW / filename
    digest = checksums.get(filename)
    if path.exists() and (digest is None or _sha256(path) == digest):
        return path, False
    free = shutil.disk_usage(RAW).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"only {free / 1024 ** 3:.1f} GB free")
    url = f"{S3}/{filename}"
    path.parent.mkdir(parents=True, exist_ok=True)
    _download(url, path, resume=path.exists())
    downloaded.append(path)
    if digest is not None and _sha256(path) != digest:
        path.unlink(missing_ok=True)
        _download(url, path, resume=False)
        downloaded.append(path)
        if _sha256(path) != digest:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {filename}")
    return path, True


def _download(url: str, path: Path, resume: bool) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(2):
        command = [
            "curl",
            "-fL",
            "-sS",
            "--connect-timeout",
            "20",
            "--max-time",
            "14400",
            "--speed-limit",
            "1000",
            "--speed-time",
            "180",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--retry-all-errors",
            "--resolve",
            f"{S3_HOST}:443:{_host_ip(refresh=True)}",
            "-A",
            "Mozilla/5.0",
            "-o",
            str(path),
            url,
        ]
        if resume:
            command.insert(1, "-C")
            command.insert(2, "-")
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


def _host_ip(refresh: bool = False) -> str:
    """Resolve the dataset host through public DNS.

    The machine's router resolver hangs, so system lookups time out. A direct
    query to 8.8.8.8 or 1.1.1.1 still returns an address curl can use.
    """
    global _HOST_IP
    if _HOST_IP is not None and not refresh:
        return _HOST_IP
    for server in ("8.8.8.8", "1.1.1.1"):
        result = subprocess.run(
            ["nslookup", S3_HOST, server],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        for line in result.stdout.splitlines():
            if not line.startswith("Address:"):
                continue
            address = line.split()[-1]
            if "#" in address or ":" in address:
                continue
            _HOST_IP = address
            return address
    raise RuntimeError(f"could not resolve {S3_HOST}")


def _checksums(path: Path) -> dict[str, str]:
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            found[parts[-1]] = parts[0]
    return found


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_status(recording_ids: list[str]) -> None:
    rows = []
    for recording_id in recording_ids:
        npz = OUT / f"{recording_id}.npz"
        error = OUT / f"{recording_id}.error"
        if npz.exists():
            with np.load(npz, allow_pickle=False) as data:
                rows.append(
                    {
                        "recording_id": recording_id,
                        "epochs": int(data["rem"].shape[0]),
                        "reason": str(data["reason"]),
                    }
                )
        elif error.exists():
            rows.append({"recording_id": recording_id, "epochs": 0, "reason": "failed"})
        else:
            rows.append({"recording_id": recording_id, "epochs": 0, "reason": "pending"})
    (OUT.parent / "status.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
