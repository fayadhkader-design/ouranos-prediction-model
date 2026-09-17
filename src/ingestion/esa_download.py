"""Select official ESA archive members with verified HTTP ranges and ZIP CRCs.

Avoids downloading a multi-GB archive. Dataset copyright/licensing remains with
its publishers. The manifest records member hashes, not a claim to verify the
full-archive MD5 when only selected members were downloaded.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import urllib.request
import zipfile
import zlib

RECORD = "https://zenodo.org/api/records/12528696"
CHANNELS = [f"channel_{i}" for i in range(41, 47)]


def get_range(url, start, end, total):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        expected = f"bytes {start}-{end}/{total}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("Server did not honor the requested byte range")
        content = response.read(end - start + 2)
    if len(content) != end - start + 1:
        raise ValueError("Truncated range response")
    return content


class ArchiveTail(io.RawIOBase):
    def __init__(self, total, tail):
        self.total, self.tail, self.position = total, tail, 0

    def seek(self, offset, whence=0):
        self.position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.total + offset
        )
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        start = self.position - (self.total - len(self.tail))
        if start < 0:
            raise ValueError("Central directory exceeds cached tail; enlarge tail")
        value = self.tail[start:] if size < 0 else self.tail[start : start + size]
        self.position += len(value)
        return value


def download(destination=Path("data/raw"), channels=CHANNELS, mission="ESA-Mission1"):
    if mission not in {"ESA-Mission1", "ESA-Mission2", "ESA-Mission3"}:
        raise ValueError("Unsupported mission")
    destination = Path(destination)
    with urllib.request.urlopen(RECORD, timeout=60) as response:
        record = json.load(response)
    file = next(f for f in record["files"] if f["key"] == f"{mission}.zip")
    total = file["size"]
    url = file["links"]["self"]
    tail = get_range(url, total - 131072, total - 1, total)
    with zipfile.ZipFile(ArchiveTail(total, tail)) as archive:
        entries = {i.filename: i for i in archive.infolist()}
    selected = [
        f"{mission}/{name}"
        for name in [
            "channels.csv",
            "labels.csv",
            "anomaly_types.csv",
            "telecommands.csv",
        ]
    ]
    selected += [f"{mission}/channels/{name}.zip" for name in channels]
    if any(name not in entries for name in selected):
        raise ValueError("Selected channel missing from official archive")
    destination.mkdir(parents=True, exist_ok=True)
    needed = sum(
        entries[name].file_size
        for name in selected
        if not (destination / name).exists()
    )
    if shutil.disk_usage(destination).free < needed + 1024**3:
        raise ValueError(
            "Need room for selected members plus a 1 GiB free-space reserve"
        )
    manifest = {
        "source": "REAL MISSION DATA",
        "record_url": RECORD,
        "doi": "10.5281/zenodo.12528696",
        "archive": file,
        "selection": {"mission": mission, "channels": list(channels)},
        "integrity": "ZIP member CRC32 and local SHA256; full archive MD5 not verified for partial download",
        "members": [],
    }
    for name in selected:
        info = entries[name]
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == info.file_size:
            crc = 0
            with target.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    crc = zlib.crc32(chunk, crc)
            valid = crc == info.CRC
        else:
            valid = False
        if not valid:
            header = get_range(url, info.header_offset, info.header_offset + 29, total)
            if header[:4] != b"PK\x03\x04":
                raise ValueError("Invalid ZIP local header")
            filename_length, extra_length = struct.unpack_from("<HH", header, 26)
            offset = info.header_offset + 30 + filename_length + extra_length
            inflater = (
                zlib.decompressobj(-15)
                if info.compress_type == zipfile.ZIP_DEFLATED
                else None
            )
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("Unsupported ZIP compression")
            temporary = target.with_suffix(target.suffix + ".part")
            crc = 0
            written = 0
            with temporary.open("wb") as stream:
                for start in range(offset, offset + info.compress_size, 8 * 1024**2):
                    blob = get_range(
                        url,
                        start,
                        min(start + 8 * 1024**2, offset + info.compress_size) - 1,
                        total,
                    )
                    blob = inflater.decompress(blob) if inflater else blob
                    stream.write(blob)
                    written += len(blob)
                    crc = zlib.crc32(blob, crc)
                if inflater:
                    blob = inflater.flush()
                    stream.write(blob)
                    written += len(blob)
                    crc = zlib.crc32(blob, crc)
            if written != info.file_size or crc != info.CRC:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Integrity check failed: {name}")
            temporary.replace(target)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["members"].append(
            {
                "path": name,
                "bytes": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "sha256": digest,
            }
        )
        print(f"Verified {name}: {info.file_size:,} bytes", flush=True)
    manifest_path = destination / mission / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default="data/raw")
    args = parser.parse_args()
    download(Path(args.destination))
