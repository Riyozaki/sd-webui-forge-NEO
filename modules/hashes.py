import hashlib
import os.path

from modules import shared
import modules.cache

dump_cache = modules.cache.dump_cache
cache = modules.cache.cache


def calculate_sha256_real(filename):
    with open(filename, "rb") as f:
        # file_digest streams into a reusable buffer in CPython, avoiding a new
        # bytes allocation for every MiB of multi-gigabyte checkpoints. It only
        # exists on Python 3.11+, which is what Neo ships with, but the manual
        # loop keeps older interpreters working.
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(f, "sha256").hexdigest()

        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


def calculate_sha256(filename):
    print("Calculating real hash: ", filename)
    return calculate_sha256_real(filename)


def forge_fake_calculate_sha256(filename):
    basename = os.path.basename(filename)
    hash_sha256 = hashlib.sha256()
    hash_sha256.update(basename.encode('utf-8'))
    return hash_sha256.hexdigest()


def sha256_from_cache(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    try:
        stat = os.stat(filename)
    except FileNotFoundError:
        return None

    if title not in hashes:
        return None

    entry = hashes[title]
    cached_sha256 = entry.get("sha256")
    if cached_sha256 is None:
        return None

    cached_mtime_ns = entry.get("mtime_ns")
    cached_size = entry.get("size")
    if cached_mtime_ns is not None and cached_size is not None:
        if cached_mtime_ns != stat.st_mtime_ns or cached_size != stat.st_size:
            return None
    elif stat.st_mtime > entry.get("mtime", 0):
        # Backwards compatibility with cache entries created before nanosecond
        # timestamps and file sizes were recorded.
        return None

    return cached_sha256


def sha256(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")

    sha256_value = sha256_from_cache(filename, title, use_addnet_hash)
    if sha256_value is not None:
        return sha256_value

    if shared.cmd_opts.no_hashing:
        return None

    print(f"Calculating sha256 for {filename}: ", end='', flush=True)
    sha256_value = calculate_sha256_real(filename)
    print(f"{sha256_value}")

    stat = os.stat(filename)
    hashes[title] = {
        "mtime": stat.st_mtime,
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "sha256": sha256_value,
    }

    dump_cache()

    return sha256_value


def addnet_hash_safetensors(b):
    """kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()

