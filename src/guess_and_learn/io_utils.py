"""
Light-weight S3 utility - no external deps beyond boto3.

• Set RESULTS_S3_PREFIX="s3://my-bucket/gnl-runs" (env-var or flag)
  - leave unset to operate purely on local disk.

All paths are handled as *relative* paths from your results/
directory, so the on-disk layout is mirrored in S3.
"""

from __future__ import annotations
import os
from pathlib import Path

import boto3
import botocore.exceptions as botoexc

# ------------------------------------------------------------------ #
#  Public-facing knob                                                #
# ------------------------------------------------------------------ #
S3_PREFIX = os.getenv("RESULTS_S3_PREFIX", "").rstrip("/")
_HAS_S3 = S3_PREFIX.startswith("s3://")
_s3 = boto3.client("s3") if _HAS_S3 else None


def _split(prefix: str) -> tuple[str, str]:
    assert prefix.startswith("s3://")
    trunk = prefix[5:]
    bucket, *rest = trunk.split("/", 1)
    key_prefix = rest[0] if rest else ""
    return bucket, key_prefix


def _to_key(local_path: Path) -> tuple[str, str]:
    bucket, root = _split(S3_PREFIX)
    rel = local_path.as_posix().lstrip("/")
    key = f"{root.rstrip('/')}/{rel}" if root else rel
    return bucket, key


# ------------------------------------------------------------------ #
#  Public helpers                                                    #
# ------------------------------------------------------------------ #
def s3_enabled() -> bool:
    return _HAS_S3


def s3_exists(local_path: Path) -> bool:
    """True = object already present *in S3* (irrespective of local file)."""
    if not _HAS_S3:
        return False
    bucket, key = _to_key(local_path)
    try:
        _s3.head_object(Bucket=bucket, Key=key)
        return True
    except botoexc.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def s3_download(local_path: Path) -> bool:
    """Fetch from S3 **only if** local file is missing.  Returns True if file now exists."""
    if local_path.exists():
        return True
    if not s3_exists(local_path):
        return False
    bucket, key = _to_key(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _s3.download_file(bucket, key, str(local_path))
    return True


def s3_upload(local_path: Path, quiet: bool = False) -> None:
    if not _HAS_S3 or not local_path.exists():
        return
    bucket, key = _to_key(local_path)
    if not quiet:
        print(f"⇡  s3://{bucket}/{key}")
    _s3.upload_file(str(local_path), bucket, key)
