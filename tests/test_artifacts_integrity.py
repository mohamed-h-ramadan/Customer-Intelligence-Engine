"""Cryptographic integrity verification test for committed serving artifacts."""

import hashlib

from cie.config import get_project_root


def test_committed_artifacts_match_sha256_manifest():
    """Verify that every file listed in artifacts.sha256 exists and matches its hash."""
    root = get_project_root()
    manifest_path = root / "artifacts.sha256"
    assert manifest_path.exists(), "artifacts.sha256 manifest must exist"

    lines = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(lines) > 0, "artifacts.sha256 manifest cannot be empty"

    for line in lines:
        parts = line.split(maxsplit=1)
        assert len(parts) == 2, f"Invalid manifest line: {line}"
        expected_sha, rel_path = parts[0], parts[1]

        file_path = root / rel_path
        assert file_path.exists(), f"Committed artifact missing: {rel_path}"

        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)

        computed_sha = h.hexdigest()
        assert computed_sha == expected_sha, (
            f"Checksum mismatch for {rel_path}! Expected: {expected_sha}, Computed: {computed_sha}"
        )
