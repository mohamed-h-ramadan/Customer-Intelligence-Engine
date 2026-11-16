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


def test_csv_artifacts_zero_crlf_and_lf_writer_parity(tmp_path):
    """Verify that committed CSV artifacts contain zero CRLF and CSV writers emit pure LF across OSes."""
    import pandas as pd

    root = get_project_root()
    csv_files = list((root / "data" / "processed").glob("**/*.csv"))
    assert len(csv_files) > 0, "No CSV files found in data/processed"

    # 1. Assert every committed CSV has zero \r\n and strictly uses \n
    for csv_file in csv_files:
        raw = csv_file.read_bytes()
        assert b"\r\n" not in raw, f"{csv_file.name} contains CRLF line endings!"
        assert b"\n" in raw, f"{csv_file.name} missing newline terminator!"

    # 2. Test that saving a DataFrame with lineterminator='\n' generates byte-identical LF output
    test_df = pd.DataFrame(
        {
            "col_a": [10, 20, 30],
            "col_b": ["apple", "banana", "cherry"],
        }
    )
    temp_csv = tmp_path / "test_out.csv"
    test_df.to_csv(temp_csv, index=False, lineterminator="\n")

    raw_output = temp_csv.read_bytes()
    assert b"\r\n" not in raw_output, "to_csv with lineterminator='\n' produced CRLF!"
    assert raw_output == b"col_a,col_b\n10,apple\n20,banana\n30,cherry\n"
