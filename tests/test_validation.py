"""Tests for the DTI metric, artifact integrity, and duplicate guards.

Run from the repo root:  python tests/test_validation.py
(or via pytest). No PYTHONPATH needed — the repo root is added to sys.path.
"""

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import src.gems.metric as metric
import src.gems.features as features  # noqa: F401  (import check)
from src.gems.metric import (
    distance_weighted_tversky,
    distance_weighted_tversky_fast,
)


def test_imports():
    assert hasattr(metric, 'distance_weighted_tversky')
    assert hasattr(metric, 'distance_weighted_tversky_fast')
    assert hasattr(features, 'dilation_tendency')


def test_metric_perfect():
    pred = np.zeros((10, 10), dtype=np.float32)
    gt = np.zeros((10, 10), dtype=bool)
    gt[5, 5] = True
    pred[5, 5] = 1.0
    dti, comp = distance_weighted_tversky_fast(pred, gt)
    assert dti > 0.99, f"perfect prediction should be ~1.0, got {dti}"
    assert abs(comp['TP_w'] - 1.0) < 1e-6
    assert comp['FN_w'] == 0.0
    assert comp['FP_w'] == 0.0


def test_metric_all_zero_prediction():
    gt = np.zeros((10, 10), dtype=bool)
    gt[5, 5] = True
    pred = np.zeros((10, 10), dtype=np.float32)
    dti, comp = distance_weighted_tversky_fast(pred, gt)
    assert dti == 0.0
    assert abs(comp['FN_w'] - 1.0) < 1e-6


def test_metric_off_by_one_fp():
    """One perfect hit + one adjacent FP at distance 1 (kernel 2/3)."""
    pred = np.zeros((10, 10), dtype=np.float32)
    gt = np.zeros((10, 10), dtype=bool)
    gt[5, 5] = True
    pred[5, 5] = 1.0
    pred[5, 6] = 1.0
    dti, comp = distance_weighted_tversky_fast(pred, gt, alpha=0.2, beta=0.8)
    # TP_w = 1.0 ; FP: the extra pixel has k_to_gt = 1-1/3 = 2/3 -> FP_w = 1/3 ; FN_w = 0
    expected = 1.0 / (1.0 + 0.2 * (1.0 / 3.0) + 0.8 * 0.0 + 1e-7)
    assert abs(dti - expected) < 1e-6, f"{dti} != {expected}"


def test_metric_diagonal_distance():
    """FP at diagonal distance sqrt(2): kernel weight 1 - sqrt(2)/3."""
    pred = np.zeros((12, 12), dtype=np.float32)
    gt = np.zeros((12, 12), dtype=bool)
    gt[6, 6] = True
    pred[7, 7] = 1.0  # no hit on gt itself -> TP comes from kernel at d=sqrt(2)
    dti, comp = distance_weighted_tversky_fast(pred, gt)
    k = 1 - np.sqrt(2) / 3
    expected_tp = k
    expected_fp = 1.0 - k
    expected_fn = 1.0 - k
    assert abs(comp['TP_w'] - expected_tp) < 1e-6
    assert abs(comp['FP_w'] - expected_fp) < 1e-6
    assert abs(comp['FN_w'] - expected_fn) < 1e-6


def test_metric_kernel_range():
    """A prediction 4 px away (d > R=3) contributes nothing to TP."""
    pred = np.zeros((12, 12), dtype=np.float32)
    gt = np.zeros((12, 12), dtype=bool)
    gt[6, 6] = True
    pred[6, 10] = 1.0  # distance 4
    dti, comp = distance_weighted_tversky_fast(pred, gt)
    assert comp['TP_w'] == 0.0
    assert abs(comp['FP_w'] - 1.0) < 1e-6  # outside kernel: full FP weight
    assert abs(comp['FN_w'] - 1.0) < 1e-6


def test_metric_empty_gt():
    pred = np.ones((5, 5), dtype=np.float32) * 0.5
    gt = np.zeros((5, 5), dtype=bool)
    dti, comp = distance_weighted_tversky_fast(pred, gt)
    assert dti == 0.0
    assert abs(comp['FP_w'] - 0.5 * 25) < 1e-6


def test_metric_fp_ignore_mask():
    """Official masking: predicted mass on catalogue pixels pays no FP penalty.

    Verified behaviour per DrivenData staff (forum thread 11516, 2026-09-16):
    'Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded
    from evaluation, so they do not count towards penalty terms.'
    """
    pred = np.zeros((10, 10), dtype=np.float32)
    gt = np.zeros((10, 10), dtype=bool)
    gt[5, 5] = True
    pred[5, 5] = 1.0
    pred[1, 1] = 1.0  # far FP
    mask = np.zeros((10, 10), dtype=bool)
    mask[1, 1] = True  # (1,1) is a known-catalogue pixel -> excluded
    dti_masked, comp_m = distance_weighted_tversky_fast(pred, gt, fp_ignore_mask=mask)
    dti_unmasked, comp_u = distance_weighted_tversky_fast(pred, gt)
    assert comp_m['FP_w'] == 0.0
    assert comp_u['FP_w'] == 1.0
    assert dti_masked > dti_unmasked
    assert abs(dti_masked - 1.0) < 1e-6


def test_metric_fast_matches_naive_random():
    """Exact equivalence of fast and naive implementations on random fields."""
    rng = np.random.default_rng(7)
    for trial in range(3):
        H, W = 40, 45
        pred = rng.random((H, W)).astype(np.float32)
        gt = rng.random((H, W)) < 0.02
        if not gt.any():
            gt[H // 2, W // 2] = True
        d_fast, c_fast = distance_weighted_tversky_fast(pred, gt)
        d_naive, c_naive = distance_weighted_tversky(pred, gt)
        assert abs(d_fast - d_naive) < 1e-9, f"trial {trial}: {d_fast} vs {d_naive}"
        for key in ('TP_w', 'FP_w', 'FN_w'):
            assert abs(c_fast[key] - c_naive[key]) < 1e-9, f"trial {trial} {key}"


def test_metric_fast_matches_naive_line():
    """GT as a vertical line (the shape used in the official scoring example)."""
    pred = np.zeros((20, 20), dtype=np.float32)
    gt = np.zeros((20, 20), dtype=bool)
    gt[4:16, 10] = True
    rng = np.random.default_rng(3)
    pred = (rng.random((20, 20)) * 0.3).astype(np.float32)
    pred[4:16, 10] = 0.9  # strong predictions on the line
    d_fast, c_fast = distance_weighted_tversky_fast(pred, gt)
    d_naive, c_naive = distance_weighted_tversky(pred, gt)
    assert abs(d_fast - d_naive) < 1e-9
    for key in ('TP_w', 'FP_w', 'FN_w'):
        assert abs(c_fast[key] - c_naive[key]) < 1e-9


def test_metric_shape_validation():
    try:
        distance_weighted_tversky_fast(np.zeros((3, 3)), np.zeros((3, 4), dtype=bool))
        raise AssertionError("expected ValueError for shape mismatch")
    except ValueError:
        pass


def test_submission_exists():
    p = REPO_ROOT / "docs/downloads/gemsdoe9_submission.tif"
    assert p.exists(), "submission tif must exist"
    assert p.stat().st_size > 1000


def test_zip_matches_tif():
    """The shipped zip must contain a byte-identical copy of the submission tif."""
    import zipfile
    tif = REPO_ROOT / "docs/downloads/gemsdoe9_submission.tif"
    zp = REPO_ROOT / "docs/downloads/gemsdoe9_submission.zip"
    assert zp.exists()
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        assert len(names) == 1 and names[0].endswith(".tif")
        inner = z.read(names[0])
    assert hashlib.sha256(inner).hexdigest() == hashlib.sha256(tif.read_bytes()).hexdigest()


def test_no_duplicate_sha():
    p = REPO_ROOT / "docs/downloads/gemsdoe9_submission.tif"
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    # Known duplicate artifacts from previous group repos (the 0.1563 pattern)
    duplicates = ["7f00890a62878d61", "f347b70daa", "37f9d5b855",
                  "4e03fc9705", "33cec71ff0", "237f0063a4", "b966d47c7c02b1c2"]
    for dup in duplicates:
        assert not sha.startswith(dup), f"Artifact is duplicate of {dup}"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    if failed:
        print(f"{failed} test(s) FAILED")
        sys.exit(1)
    print(f"All {len(fns)} tests PASS")
