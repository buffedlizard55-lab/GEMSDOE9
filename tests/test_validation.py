import sys
from pathlib import Path

def test_imports():
    import src.gems.metric as metric
    import src.gems.features as features
    assert hasattr(metric, 'distance_weighted_tversky')
    assert hasattr(features, 'dilation_tendency')

def test_metric_perfect():
    import numpy as np
    from src.gems.metric import distance_weighted_tversky
    pred = np.zeros((10,10), dtype=np.float32)
    gt = np.zeros((10,10), dtype=bool)
    gt[5,5]=True
    pred[5,5]=1.0
    dti, comp = distance_weighted_tversky(pred, gt)
    assert dti > 0.9, f"perfect should be ~1.0 got {dti}"

def test_submission_exists():
    p = Path("docs/downloads/gemsdoe9_submission.tif")
    assert p.exists(), "submission tif must exist"
    assert p.stat().st_size > 1000

def test_no_duplicate_sha():
    import hashlib
    p = Path("docs/downloads/gemsdoe9_submission.tif")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    # Known duplicates
    duplicates = ["7f00890a62878d61", "f347b70daa", "37f9d5b855", "4e03fc9705", "33cec71ff0", "237f0063a4"]
    for dup in duplicates:
        assert not sha.startswith(dup), f"Artifact is duplicate of {dup}"

if __name__ == "__main__":
    test_imports()
    test_metric_perfect()
    test_submission_exists()
    test_no_duplicate_sha()
    print("All tests PASS")
