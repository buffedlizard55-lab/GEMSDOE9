import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gems9 import dti, topk, candidate, validate
from refresh_feed import extract


def test_dti_perfect_and_near():
    gt = np.zeros((11, 11), bool); gt[5, 5] = True
    exact = np.zeros_like(gt); exact[5, 5] = True
    near = np.zeros_like(gt); near[5, 6] = True
    far = np.zeros_like(gt); far[5, 9] = True
    assert dti(exact, gt) == pytest.approx(1)
    assert dti(near, gt) == pytest.approx((2/3) / ((2/3) + .2/3 + .8/3))
    assert dti(far, gt) == pytest.approx(0)
    assert dti(np.zeros_like(gt), gt) == pytest.approx(0)


def test_topk_no_zero_score_or_ineligible():
    s = np.array([[0, 5, 5], [np.nan, 10, 0]], np.float32)
    eligible = np.array([[True, True, True], [True, False, True]])
    assert topk(s, eligible, 4).tolist() == [[False, True, True], [False, False, False]]


def test_signed_gradient_synthetic():
    # Opposing depth/gravity: signal; parallel: zero, regardless of units.
    x = np.tile(np.arange(30, dtype=np.float32), (30, 1))
    valid = np.ones_like(x, dtype=bool)
    opposite = candidate({'depth_to_base_surf': x, 'iso_grav_anom': -x}, valid)['basin_anti_gradient']
    parallel = candidate({'depth_to_base_surf': x, 'iso_grav_anom': x}, valid)['basin_anti_gradient']
    assert opposite[10, 10] > 0
    assert parallel[10, 10] == 0


def test_feed_parse_and_fail_closed():
    html = '<table><tr><th>Rank</th><th>Team members</th><th>Participant</th><th>Score</th></tr><tr><td>#1</td><td></td><td>DARD</td><td>0.3049</td></tr></table>'
    assert extract(html)['public_dti'] == .3049
    with pytest.raises(ValueError): extract('<p>login required</p>')


def test_real_template_validator(tmp_path):
    import rasterio
    template = Path(__file__).resolve().parents[1] / 'data' / 'sample_submission.tif'
    if not template.exists(): pytest.skip('download template to run raster format tests')
    with rasterio.open(template) as src:
        original = src.read(1)
        profile = src.profile
    dest = tmp_path / 's.tif'
    with rasterio.open(dest, 'w', **profile) as out: out.write(original, 1)
    assert validate(dest, template)['format_valid']
    original[1000, 1000] = np.inf
    with rasterio.open(dest, 'w', **profile) as out: out.write(original, 1)
    with pytest.raises(ValueError): validate(dest, template)
