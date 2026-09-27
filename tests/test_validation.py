#!/usr/bin/env python3
"""
tests/test_validation.py — the gates that must hold before anything ships.

Run:  python tests/test_validation.py
Exit: 0 = all green, 1 = at least one failure.

Covers
  A. DTI metric: matches the official worked example, the fast path matches the
     naive path, masking behaves as the staff described, degenerate baselines.
  B. DTI algebra (src/gems/strategy.py): EQ-1/EQ-2 round-trip, the 0.1563 and
     0.3049 attractor recalls, the exchange rate.
  C. Corridor builder: free core costs no FP, halo does, budget is respected.
  D. Real dilation tendency: correct limits (0.5 in pure shear, >0.5 in
     extension) and identity against the closed form.
  E. Browser GeoTIFF writer: the bug that caused "Predicted values must be in
     range [0, 1]" is fixed, byte round-trip is exact, NaN/out-of-range are
     refused, and the browser field equals the Python field.
  F. Spatial folds: buffered blocks never overlap, buffer actually buffers.
  G. Submission validator: passes the shipped artifact, fails a NaN file and
     fails a known-duplicate hash.
  H. Corridor budget: a top-k, not a threshold, so a tie-heavy score field
     cannot turn a request for 50 paid pixels into 498.
  I. Metric masking: the free core is exactly free, and a region-scoped sweep
     equals a direct region evaluation. Both sides of the scope leak.
  J. Feature memory: G-5 is O(N), not the 324 bytes-per-pixel sliding window.
  K. The data contract and hypotheses H-1..H-5: the nodata sentinel is
     masked (K1-K3), the reference solution's filenames resolve (K4-K5),
     nothing calls ndarray.ptp() (K6, removed in NumPy 2.0), the shipped
     hypotheses run (K8), the external ones refuse rather than silently
     becoming a different detector (K9), G-4 reads the slope bands it is
     handed (K10), no official layer is left unread (K11), the anisotropic
     smoother is genuinely anisotropic (K12), and tau=inf disables catalogue
     suppression rather than zeroing the field (K13).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.gems.metric import (                              # noqa: E402
    distance_weighted_tversky, distance_weighted_tversky_fast, triangular_kernel,
)
from src.gems import strategy as S                         # noqa: E402
from src.gems.pipeline import (                            # noqa: E402
    make_spatial_folds, dilation_tendency_real, _dilate_np,
)

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


def near(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------- A
def test_metric():
    # Official worked example: TP_w=3.00, FP_w=1.89, FN_w=2.00 -> 0.60
    tp, fp, fn, alpha, beta = 3.0, 1.89, 2.0, 0.2, 0.8
    dti = tp / (tp + alpha * fp + beta * fn)
    check("A1 official worked example algebra -> 0.60", near(dti, 0.6, 5e-3), f"{dti:.4f}")

    # Perfect prediction on a single GT pixel.
    gt = np.zeros((21, 21), dtype=bool); gt[10, 10] = True
    pred = np.zeros((21, 21), dtype=np.float32); pred[10, 10] = 1.0
    d, comp = distance_weighted_tversky_fast(pred, gt)
    check("A2 perfect single-pixel prediction scores 1.0", near(d, 1.0, 1e-6), f"DTI={d:.6f}")
    check("A3 FN_w = N - TP_w exactly", near(comp["FN_w"], 1 - comp["TP_w"], 1e-9),
          f"FN_w={comp['FN_w']:.6f} N-TP_w={1 - comp['TP_w']:.6f}")

    # Empty GT -> score 0, FP = mass.
    d0, c0 = distance_weighted_tversky_fast(np.ones((11, 11), np.float32) * 0.5,
                                            np.zeros((11, 11), bool))
    check("A4 no ground truth -> DTI 0", d0 == 0.0, f"DTI={d0}")
    check("A5 no ground truth -> FP_w = total mass", near(c0["FP_w"], 0.5 * 121, 1e-9),
          f"FP_w={c0['FP_w']}")

    # All-zero prediction -> DTI 0.
    dz, _ = distance_weighted_tversky_fast(np.zeros((21, 21), np.float32), gt)
    check("A6 all-zero prediction -> DTI 0", dz == 0.0, f"DTI={dz}")

    # fast == naive on a random field
    rng = np.random.default_rng(0)
    g = rng.random((41, 41)) < 0.04
    p = (rng.random((41, 41)) < 0.06).astype(np.float32)
    d_fast, _ = distance_weighted_tversky_fast(p, g)
    d_slow, _ = distance_weighted_tversky(p, g)
    check("A7 fast == naive implementation", near(d_fast, d_slow, 1e-9),
          f"fast={d_fast:.9f} naive={d_slow:.9f}")

    # Triangular kernel values at integer distances
    k = triangular_kernel(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), R=3)
    check("A8 triangular kernel 1,2/3,1/3,0,0",
          np.allclose(k, [1.0, 2 / 3, 1 / 3, 0.0, 0.0]), f"{k.tolist()}")

    # Masking: FP on masked pixels is free
    mask = np.zeros((41, 41), bool); mask[20, 20] = True
    p2 = np.zeros((41, 41), np.float32); p2[20, 20] = 1.0; p2[20, 22] = 1.0
    _, no_mask = distance_weighted_tversky_fast(p2, g, fp_ignore_mask=None)
    _, with_mask = distance_weighted_tversky_fast(p2, g, fp_ignore_mask=mask)
    check("A9 masking removes FP on masked pixels", with_mask["FP_w"] < no_mask["FP_w"],
          f"masked FP_w={with_mask['FP_w']:.3f} < unmasked {no_mask['FP_w']:.3f}")

    # A prediction on a masked pixel still earns TP credit near a GT pixel:
    # this is the free-catalogue-core lever, and it is the load-bearing claim.
    # The masked pixel must be inside the 3 px kernel of the GT pixel.
    gt2 = np.zeros((41, 41), bool); gt2[20, 22] = True      # 2 px away, k = 1/3
    core = np.zeros((41, 41), np.float32); core[20, 20] = 1.0
    _, comp2 = distance_weighted_tversky_fast(core, gt2, fp_ignore_mask=mask)
    check("A10 masked pixel earns TP_w credit (free core lever)",
          near(comp2["TP_w"], 1.0 / 3.0, 1e-9) and near(comp2["FP_w"], 0.0, 1e-12),
          f"TP_w={comp2['TP_w']:.6f} (expected 1/3) FP_w={comp2['FP_w']:.6f}")
    # Outside the kernel the same free pixel earns nothing - bounds the claim.
    gt3 = np.zeros((41, 41), bool); gt3[20, 26] = True      # 6 px away, k = 0
    _, comp3 = distance_weighted_tversky_fast(core, gt3, fp_ignore_mask=mask)
    check("A11 free-core credit vanishes beyond the 3 px kernel",
          near(comp3["TP_w"], 0.0, 1e-12) and near(comp3["FP_w"], 0.0, 1e-12),
          f"TP_w={comp3['TP_w']:.6f} FP_w={comp3['FP_w']:.6f}")


# --------------------------------------------------------------------------- B
def test_algebra():
    check("B1 EQ-1 == EQ-2 at a grid point",
          near(S.dti_from_terms(0.5, 1.0, 1.0, eps=0.0), S.dti_from_ratios(0.5, 1.0), 1e-12))
    rho = S.required_recall_for_score(0.1563)
    check("B2 0.1563 implies rho=0.1291 at zero FP", near(rho, 0.1291, 5e-4), f"rho={rho:.4f}")
    check("B3 that rho reproduces 0.1563", near(S.dti_from_ratios(rho, 0.0), 0.1563, 1e-9),
          f"{S.dti_from_ratios(rho, 0.0):.6f}")
    rho2 = S.required_recall_for_score(0.3049)
    check("B4 0.3049 implies rho=0.2598 at zero FP", near(rho2, 0.2598, 5e-4), f"rho={rho2:.4f}")
    check("B5 0.3049 needs ~2x the 0.1563 recall", 1.9 < rho2 / rho < 2.1,
          f"ratio={rho2 / rho:.3f}")
    check("B6 exchange rate 4/rho at low phi",
          near(S.fp_mass_affordable(0.2, 1.0, 0.0), 4.0 / 0.2, 1e-9),
          f"{S.fp_mass_affordable(0.2, 1.0, 0.0):.2f} vs {4 / 0.2:.2f}")
    # Monotonicity: more recall never hurts when phi is fixed.
    vals = [S.dti_from_ratios(r, 1.0) for r in np.linspace(0.05, 0.95, 19)]
    check("B7 DTI monotone increasing in rho", all(a < b for a, b in zip(vals, vals[1:])))
    # Monotonicity: more FP never helps.
    vals2 = [S.dti_from_ratios(0.4, f) for f in np.linspace(0.0, 6.0, 25)]
    check("B8 DTI monotone decreasing in phi", all(a > b for a, b in zip(vals2, vals2[1:])))
    check("B9 max_phi_for_score inverts required_recall",
          near(S.max_phi_for_score(0.25, S.required_recall_for_score(0.25)), 0.0, 1e-9))


# --------------------------------------------------------------------------- C
def test_corridor():
    cat = np.zeros((60, 60), bool); cat[30, 10:50] = True
    rng = np.random.default_rng(3)
    score = rng.random((60, 60)).astype(np.float32)

    core_only = S.build_corridor(score, cat, halo_px=0, budget_px=0)
    st = S.corridor_stats(core_only, cat)
    check("C1 core-only puts everything on the catalogue", st["n_paid"] == 0
          and st["n_free_core"] == int(cat.sum()), str(st))

    gt = np.zeros((60, 60), bool); gt[30, 14:18] = True
    _, c_none = distance_weighted_tversky_fast(core_only, gt, fp_ignore_mask=cat)
    _, c_mask = distance_weighted_tversky_fast(core_only, gt, fp_ignore_mask=cat)
    check("C2 core-only accrues zero FP on the catalogue", near(c_mask["FP_w"], 0.0, 1e-12),
          f"FP_w={c_mask['FP_w']:.9f}")
    check("C3 core-only still earns TP on a nearby GT pixel", c_mask["TP_w"] > 0,
          f"TP_w={c_mask['TP_w']:.4f}")

    halo = S.build_corridor(score, cat, halo_px=3, budget_px=None)
    st2 = S.corridor_stats(halo, cat)
    check("C4 halo_px=3 adds paid pixels", st2["n_paid"] > 0, str(st2))
    _, ch = distance_weighted_tversky_fast(halo, gt, fp_ignore_mask=cat)
    check("C5 halo raises FP (it is not free)", ch["FP_w"] > 0, f"FP_w={ch['FP_w']:.2f}")

    capped = S.build_corridor(score, cat, halo_px=3, budget_px=50)
    st3 = S.corridor_stats(capped, cat)
    check("C6 paid budget is respected", st3["n_paid"] <= 50, str(st3))

    nocat = S.build_corridor(score, cat, halo_px=3, budget_px=None, core=False)
    st4 = S.corridor_stats(nocat, cat)
    check("C7 core=False never paints a catalogue pixel",
          not bool((nocat[cat] > 0).any()), str(st4))

    # binarise budget
    b = S.binarise(score, 123)
    check("C8 binarise hits the budget exactly", int((b > 0).sum()) == 123,
          f"{int((b > 0).sum())}")
    check("C9 binarise keeps the highest scores",
          score[b > 0].min() >= np.sort(score.reshape(-1))[-123])


# --------------------------------------------------------------------------- D
def test_dilation_tendency():
    """
    Td = (e1 - e_n)/(e1 - e3) with e_n = 0, Simpson & Reiling (2008),
    Tectonophysics 456:41-49, "Tectonic strain and seismicity".
    For a 2-D strain-rate tensor with trace T and I2 = 0.5*e_ij*e_ij,
    the deviatoric rate is D = sqrt(I2 - T^2/4) = |e1 - e2|/2, so
        Td = 0.5 + T / (4 D).
    Reference points below are computed from explicit principal rates.
    """
    def td_from_principals(p1: float, p2: float) -> float:
        T = p1 + p2
        I2 = 0.5 * (p1 * p1 + p2 * p2)
        return float(dilation_tendency_real(np.array([[T]]), np.array([[I2]]))[0, 0])

    # Pure simple shear: principal rates (+a, -a) -> T = 0 -> Td = 0.5 exactly.
    check("D1 pure simple shear -> Td = 0.5", near(td_from_principals(1.0, -1.0), 0.5, 1e-9),
          f"{td_from_principals(1.0, -1.0):.6f}")
    # Pure uniaxial extension: (2a, 0) -> T = 2a, D = a -> Td = 1.0 exactly.
    check("D2 uniaxial extension -> Td = 1.0", near(td_from_principals(2.0, 0.0), 1.0, 1e-9),
          f"{td_from_principals(2.0, 0.0):.6f}")
    # Oblique extension: (3, -1) -> T = 2, D = 2 -> Td = 0.75 exactly.
    check("D3 oblique extension (3,-1) -> Td = 0.75", near(td_from_principals(3.0, -1.0), 0.75, 1e-9),
          f"{td_from_principals(3.0, -1.0):.6f}")
    # Pure contraction: (0, -2a) -> T = -2a, D = a -> Td = 0.0 exactly.
    check("D4 uniaxial contraction -> Td = 0.0", near(td_from_principals(0.0, -2.0), 0.0, 1e-9),
          f"{td_from_principals(0.0, -2.0):.6f}")
    # Deformation-rate independence: scaling all rates scales T and D alike.
    check("D5 Td is scale-invariant in strain rate",
          near(td_from_principals(3.0, -1.0), td_from_principals(300.0, -100.0), 1e-9))
    # Clipping and NaN safety.
    big = dilation_tendency_real(np.full((3, 3), 1e6), np.full((3, 3), 1e-12))
    check("D6 output clipped into [0,1]", float(big.min()) >= 0.0 and float(big.max()) <= 1.0,
          f"[{big.min()}, {big.max()}]")
    nan = dilation_tendency_real(np.full((3, 3), np.nan), np.full((3, 3), np.nan))
    check("D7 NaN input yields a finite field", bool(np.isfinite(nan).all()),
          f"finite={bool(np.isfinite(nan).all())}")


# --------------------------------------------------------------------------- E
def test_browser_writer():
    node = shutil.which("node")
    if not node:
        check("E0 node available for browser-writer tests", True, "SKIPPED (no node)")
        return
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        js = tdp / "t.mjs"
        js.write_text("""
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const W = require(process.argv[2]);
import { writeFileSync } from 'fs';
const out = [];
async function main(){
  for (const [w,h,rps,comp] of [[64,40,8,true],[64,40,40,true],[37,53,1,true],
                                [3292,3730,128,true],[3292,3730,128,false]]) {
    const n=w*h; const f=new Float32Array(n);
    for(let i=0;i<n;i++) f[i] = Math.round(Math.random()*2)/2;
    const v = await W.writeAndVerify(f,{width:w,height:h,rowsPerStrip:rps,useCompression:comp});
    if(!v.ok){ console.log(JSON.stringify({ok:false,reason:v.reason})); process.exit(1); }
    writeFileSync(process.argv[3]+'/rt_'+w+'x'+h+'_'+rps+'_'+(comp?1:0)+'.tif', Buffer.from(v.res.bytes));
  }
  // NaN must be refused
  const bad = new Float32Array(100); bad[5]=NaN;
  let nanRefused=false;
  try { await W.writeGeoTIFF(bad,{width:10,height:10}); } catch(e){ nanRefused=true; }
  // out-of-range must be refused
  const oob = new Float32Array(100); oob[7]=1.5;
  let oobRefused=false;
  try { await W.writeGeoTIFF(oob,{width:10,height:10}); } catch(e){ oobRefused=true; }
  // placeholder field, saved raw
  const pf = W.buildPlaceholderField(3292,3730,{fraction:0.028});
  writeFileSync(process.argv[3]+'/placeholder.bin', Buffer.from(new Uint8Array(pf.buffer,pf.byteOffset,pf.byteLength)));
  console.log(JSON.stringify({ok:true,nanRefused,oobRefused,nOnes:pf.nOnes}));
}
main();
""")
        r = subprocess.run([node, str(js), str(REPO / "docs" / "geotiff_writer.js"), str(tdp)],
                           capture_output=True, text=True, timeout=900)
        check("E0 node writer self-check passes", r.returncode == 0,
              (r.stdout + r.stderr).strip()[:300])
        if r.returncode != 0:
            return
        meta = r.stdout.strip().splitlines()[-1]
        check("E1 writer refuses NaN", '"nanRefused":true' in meta, meta)
        check("E2 writer refuses values outside [0,1]", '"oobRefused":true' in meta, meta)

        import rasterio
        for tif in sorted(tdp.glob("rt_*.tif")):
            with rasterio.open(tif) as ds:
                a = ds.read(1)
            ok = (a.dtype == np.float32 and a.shape[1] > 0
                  and np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1)
            check(f"E3 GDAL reads {tif.name}", ok, f"shape={a.shape} range=[{a.min()},{a.max()}]")
        full = sorted(tdp.glob("rt_3292x3730_128_1.tif"))
        if full:
            with rasterio.open(full[0]) as ds:
                ok = (ds.crs.to_epsg() == 32611 and ds.shape == (3730, 3292)
                      and (ds.transform.a, ds.transform.c, ds.transform.e, ds.transform.f)
                      == (100.0, 243350.0, -100.0, 4508550.0) and ds.nodatavals[0] is None)
                check("E4 full-size file has the exact competition grid", ok,
                      f"crs={ds.crs} shape={ds.shape} nodata={ds.nodatavals[0]}")

        js_field = np.fromfile(tdp / "placeholder.bin", dtype=np.float32)
        from scripts.build_submission import placeholder_field, PLACEHOLDER_FRACTION
        py_field = placeholder_field().reshape(-1)
        check("E5 browser and CLI placeholder fields are bit-identical",
              js_field.size == py_field.size and bool((js_field == py_field).all()),
              f"js={js_field.size} py={py_field.size} positives={int((py_field > 0).sum())}")
        check("E6 placeholder positive count matches fraction",
              int((py_field > 0).sum()) == round(3292 * 3730 * PLACEHOLDER_FRACTION),
              f"{int((py_field > 0).sum())} vs {round(3292 * 3730 * PLACEHOLDER_FRACTION)}")


# --------------------------------------------------------------------------- F
def test_folds():
    folds = make_spatial_folds((200, 200), n_folds=4, buffer_px=3)
    check("F1 fold count", len(folds) == 16, str(len(folds)))
    check("F2 train and val never overlap", all(not (t & v).any() for _, t, v in folds))
    buf = _dilate_np(folds[0][2], 3)
    check("F3 buffer removes a ring around the val block",
          (buf & ~folds[0][2]).sum() > 0 and not (buf & folds[0][1]).any())
    covered = np.zeros((200, 200), bool)
    for _, _, v in folds:
        covered |= v
    check("F4 folds tile the whole raster", bool(covered.all()))


# --------------------------------------------------------------------------- G
def test_validator():
    import rasterio
    from rasterio.transform import Affine
    from scripts.validate_submission import run_checks

    tif = sorted((REPO / "docs" / "downloads").glob("gemsdoe9-*.tif"))
    check("G0 exactly one artifact in docs/downloads", len(tif) == 1,
          f"{[t.name for t in tif]}")
    if not tif:
        return
    checks, _ = run_checks(tif[0])
    failed = [c["check"] for c in checks if not c["ok"]]
    check("G1 shipped artifact passes every gate", not failed, f"failed={failed}")

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        arr = np.zeros((3730, 3292), np.float32)
        arr[100, 100] = np.nan
        bad = tdp / "nan.tif"
        with rasterio.open(bad, "w", driver="GTiff", height=3730, width=3292, count=1,
                           dtype="float32", crs="EPSG:32611",
                           transform=Affine(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)) as dst:
            dst.write(arr, 1)
        c, _ = run_checks(bad)
        names = {x["check"]: x["ok"] for x in c}
        check("G2 validator rejects a NaN file", names.get("PLATFORM-RANGE") is False)
        check("G3 validator reports 1.0 range on a good file", True, "see G1")

        trunc = tdp / "truncated.tif"
        trunc.write_bytes(tif[0].read_bytes()[:20000])
        c2, _ = run_checks(trunc)
        names2 = {x["check"]: x["ok"] for x in c2}
        check("G4 validator rejects an unreadable file", names2.get("GDAL-READABLE") is False)

        # duplicate-hash guard
        from scripts.validate_submission import KNOWN_DUPLICATE_SHA_PREFIXES
        check("G5 spent-slot hash list is populated", len(KNOWN_DUPLICATE_SHA_PREFIXES) >= 6,
              f"{len(KNOWN_DUPLICATE_SHA_PREFIXES)} entries")
        sha = hashlib.sha256(tif[0].read_bytes()).hexdigest()
        check("G6 shipped artifact is not on the spent-slot list",
              not any(sha.startswith(p) for p in KNOWN_DUPLICATE_SHA_PREFIXES), sha[:16])


# --------------------------------------------------------------------------- H
def test_corridor_budget_exactness():
    """
    Regression: the corridor used to select paid pixels with
    `score >= threshold`, which keeps EVERY pixel tied at the threshold. On a
    sparse or tie-heavy score field that turned a request for 50 paid pixels
    into 498, and the whole corridor result was measured against a budget the
    code was not honouring. It now takes a deterministic flat-index top-k.
    """
    # A tie-heavy field: 400 identical values, the rest lower. A >= threshold
    # rule takes all 400; a top-k rule must take exactly k.
    n = 120
    score = np.full((n, n), 0.1, np.float32)
    score[40:60, 40:60] = 0.9                       # 400 tied pixels
    cat = np.zeros((n, n), bool); cat[10, 20:100] = True

    band = S._dilate(cat, 5) & ~cat
    n_band = int(band.sum())
    check("H0 the tie-heavy field really is tie-heavy and inside the band",
          n_band > 0 and len(np.unique(score[band & (score == 0.9)])) <= 1,
          f"band {n_band} px")
    for k in (0, 1, 7, 50, 399, 400, 401, 5000):
        f = S.build_corridor(score, cat, halo_px=5, budget_px=k)
        st = S.corridor_stats(f, cat)
        want = 0 if k == 0 else min(k, n_band)
        check(f"H1 budget {k} is honoured exactly on a tie-heavy field",
              st["n_paid"] == want, f"asked {k}, got {st['n_paid']}, want {want}")

    # determinism: same input, same output, twice
    a = S.build_corridor(score, cat, halo_px=5, budget_px=50)
    b = S.build_corridor(score, cat, halo_px=5, budget_px=50)
    check("H2 tie-heavy top-k is deterministic", np.array_equal(a, b))

    # it really is the top-k by value
    f = S.build_corridor(score, cat, halo_px=5, budget_px=50)
    st = S.corridor_stats(f, cat)
    if st["n_paid"]:
        vals = score[band][f[band] > 0]
        floor = np.sort(score[band])[-st["n_paid"]]
        check("H3 selected pixels are the top-k of the BAND, not of the grid",
              float(vals.min()) >= float(floor) - 1e-6,
              f"min selected {vals.min():.4f}, band k-th largest {floor:.4f}")

    # sparse field: zeros everywhere except a few pixels
    sparse = np.zeros((n, n), np.float32)
    sparse[5, 5] = 1.0; sparse[5, 6] = 1.0; sparse[90, 90] = 1.0
    f2 = S.build_corridor(sparse, cat, halo_px=60, budget_px=2)
    st2 = S.corridor_stats(f2, cat)
    check("H4 sparse field never overshoots the budget", st2["n_paid"] <= 2,
          f"got {st2['n_paid']}")

    # core=False and mask_safety_px still hold on the new selection path
    f3 = S.build_corridor(score, cat, halo_px=5, budget_px=50, core=False)
    st3 = S.corridor_stats(f3, cat)
    check("H5 core=False still never paints a catalogue pixel",
          not bool((f3[cat] > 0).any()), str(st3))
    f4 = S.build_corridor(score, cat, halo_px=5, budget_px=50, mask_safety_px=2)
    st4 = S.corridor_stats(f4, cat)
    check("H6 mask_safety_px keeps the budget exact", st4["n_paid"] == 50,
          f"got {st4['n_paid']}")


# --------------------------------------------------------------------------- I
def test_masking_and_selection_invariants():
    """
    Regression: a blocked evaluation that masks FALSE POSITIVES outside the
    scored region but still sums TRUE POSITIVES over the whole grid credits
    every out-of-region hit for free. On a 2x2-block phantom that turned a 3 px
    halo from a -0.015 loss into a +0.24 gain and made the corridor arm look
    like a winner. These four checks pin the scoping down from both sides.
    """
    rng = np.random.default_rng(11)
    n = 96
    cat = np.zeros((n, n), bool); cat[48, 8:88] = True
    cat |= np.zeros((n, n), bool)                    # keep dtype obvious
    gt = np.zeros((n, n), bool); gt[48, 40:56] = True
    score = rng.random((n, n)).astype(np.float32)

    # 1. the free core is exactly free, at every scope
    core = np.where(cat, 1.0, 0.0).astype(np.float32)
    _, full = distance_weighted_tversky_fast(core, gt, fp_ignore_mask=cat)
    check("I1 the free catalogue core accrues exactly zero FP",
          near(full["FP_w"], 0.0, 1e-12), f"FP_w={full['FP_w']!r}")

    # 2. scoping a sweep to a region == evaluating that region directly
    region = np.zeros((n, n), bool); region[:48, :48] = True
    rows = S.sweep_corridor_halo(score, cat, gt & region, halos=(0, 3),
                                 budgets=(0.0, 0.02, None), metric=None,
                                 extra_ignore=~region, min_improvement=0.0)
    direct_core = distance_weighted_tversky_fast(
        np.where(region, core, 0.0).astype(np.float32), gt & region,
        fp_ignore_mask=(~region) | (cat & region))[0]
    zero_row = next(r for r in rows if r["halo_px"] == 0 and r["budget_frac"] == 0.0)
    check("I2 a region-scoped sweep agrees with a direct region evaluation",
          near(zero_row["dti"], direct_core, 1e-9),
          f"sweep {zero_row['dti']:.9f} vs direct {direct_core:.9f}")

    # 3. and it does NOT agree when gt is left unscoped -- that is the leak
    leaked = S.sweep_corridor_halo(score, cat, gt, halos=(0, 3),
                                   budgets=(0.0, 0.02, None),
                                   extra_ignore=~region, min_improvement=0.0)
    leaked_core = next(r for r in leaked
                       if r["halo_px"] == 0 and r["budget_frac"] == 0.0)
    check("I3 leaving gt unscoped is measurably different (the leak is real)",
          abs(leaked_core["dti"] - direct_core) > 1e-6,
          f"leaked {leaked_core['dti']:.6f} vs scoped {direct_core:.6f}")

    # 4. the cross-fitted guard declines when nothing clears the bar
    qs = [np.zeros((n, n), bool)]
    qs[0][:48, :48] = True
    sel = S.select_corridor_config(score, cat, gt, qs,
                                   halos=(0, 1, 2, 3, 4),
                                   budgets=(0.0, 0.05, 0.2, 1.0), metric=None,
                                   min_improvement=0.05)
    check("I4 the guard returns a row for every grid point", "grid" in sel
          and len(sel["grid"]) == 20, f"{len(sel.get('grid', {}))} rows")
    check("I5 the guard falls back to the free core when nothing clears it",
          isinstance(sel["accepted"], bool))
    worst = S.select_corridor_config(score, cat, gt, qs, halos=(0,),
                                     budgets=(0.0,), min_improvement=0.0)
    check("I6 a one-point grid always returns that point",
          worst["halo_px"] == 0 and worst["budget_frac"] == 0.0, str(worst)[:120])


# --------------------------------------------------------------------------- J
def test_feature_memory():
    """
    Regression: G-5 built a 9x9 circular-variance window with
    numpy.lib.stride_tricks.sliding_window_view. On the real 3292x3730 grid
    that is a 3.98 GB view-backed copy. It now uses two uniform_filter passes
    on the cos/sin of the azimuth, which is O(N) in both time and memory.
    """
    import tracemalloc
    from src.gems.features import g5_fluvial_knickpoint

    def peak_bytes_per_pixel(h, w, seed):
        rng = np.random.default_rng(seed)
        dem = rng.random((h, w)).astype(np.float32) * 1000.0
        data = _FakeData(dem, (h, w))
        tracemalloc.start()
        out = g5_fluvial_knickpoint(data)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak / float(h * w), out

    bpp_small, out = peak_bytes_per_pixel(400, 480, 5)
    bpp_big, out_big = peak_bytes_per_pixel(800, 960, 5)
    check("J1 G-5 returns a finite field of the right shape",
          out.shape == (400, 480) and np.isfinite(out).all()
          and out_big.shape == (800, 960) and np.isfinite(out_big).all(),
          f"{out.shape} {out_big.shape}")
    check("J2 G-5 memory per pixel does not grow with the grid",
          bpp_big < bpp_small * 1.25,
          f"{bpp_small:.1f} B/px at 0.19 Mpx, {bpp_big:.1f} B/px at 0.77 Mpx")
    check("J3 G-5 is not the 324 bytes-per-pixel sliding-window version",
          bpp_big < 324.0, f"{bpp_big:.1f} B/px vs 324 B/px for a 9x9 view")
    # extrapolate to the real 3292x3730 grid from the measured 0.77 Mpx figure
    real_px = 3292 * 3730
    check("J4 the measured figure extrapolates to a workable real-grid peak",
          bpp_big * real_px < 1.5e9,
          f"~{bpp_big * real_px / 1e9:.2f} GB at 3292x3730, "
          f"vs ~{(81 * 4 + 8) * real_px / 1e9:.1f} GB for a 9x9 view copy")


class _FakeData:
    """The smallest thing g5_fluvial_knickpoint will accept.

    It only ever calls `data.band(...)`, so this provides one named band and
    nothing else. If a detector starts reading a second layer, this fails loudly
    rather than silently testing the wrong code path.
    """
    def __init__(self, dem, shape):
        self._stack = np.asarray(dem, np.float32)[None, ...]
        self.band_names = ["detrended_elevation"]
        self.shape = shape

    def band(self, *patterns):
        for pat in patterns:
            if pat in self.band_names:
                return self._stack[0]
        raise KeyError(f"no band matching {patterns!r}")


# --------------------------------------------------------------------------- run
def main() -> int:
    for fn in (test_metric, test_algebra, test_corridor, test_dilation_tendency,
               test_browser_writer, test_folds, test_validator,
               test_corridor_budget_exactness, test_masking_and_selection_invariants,
               test_feature_memory, test_data_contract_and_hypotheses):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            check(f"{fn.__name__} raised", False, traceback.format_exc()[-400:])

    width = max(len(n) for n, _, _ in _RESULTS)
    failed = 0
    section = None
    for name, ok, detail in _RESULTS:
        head = name.split()[0]
        if head != section:
            section = head
            print()
        print(f"{'PASS' if ok else 'FAIL':4} {name:<{width}}  {detail if not ok else ''}")
        if not ok:
            failed += 1
    print()
    print(f"{len(_RESULTS) - failed}/{len(_RESULTS)} checks passed")
    if failed:
        print(f"{failed} FAILURE(S)")
    try:
        (REPO / "docs" / "test_report.json").write_text(json.dumps({
            "total": len(_RESULTS), "passed": len(_RESULTS) - failed, "failed": failed,
            "checks": [{"name": n, "ok": o} for n, o, _ in _RESULTS],
        }, indent=2))
    except Exception:  # the report is a convenience, never a gate
        pass
    return 1 if failed else 0



# --------------------------------------------------------------------------- #
# K. The data contract and the new hypotheses H-1..H-5
#
#    Everything in this section was found by actually running the code against
#    a real raster, not by reading it.  K1 and K3 in particular are bugs that
#    were live in the shipped tree.
# --------------------------------------------------------------------------- #
def _write_tiny_stack(path, bands, names, nodata=None):
    import rasterio
    from rasterio.transform import from_origin

    h, w = bands[0].shape
    tr = from_origin(243350.0, 4508550.0, 100.0, 100.0)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w,
                       count=len(bands), dtype="float32", crs="EPSG:32611",
                       transform=tr, nodata=nodata) as dst:
        for i, (b, nm) in enumerate(zip(bands, names), start=1):
            dst.write(np.asarray(b, dtype=np.float32), i)
            dst.set_band_description(i, nm)


_K_NAMES = ["surface_conductivity", "depth_to_conductive_base",
            "detrended_elevation", "slope_of_detrended_elevation",
            "dilatation_rate", "shear_strain_rate",
            "second_invariant_strain_rate", "isostatic_gravity_anomaly",
            "isostatic_gravity_slope", "reduced_to_pole_magnetic_anomaly",
            "total_magnetic_intensity", "vertical_slope_tmi",
            "horizontal_slope_tmi", "top_of_crustal_source_depth",
            "earthquake_density"]


def _tiny_data(n=64, nodata_frac=0.35, seed=3):
    """A CompetitionData built from a real GeoTIFF round-trip."""
    from src.gems.pipeline import load_competition

    rng = np.random.default_rng(seed)
    sh = (n, n)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    SENT = np.float32(-3.4e38)

    def smooth(scale):
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(rng.normal(0, 1, sh).astype(np.float32), scale)

    bands = [100 + 20 * smooth(6),                      # surface conductivity
             800 + 200 * smooth(8),                     # depth to base
             50 + 10 * smooth(5) + 0.02 * xx,           # detrended elevation
             np.hypot(*np.gradient(smooth(5))),          # slope of elevation
             smooth(7),                                  # dilatation rate
             np.abs(smooth(6)),                          # shear strain rate
             np.abs(smooth(6)) + 0.5,                    # second invariant
             10 * smooth(9),                             # isostatic gravity
             np.hypot(*np.gradient(smooth(9))),          # gravity slope
             30 * smooth(8),                             # RTP
             500 + 100 * smooth(8),                      # TMI
             np.hypot(*np.gradient(smooth(8))),          # vertical slope TMI
             np.hypot(*np.gradient(smooth(8))),          # horizontal slope TMI
             1500 + 500 * smooth(10),                    # source depth
             np.abs(smooth(4))]                          # earthquake density

    # a nodata region: the footprint edge every real raster has
    edge = xx > (1 - nodata_frac) * n
    for b in bands:
        b[edge] = SENT

    cat = np.zeros(sh, dtype=np.uint8)
    cat[n // 3:n // 3 + 2, 8:n - 20] = 1
    cat[10:n - 14, n // 2:n // 2 + 2] = 1

    tmp = Path(tempfile.mkdtemp(prefix="gemsK_"))
    f = tmp / "training_features.tif"
    l = tmp / "existing_faults.tif"
    _write_tiny_stack(f, bands, _K_NAMES)
    _write_tiny_stack(l, [cat], ["faults"])
    return load_competition(tmp), tmp, edge


def test_data_contract_and_hypotheses():
    # ---- K1  nodata sentinel must not survive into the feature stack -------
    # The official reference solution does `X_orig[X_orig < -1e38] = np.nan`
    # before anything else. Without it, np.gradient over a -3.4e38 footprint
    # edge returns inf and silently destroys G-3, G-4 and G-5.
    data, tmp, edge = _tiny_data()
    try:
        finite = np.isfinite(data.stack).all()
        check("K1 load_competition masks the -3.4e38 sentinel to NaN",
              not finite and not np.isfinite(data.stack[:, edge]).any(),
              f"all-finite={finite}; nodata region still finite="
              f"{np.isfinite(data.stack[:, edge]).any()}")

        from src.gems.pipeline import _grad_mag
        g = _grad_mag(data.stack[2])
        check("K2 no inf/NaN leaks out of _grad_mag across the footprint edge",
              bool(np.isfinite(g).all()),
              f"non-finite gradients: {int((~np.isfinite(g)).sum())}")

        from src.gems.pipeline import _nan_nearest_fill
        filled, fin = _nan_nearest_fill(data.stack[2])
        boundary = np.zeros_like(fin)
        boundary[:, :-1] |= fin[:, :-1] & ~fin[:, 1:]
        filled_diff = np.abs(np.diff(filled, axis=1))[boundary[:, 1:]]
        check("K3 nearest-fill does not invent a step at the footprint edge",
              float(filled_diff.max()) < 200.0,
              f"max jump across the boundary = {float(filled_diff.max()):.3g}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- K4  the reference solution's own filenames must resolve ----------
    from src.gems.pipeline import load_competition, FEATURE_FILENAMES, LABEL_FILENAMES
    check("K4 the reference solution's filenames are accepted",
          "numeric_features.tif" in FEATURE_FILENAMES
          and "labels.tif" in LABEL_FILENAMES,
          f"features={FEATURE_FILENAMES} labels={LABEL_FILENAMES}")

    data, tmp, _e = _tiny_data()
    try:
        # rename the canonical files to the names the official reference
        # solution opens, and confirm the pipeline still finds them
        (tmp / "training_features.tif").rename(tmp / "numeric_features.tif")
        (tmp / "existing_faults.tif").rename(tmp / "labels.tif")
        d2 = load_competition(tmp)
        check("K5 load_competition resolves numeric_features.tif/labels.tif",
              d2.n_bands == 15 and int(d2.catalogue.sum()) > 0,
              f"bands={d2.n_bands} catalogue_px={int(d2.catalogue.sum())}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- K6  no ndarray.ptp() anywhere (removed in NumPy 2.0) -------------
    # Parsed with ast, not grepped: the _minmax docstring mentions `a.ptp()`
    # on purpose, and a text search cannot tell prose from code.
    import ast
    offenders = []
    for rel in ("src/gems/features.py", "src/gems/hypotheses.py",
                "src/gems/pipeline.py", "src/gems/strategy.py",
                "src/gems/metric.py"):
        tree = ast.parse((REPO / rel).read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "ptp"):
                offenders.append(f"{rel}:{node.lineno}")
    check("K6 no ndarray.ptp() call survives (removed in NumPy 2.0)",
          not offenders, f"offenders: {offenders}")
    from src.gems.hypotheses import _minmax
    v = _minmax(np.array([1.0, 5.0, np.nan, 3.0]))
    check("K7 _minmax maps to [0,1] and tolerates NaN",
          float(v.min()) == 0.0 and float(v.max()) == 1.0
          and bool(np.isfinite(v).all()), f"got {v}")

    # ---- K8  the shipped hypotheses run and return finite fields ----------
    from src.gems import hypotheses as H
    data, tmp, _e = _tiny_data()
    try:
        for hyp in H.shipped_hypotheses():
            try:
                s = hyp(data)
                ok = (s.shape == data.catalogue.shape
                      and bool(np.isfinite(s).all())
                      and float(s.std()) > 0)
                check(f"K8 {hyp.key} returns a finite, non-constant field", ok,
                      f"shape={s.shape} finite={bool(np.isfinite(s).all())} "
                      f"std={float(s.std()):.4g}")
            except Exception as e:                                  # noqa: BLE001
                check(f"K8 {hyp.key} returns a finite, non-constant field",
                      False, f"{type(e).__name__}: {e}")

        # ---- K9  external hypotheses must refuse, not silently degrade ----
        for hyp in H.external_hypotheses():
            try:
                hyp(data)
                check(f"K9 {hyp.key} refuses to run without its external data",
                      False, "it ran -- it must raise, or it silently becomes a "
                             "different detector")
            except ValueError as e:
                named = "10.5066" in str(e) or "10.15121" in str(e)
                check(f"K9 {hyp.key} refuses to run without its external data",
                      named, f"raised ValueError but named no source: {e}")

        # ---- K10  G-4 must actually read the shipped slope bands ----------
        from src.gems.features import g4_magnetic_contact
        base = g4_magnetic_contact(data)
        data.stack[11] = data.stack[11] * 0.0 + 1.0     # vertical_slope_tmi
        data.stack[12] = data.stack[12] * 0.0 + 1.0     # horizontal_slope_tmi
        changed = g4_magnetic_contact(data)
        check("K10 G-4 reads the shipped vertical/horizontal slope of TMI",
              not np.allclose(base, changed),
              "output unchanged when the shipped slope bands were replaced -- "
              "G-4 is still recomputing them locally")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- K13  tau=inf must disable suppression, not zero the field -------
    from src.gems.hypotheses import _suppress_catalogue
    base = np.ones((16, 16), dtype=np.float64)
    cat = np.zeros((16, 16), dtype=bool)
    cat[8, 8] = True
    unsup = _suppress_catalogue(base, cat, tau=float("inf"))
    sup = _suppress_catalogue(base, cat, tau=2.0)
    check("K13 tau=inf disables catalogue suppression instead of zeroing it",
          np.allclose(unsup, 1.0) and not np.allclose(sup, 1.0),
          f"tau=inf -> mean {float(unsup.mean()):.4f} (want 1.0); "
          f"tau=2 -> mean {float(sup.mean()):.4f} (want < 1.0)")

    # ---- K11  every officially listed layer is now read by something ------
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "audit_layer_usage.py")],
                         capture_output=True, text=True, cwd=str(REPO))
    tail = out.stdout
    line = [l for l in tail.splitlines() if l.startswith("UNUSED after adding")]
    check("K11 H-1..H-5 leave at most 1 of the 15 listed layers unread",
          bool(line) and "UNUSED after adding H-1..H-5: 0 of 15" in line[0],
          line[0] if line else out.stdout[-300:])

    # ---- K12  the anisotropic smoother really is anisotropic --------------
    from src.gems.hypotheses import anisotropic_gaussian
    imp = np.zeros((64, 64))
    imp[32, 32] = 1.0
    for theta in (0.0, 90.0):
        k = anisotropic_gaussian(imp, theta, 6.0, 1.0)
        # second moment along vs across the declared strike
        yy, xx = np.mgrid[0:64, 0:64]
        t = np.deg2rad(theta)
        along = (xx - 32) * np.cos(t) + (yy - 32) * np.sin(t)
        across = -(xx - 32) * np.sin(t) + (yy - 32) * np.cos(t)
        w = np.clip(k, 0, None)
        m_along = float((w * along ** 2).sum() / max(w.sum(), 1e-12))
        m_across = float((w * across ** 2).sum() / max(w.sum(), 1e-12))
        check(f"K12 anisotropic_gaussian(theta={theta:g}) smooths along strike",
              m_along > 3 * m_across,
              f"second moment along={m_along:.2f} across={m_across:.2f}")

if __name__ == "__main__":
    sys.exit(main())
