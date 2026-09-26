#!/usr/bin/env python3
"""Generate reproducible TIFF band inventory, no geological interpretation inferred."""
import csv
import hashlib
from html import escape
from pathlib import Path
import rasterio

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'training_features.tif'
DEST = ROOT / 'docs' / 'data' / 'feature_inventory.csv'
PIN = '4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5'
OFFICIAL = 'https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#provided-features'
MIRROR = 'https://github.com/buffedlizard55-lab/5GEMSDOE/blob/main/data/bridge/manifest.json'


def main():
    h = hashlib.sha256()
    with SOURCE.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    if h.hexdigest() != PIN: raise ValueError('input SHA mismatch')
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(SOURCE) as src, DEST.open('w', newline='') as out:
        if src.count != 19 or src.crs.to_epsg() != 32611: raise ValueError('unexpected feature grid')
        w = csv.writer(out)
        w.writerow(['band_1_based', 'tiff_band_name_tag', 'tiff_description', 'grid', 'input_sha256', 'official_feature_families', 'mirror_manifest', 'review_flag'])
        for i in range(1, src.count + 1):
            name = src.tags(i).get('band_name', '')
            desc = src.descriptions[i-1] or ''
            flag = 'Problem text describes a depth to conductive base, TIFF tag describes basement depth; not synonymous' if i == 15 else ('Sibling audit questions tc/tilt tag; identity NOT independently confirmed in G9' if i == 6 else '')
            w.writerow([i, name, desc, f'{src.width}x{src.height} EPSG:{src.crs.to_epsg()} 100m', PIN, OFFICIAL, MIRROR, flag])
    # The HTML table is generated from the verified CSV, never hand-copied.
    with DEST.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    cells = ''.join('<tr><td>'+escape(r['band_1_based'])+'</td><td><code>'+escape(r['tiff_band_name_tag'])+'</code></td><td>'+escape(r['tiff_description'])+'</td><td>'+escape(r['review_flag'] or 'No tag discrepancy recorded')+'</td></tr>' for r in rows)
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>19 verified feature tags · GEMSDOE9</title><link rel="stylesheet" href="style.css"></head><body><header><nav><a class="logo" href="index.html">GEMS / 09</a><div class="links"><a href="index.html">Decision</a><a href="executive_summary.html">How to submit</a><a href="research.html">Research</a><a href="sources.html">Sources</a></div></nav></header><main><p class="eyebrow">MEASURED FROM TIFF / NOT A SCIENTIFIC INTERPRETATION</p><h1>Every provided feature tag, auditable.</h1><p class="lead">19 band descriptions read directly from SHA-256-pinned competition-derived raster <code>4371c82e…</code>. The <a href="data/feature_inventory.csv">CSV</a> records exact names, source links and flags per row. <a href="https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#provided-features">Official feature-family description ↗</a> · <a href="https://github.com/buffedlizard55-lab/5GEMSDOE/blob/main/data/bridge/manifest.json">Mirror manifest ↗</a>. Publisher did not publish a per-band checksum here. A tag is not independent proof of physical identity.</p><div class="scroll"><table><thead><tr><th>Band</th><th>TIFF tag</th><th>TIFF description (verbatim)</th><th>Review flag</th></tr></thead><tbody>''' + cells + '''</tbody></table></div></main><footer><a href="sources.html">Source ledger →</a></footer></body></html>'''
    (ROOT / 'docs' / 'features.html').write_text(html + '\n')
    print(DEST)

if __name__ == '__main__': main()
