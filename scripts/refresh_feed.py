#!/usr/bin/env python3
"""Fail-closed official leaderboard snapshot refresh. Never estimates new scores.

GitHub runner egress required; on fetch/parse failure retains previous successful
observation, writes error metadata and still exits 0 so the site displays staleness.
"""
import datetime as dt
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.request

URL = 'https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/'
DEST = Path(__file__).resolve().parents[1] / 'docs' / 'data' / 'feed.json'

class Rows(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows = []; self.current = None; self.in_cell = 0
    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.current = []
        elif tag in ('td', 'th') and self.current is not None:
            self.current.append(''); self.in_cell += 1
    def handle_data(self, data):
        if self.current is not None and self.in_cell:
            self.current[-1] += data
    def handle_endtag(self, tag):
        if tag in ('td', 'th'): self.in_cell = max(0, self.in_cell - 1)
        elif tag == 'tr' and self.current is not None:
            self.rows.append([re.sub(r'\s+', ' ', c).strip() for c in self.current]); self.current = None; self.in_cell = 0

def extract(html):
    parser = Rows(); parser.feed(html)
    for row in parser.rows:
        # Explicit rank + numeric final score, not an arbitrary number elsewhere on page.
        if len(row) >= 4 and row[0] in ('#1', '1', '1.'):
            # Official columns: rank, team members, participant, public score, shared work.
            score = row[3]
            if re.fullmatch(r'0\.\d{4}', score) and row[2]:
                # The HTML page may include a relative timestamp; keep the
                # source row rather than pretending the display name is parsed.
                return {'rank': 1, 'participant_row': row[2][:180], 'public_dti': float(score)}
    raise ValueError('Leaderboard row parser did not find unambiguous #1 and numeric score; retaining last successful score')

def refresh():
    previous = json.loads(DEST.read_text()) if DEST.exists() else {}
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    try:
        req = urllib.request.Request(URL, headers={'User-Agent': 'GEMSDOE9-public-source-audit/1.0'})
        with urllib.request.urlopen(req, timeout=20) as response:
            if response.geturl().rstrip('/') != URL.rstrip('/'):
                raise ValueError('leaderboard redirect: auth/other response')
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000: raise ValueError('oversize page')
        top = extract(raw.decode('utf-8'))
        output = {'source': URL, 'last_success_utc': stamp, 'top': top, 'status': 'ok'}
    except (OSError, UnicodeError, ValueError) as exc:
        output = previous.copy()
        output['status'] = 'stale: automatic refresh failed'
        output['last_attempt_utc'] = stamp
        output['error'] = str(exc)[:240]
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print(output['status'], output.get('top'), output.get('error', ''))
    return output

if __name__ == '__main__': refresh()
