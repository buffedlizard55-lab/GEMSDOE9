#!/usr/bin/env bash
# Public, user-supplied mirror via a sibling GitHub repo. No DrivenData auth.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
repo=buffedlizard55-lab/5GEMSDOE
base=data/bridge
# SHA-256 from sibling bridge manifest, independently checked on 2026-09-26.
names=(example_submission.tif existing_faults.tif)
shas=(2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc 7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093)
outs=(sample_submission.tif labels.tif)
for i in 0 1; do
  dest="data/${outs[i]}"
  if ! echo "${shas[i]}  $dest" | sha256sum -c --status 2>/dev/null; then
    gh api -H 'Accept: application/vnd.github.raw' "repos/$repo/contents/$base/${names[i]}" > "$dest.tmp"
    echo "${shas[i]}  $dest.tmp" | sha256sum -c
    mv "$dest.tmp" "$dest"
  fi
done
whole=data/training_features.tif
if ! echo "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5  $whole" | sha256sum -c --status 2>/dev/null; then
  parts=(0a330f8951af6c921029e25c84a579319d2db554d62d30d894d6ddc97f98cff7 3c98037b2c997e3bbfcdfc2d9a982e8b820a77410dd7404b05cdb80594922c50 c375c4dbc40c59bbaece572b5e348700b75935f9a30c879c82b6417e0836f31c b164159e6d0cb2595bc9f63a948af2646b124114a9c5921880c7516092137320 fa0a6f9c936fac1d6f20ca37f5929b2d60bf7a80f3d477dcab886f941aee2696)
  for i in 0 1 2 3 4; do
    printf -v n '%03d' "$i"
    part="data/part-$n"
    if ! echo "${parts[i]}  $part" | sha256sum -c --status 2>/dev/null; then
      gh api -H 'Accept: application/vnd.github.raw' "repos/$repo/contents/$base/gems-geodawn-numerical-features.tif.part-$n" > "$part.tmp"
      echo "${parts[i]}  $part.tmp" | sha256sum -c
      mv "$part.tmp" "$part"
    fi
  done
  cat data/part-0* > "$whole.tmp"
  echo "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5  $whole.tmp" | sha256sum -c
  mv "$whole.tmp" "$whole"
fi
echo 'All three mirror SHA-256 pins verified. Compare official source separately if authenticated.'
