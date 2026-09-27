/**
 * GEMSDOE9 — browser GeoTIFF + ZIP writer, zero dependencies.
 *
 * WHY THIS FILE WAS REWRITTEN (2026-09-27)
 * -----------------------------------------
 * The previous `docs/geotiff_writer.js` emitted a structurally invalid TIFF:
 * the StripOffsets (tag 273) and StripByteCounts (tag 279) values were marked
 * "inline" before their real arrays existed, so only the first 4 bytes of each
 * array ever reached the file. Strips 2..N pointed at offset 0. GDAL/rasterio
 * could not read the result at all:
 *
 *     RasterioIOError: TIFFReadEncodedStrip() failed.
 *
 * Every .tif produced by the "Build submission.tif" button was therefore
 * unreadable, which is the direct cause of the DrivenData form rejecting a
 * download from this site with "Predicted values must be in range [0, 1]".
 * Reproduced and fixed here; see tests/test_geotiff_writer.py.
 *
 * WHAT THIS WRITER GUARANTEES
 * ---------------------------
 * 1. A structurally valid classic (non-BigTIFF) little-endian TIFF, single band,
 *    SampleFormat=IEEE float (32-bit), width 3292, height 3730, EPSG:32611.
 * 2. All 12,279,160 pixel values are FINITE and inside [0, 1]. The writer
 *    refuses to emit NaN or Inf. This is deliberate: the competition spec says
 *    "data outside the bounds is null or nan", but NaN is not in [0, 1] and the
 *    platform's range check has already been observed to reject a file that
 *    contained it. Writing 0.0 outside the footprint satisfies "null" in the
 *    scorer's eyes (a prediction of "no fault") and can never trip a range
 *    validator, so it is the only variant we publish.
 * 3. A real self-check: after writing, the bytes are parsed back from scratch
 *    (IFD walk -> strip offsets -> inflate -> float decode) and compared to the
 *    input array element-for-element. The download is only offered if it passes.
 *
 * Compressed with CompressionStream('deflate') when the browser has it
 * (TIFF Compression=8, "Adobe Deflate"); otherwise uncompressed strips
 * (Compression=1). Both are read by GDAL/libtiff.
 */

(function (global) {
  'use strict';

  // ---------------------------------------------------------------- constants
  const WIDTH = 3292;
  const HEIGHT = 3730;
  const EPSG = 32611;
  const ORIGIN_X = 243350.0;
  const ORIGIN_Y = 4508550.0;
  const RES = 100.0;

  const T_BYTE = 1, T_ASCII = 2, T_SHORT = 3, T_LONG = 4,
        T_RATIONAL = 5, T_FLOAT = 11, T_DOUBLE = 12;
  const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8 };

  // ------------------------------------------------------------------ helpers
  function align4(n) { return (n + 3) & ~3; }
  function u8(v) { return new Uint8Array([v & 0xff]); }
  function u16(v) { const b = new Uint8Array(2); new DataView(b.buffer).setUint16(0, v, true); return b; }
  function u32(v) { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); return b; }
  function f64arr(a) {
    const b = new Uint8Array(a.length * 8);
    const dv = new DataView(b.buffer);
    for (let i = 0; i < a.length; i++) dv.setFloat64(i * 8, a[i], true);
    return b;
  }
  function u16arr(a) {
    const b = new Uint8Array(a.length * 2);
    const dv = new DataView(b.buffer);
    for (let i = 0; i < a.length; i++) dv.setUint16(i * 2, a[i], true);
    return b;
  }
  function u32arr(a) {
    const b = new Uint8Array(a.length * 4);
    const dv = new DataView(b.buffer);
    for (let i = 0; i < a.length; i++) dv.setUint32(i * 4, a[i], true);
    return b;
  }
  function ascii(str) {
    const b = new Uint8Array(str.length + 1); // trailing NUL is the terminator
    for (let i = 0; i < str.length; i++) b[i] = str.charCodeAt(i) & 0xff;
    return b;
  }
  function cat(list) {
    let n = 0;
    for (const a of list) n += a.length;
    const out = new Uint8Array(n);
    let p = 0;
    for (const a of list) { out.set(a, p); p += a.length; }
    return out;
  }
  function bytesOf(f32) {
    return new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  }

  // ------------------------------------------------------------------ inflate
  async function deflate(u8in) {
    if (typeof global.CompressionStream !== 'function') return null;
    try {
      const cs = new global.CompressionStream('deflate'); // zlib-wrapped
      const w = cs.writable.getWriter();
      w.write(u8in); w.close();
      return new Uint8Array(await new Response(cs.readable).arrayBuffer());
    } catch (e) { return null; }
  }

  /** Inflate zlib-wrapped data (used only by the self-check parser). */
  async function inflate(u8in) {
    if (typeof global.DecompressionStream !== 'function') return null;
    const ds = new global.DecompressionStream('deflate');
    const w = ds.writable.getWriter();
    w.write(u8in); w.close();
    return new Uint8Array(await new Response(ds.readable).arrayBuffer());
  }

  // --------------------------------------------------------------- TIFF write
  /**
   * @param {Float32Array} field  width*height values, all finite, all in [0,1]
   * @param {object} opts {width, height, rowsPerStrip, useCompression}
   * @returns {Promise<{bytes:Uint8Array, width, height, nStrips, compression, totalBytes}>}
   */
  async function writeGeoTIFF(field, opts) {
    opts = opts || {};
    const width = opts.width || WIDTH;
    const height = opts.height || HEIGHT;
    const rowsPerStrip = opts.rowsPerStrip || 128;
    const allowCompress = opts.useCompression !== false;

    if (!(field instanceof Float32Array)) throw new Error('field must be a Float32Array');
    if (field.length !== width * height) {
      throw new Error(`field length ${field.length} != ${width}x${height}=${width * height}`);
    }

    // ---- hard gate: no NaN, no Inf, nothing outside [0,1] -------------------
    let bad = 0, minV = Infinity, maxV = -Infinity, nPos = 0;
    for (let i = 0; i < field.length; i++) {
      const v = field[i];
      if (!Number.isFinite(v) || v < 0 || v > 1) bad++;
      else { if (v < minV) minV = v; if (v > maxV) maxV = v; if (v > 0) nPos++; }
    }
    if (bad > 0) {
      throw new Error(
        `refusing to write: ${bad} pixel(s) are NaN/Inf/outside[0,1]. ` +
        `This is exactly what makes the submission form say ` +
        `"Predicted values must be in range [0, 1]". Clamp first.`);
    }

    // ---- strips -------------------------------------------------------------
    const raw = bytesOf(field);
    const nStrips = Math.ceil(height / rowsPerStrip);
    const strips = [];
    const compressed = [];
    for (let s = 0; s < nStrips; s++) {
      const rows = Math.min(rowsPerStrip, height - s * rowsPerStrip);
      const start = s * rowsPerStrip * width * 4;
      const chunk = raw.subarray(start, start + rows * width * 4);
      let payload = chunk;
      if (allowCompress) {
        const z = await deflate(chunk);
        if (z && z.length < chunk.length) payload = z;
      }
      strips.push({ bytes: payload, rawLen: chunk.length });
      compressed.push(payload !== chunk);
    }
    const anyCompressed = compressed.some(Boolean);
    const compression = anyCompressed ? 8 : 1; // 8 = Adobe Deflate

    // ---- IFD entries --------------------------------------------------------
    const geoKeys = [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, EPSG];
    const gdalMeta = '  <GDALMetadata>\n    <Item name="AREA_OR_POINT">Area</Item>\n  </GDALMetadata>\n';

    const entries = [
      { tag: 256, type: T_LONG,   count: 1,  data: u32(width) },
      { tag: 257, type: T_LONG,   count: 1,  data: u32(height) },
      { tag: 258, type: T_SHORT,  count: 1,  data: u16(32) },        // BitsPerSample
      { tag: 259, type: T_SHORT,  count: 1,  data: u16(compression) },// Compression
      { tag: 262, type: T_SHORT,  count: 1,  data: u16(1) },         // Photometric = MinIsBlack
      { tag: 273, type: T_LONG,   count: nStrips, data: null, name: 'StripOffsets' },
      { tag: 277, type: T_SHORT,  count: 1,  data: u16(1) },         // SamplesPerPixel
      { tag: 278, type: T_LONG,   count: 1,  data: u32(rowsPerStrip) },
      { tag: 279, type: T_LONG,   count: nStrips, data: null, name: 'StripByteCounts' },
      { tag: 284, type: T_SHORT,  count: 1,  data: u16(1) },         // PlanarConfiguration
      { tag: 305, type: T_SHORT,  count: nStrips, data: u16arr(
          Array.from({ length: nStrips }, () => 1)) },                // SampleFormat=IEEE float
      { tag: 339, type: T_SHORT,  count: 1,  data: u16(3) },         // SampleFormat = 3 (IEEE fp)
      { tag: 33550, type: T_DOUBLE, count: 3, data: f64arr([RES, RES, 0]) },
      { tag: 33922, type: T_DOUBLE, count: 6, data: f64arr([0, 0, 0, ORIGIN_X, ORIGIN_Y, 0]) },
      { tag: 34735, type: T_SHORT, count: geoKeys.length, data: u16arr(geoKeys) },
      { tag: 42112, type: T_ASCII, count: gdalMeta.length + 1, data: ascii(gdalMeta) }, // GDAL_METADATA
    ];
    // Tag 42113 (GDAL_NODATA) is deliberately NOT written: we publish all-finite
    // files with no nodata declaration, so no reader can ever reinterpret a pixel
    // as nodata and substitute a value outside [0, 1].

    // ---- layout -------------------------------------------------------------
    const ifdStart = 8;
    const ifdLen = 2 + entries.length * 12 + 4;
    let cursor = align4(ifdStart + ifdLen);

    // Size must come from the *declared* type+count, not from `data`, because
    // StripOffsets / StripByteCounts are laid out before their arrays exist.
    for (const e of entries) {
      const size = (TYPE_SIZE[e.type] || 1) * e.count;
      e.external = size > 4;
      if (e.external) { e.offset = cursor; cursor = align4(cursor + size); }
    }
    // strip payloads follow the IFD extras
    const offs = new Array(nStrips);
    const lens = new Array(nStrips);
    const payloads = new Array(nStrips);
    for (let s = 0; s < nStrips; s++) {
      offs[s] = cursor;
      lens[s] = strips[s].bytes.length;
      payloads[s] = strips[s].bytes;
      cursor = align4(cursor + lens[s]);
    }
    const total = cursor;

    // Fill in the two deferred arrays now that offsets are known.
    entries.find(e => e.name === 'StripOffsets').data = u32arr(offs);
    entries.find(e => e.name === 'StripByteCounts').data = u32arr(lens);

    // Sanity: every entry that says "external" must now have a data array.
    for (const e of entries) {
      if (e.external && (!e.data || e.data.length === 0)) {
        throw new Error(`internal layout error: tag ${e.tag} marked external but has no data`);
      }
    }

    const out = new Uint8Array(total);
    const dv = new DataView(out.buffer);
    out[0] = 0x49; out[1] = 0x49;            // 'II'
    dv.setUint16(2, 42, true);               // classic TIFF
    dv.setUint32(4, ifdStart, true);
    dv.setUint16(ifdStart, entries.length, true);
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      const p = ifdStart + 2 + i * 12;
      dv.setUint16(p, e.tag, true);
      dv.setUint16(p + 2, e.type, true);
      dv.setUint32(p + 4, e.count, true);
      if (e.external) {
        dv.setUint32(p + 8, e.offset, true);
        out.set(e.data, e.offset);
      } else {
        out.set(e.data.subarray(0, 4), p + 8);
      }
    }
    dv.setUint32(ifdStart + 2 + entries.length * 12, 0, true); // next IFD = 0
    for (let s = 0; s < nStrips; s++) out.set(payloads[s], offs[s]);

    return {
      bytes: out, width, height, nStrips, compression,
      rowsPerStrip, totalBytes: total,
      stats: { min: minV, max: maxV, nPositive: nPos, nTotal: field.length }
    };
  }

  // ----------------------------------------------------------- TIFF read-back
  /**
   * Independent parser: walks the IFD of `bytes`, reassembles the raster and
   * returns {ok, reason, data}. Deliberately shares no code with the writer.
   */
  async function readGeoTIFF(bytes) {
    try {
      const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      if (dv.getUint16(0, true) !== 0x4949) return { ok: false, reason: 'not little-endian II' };
      if (dv.getUint16(2, true) !== 42) return { ok: false, reason: 'bad magic (expected 42)' };
      const ifdStart = dv.getUint32(4, true);
      const nEntries = dv.getUint16(ifdStart, true);
      const tags = {};
      for (let i = 0; i < nEntries; i++) {
        const p = ifdStart + 2 + i * 12;
        const tag = dv.getUint16(p, true);
        const type = dv.getUint16(p + 2, true);
        const count = dv.getUint32(p + 4, true);
        const size = (TYPE_SIZE[type] || 1) * count;
        let data;
        if (size <= 4) {
          data = bytes.subarray(p + 8, p + 8 + size);
        } else {
          const at = dv.getUint32(p + 8, true);
          if (at + size > bytes.length) return { ok: false, reason: `tag ${tag} points outside file` };
          data = bytes.subarray(at, at + size);
        }
        tags[tag] = { type, count, data };
      }
      const readNum = (tag, dv2) => {
        const t = tags[tag]; if (!t) return null;
        if (t.type === T_SHORT) return dv2.getUint16(0, true);
        if (t.type === T_LONG) return dv2.getUint32(0, true);
        return null;
      };
      const width = readNum(256, new DataView(tags[256].data.buffer, tags[256].data.byteOffset, tags[256].data.length));
      const height = readNum(257, new DataView(tags[257].data.buffer, tags[257].data.byteOffset, tags[257].data.length));
      const compression = readNum(259, new DataView(tags[259].data.buffer, tags[259].data.byteOffset, tags[259].data.length));
      const rowsPerStrip = readNum(278, new DataView(tags[278].data.buffer, tags[278].data.byteOffset, tags[278].data.length));
      const sampleFormat = readNum(339, new DataView(tags[339].data.buffer, tags[339].data.byteOffset, tags[339].data.length));
      if (width === null || height === null) return { ok: false, reason: 'missing ImageWidth/ImageLength' };
      if (sampleFormat !== 3) return { ok: false, reason: `SampleFormat=${sampleFormat}, expected 3 (IEEE float)` };
      const offs = new DataView(tags[273].data.buffer, tags[273].data.byteOffset, tags[273].data.byteLength);
      const lens = new DataView(tags[279].data.buffer, tags[279].data.byteOffset, tags[279].data.byteLength);
      const nStrips = offs.byteLength / 4;
      const outBytes = new Uint8Array(width * height * 4);
      for (let s = 0; s < nStrips; s++) {
        const off = offs.getUint32(s * 4, true);
        const len = lens.getUint32(s * 4, true);
        if (off + len > bytes.length) return { ok: false, reason: `strip ${s} outside file (off=${off} len=${len} size=${bytes.length})` };
        let payload = bytes.subarray(off, off + len);
        if (compression === 8) {
          const raw = await inflate(payload);
          if (!raw) return { ok: false, reason: 'DecompressionStream unavailable to verify compressed strips' };
          payload = raw;
        }
        const rows = Math.min(rowsPerStrip, height - s * rowsPerStrip);
        const n = rows * width * 4;
        if (payload.length < n) return { ok: false, reason: `strip ${s} too short (${payload.length} < ${n})` };
        outBytes.set(payload.subarray(0, n), s * rowsPerStrip * width * 4);
      }
      return { ok: true, data: new Float32Array(outBytes.buffer, outBytes.byteOffset, outBytes.length / 4),
               width, height, compression, nStrips };
    } catch (e) {
      return { ok: false, reason: 'parser exception: ' + e.message };
    }
  }

  /** Full round-trip guarantee: write -> parse -> compare -> report. */
  async function writeAndVerify(field, opts) {
    const res = await writeGeoTIFF(field, opts);
    const back = await readGeoTIFF(res.bytes);
    if (!back.ok) return { ok: false, reason: 'read-back failed: ' + back.reason, res };
    if (back.width !== res.width || back.height !== res.height) {
      return { ok: false, reason: `read-back shape ${back.width}x${back.height} != ${res.width}x${res.height}`, res };
    }
    const a = back.data, b = field;
    let maxAbsDiff = 0, mismatches = 0;
    for (let i = 0; i < b.length; i++) {
      const d = Math.abs(a[i] - b[i]);
      if (d > maxAbsDiff) maxAbsDiff = d;
      if (d > 0) mismatches++;
    }
    if (maxAbsDiff !== 0) {
      return { ok: false, reason: `round-trip mismatch: ${mismatches} px differ, max |diff| = ${maxAbsDiff}`, res };
    }
    let nNaN = 0;
    for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) nNaN++;
    if (nNaN > 0) return { ok: false, reason: `${nNaN} non-finite values after read-back`, res };
    return { ok: true, res, back, maxAbsDiff: 0 };
  }

  // ---------------------------------------------------------------- sha / zip
  async function sha256Hex(bytes) {
    if (global.crypto && global.crypto.subtle) {
      const d = await global.crypto.subtle.digest('SHA-256', bytes);
      const o = new Uint8Array(d);
      let s = '';
      for (let i = 0; i < o.length; i++) s += o[i].toString(16).padStart(2, '0');
      return s;
    }
    throw new Error('WebCrypto SHA-256 unavailable — refusing to name the file without a real hash');
  }

  const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
      t[n] = c >>> 0;
    }
    return t;
  })();
  function crc32(u8) {
    let c = 0xFFFFFFFF;
    for (let i = 0; i < u8.length; i++) c = CRC_TABLE[(c ^ u8[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
  }
  async function deflateRaw(u8in) {
    if (typeof global.CompressionStream !== 'function') return null;
    try {
      const cs = new global.CompressionStream('deflate-raw');
      const w = cs.writable.getWriter();
      w.write(u8in); w.close();
      return new Uint8Array(await new Response(cs.readable).arrayBuffer());
    } catch (e) { return null; }
  }

  /** Minimal but spec-correct ZIP containing one file. */
  async function makeZip(entryName, data) {
    const nameBytes = new Uint8Array(entryName.length);
    for (let i = 0; i < entryName.length; i++) nameBytes[i] = entryName.charCodeAt(i) & 0xff;
    const crc = crc32(data);
    let method = 0, payload = data;
    const z = await deflateRaw(data);
    if (z && z.length < data.length) { method = 8; payload = z; }
    // DOS time/date: fixed 1980-01-01 00:00:00 keeps the build deterministic.
    const dosTime = 0, dosDate = 33;
    const local = cat([
      u32(0x04034b50), u16(20), u16(0), u16(method), u16(dosTime), u16(dosDate),
      u32(crc), u32(payload.length), u32(data.length),
      u16(nameBytes.length), u16(0), nameBytes, payload
    ]);
    const central = cat([
      u32(0x02014b50), u16(20), u16(20), u16(0), u16(method), u16(dosTime), u16(dosDate),
      u32(crc), u32(payload.length), u32(data.length),
      u16(nameBytes.length), u16(0), u16(0), u16(0), u16(0), u32(0), u32(0), nameBytes
    ]);
    const eocd = cat([
      u32(0x06054b50), u16(0), u16(0), u16(1), u16(1),
      u32(central.length), u32(local.length), u16(0)
    ]);
    return { bytes: cat([local, central, eocd]), method, crc };
  }

  // ------------------------------------------------------------- field making
  /**
   * HONESTY NOTE — READ BEFORE USING
   * ---------------------------------
   * This builds a *geometric placeholder*, not a trained model. It exists so
   * that the format/IO/upload path can be exercised end-to-end without the
   * competition rasters (which need a DrivenData login to download). It uses no
   * GeoDAWN data whatsoever: it is a fixed-seed oriented band pattern. Treat any
   * score it produces as meaningless.
   *
   * The real predictor is built by scripts/build_submission.py from
   * data/processed/prediction.npy, which scripts/train_model.py produces once
   * data/ is populated. See docs/executive_summary.html.
   */
  function buildPlaceholderField(width, height, opts) {
    opts = opts || {};
    const frac = opts.fraction || 0.028;
    const seed = opts.seed === undefined ? 20260927 : opts.seed;
    const n = width * height;
    const out = new Float32Array(n);

    // Deterministic xorshift32 so every browser build is byte-identical.
    let s = seed >>> 0 || 1;
    const rnd = () => {
      s ^= s << 13; s >>>= 0;
      s ^= s >>> 17;
      s ^= s << 5;  s >>>= 0;
      return s / 4294967296;
    };
    // Two crossing oriented band fields at 30 deg and 120 deg (Walker-Lane-like).
    const f1 = 14, f2 = 11;
    const c1x = Math.cos(Math.PI / 6), c1y = Math.sin(Math.PI / 6);
    const c2x = Math.cos(2 * Math.PI / 3), c2y = Math.sin(2 * Math.PI / 3);
    const score = new Float32Array(n);
    for (let y = 0, i = 0; y < height; y++) {
      const v = y / height;
      for (let x = 0; x < width; x++, i++) {
        const u = x / width;
        const a = Math.sin(2 * Math.PI * f1 * (u * c1x + v * c1y));
        const b = Math.sin(2 * Math.PI * f2 * (u * c2x + v * c2y));
        score[i] = 0.45 * a * b + 0.35 * a + 0.35 * b + 0.30 * rnd();
      }
    }
    // Exact top-k threshold via a copy + sort (deterministic, no tie jitter).
    const copy = Float32Array.from(score);
    copy.sort();
    const k = Math.max(1, Math.min(n, Math.floor(n * frac)));
    const thresh = copy[n - k];
    let ones = 0;
    for (let i = 0; i < n; i++) { if (score[i] >= thresh) { out[i] = 1.0; ones++; } }
    out.nOnes = ones;
    return out;
  }

  // ------------------------------------------------------------------ exports
  const api = {
    writeGeoTIFF, readGeoTIFF, writeAndVerify, sha256Hex, makeZip,
    buildPlaceholderField, crc32,
    WIDTH, HEIGHT, EPSG, ORIGIN_X, ORIGIN_Y, RES
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.Gems9Writer = api;
})(typeof window !== 'undefined' ? window : globalThis);
