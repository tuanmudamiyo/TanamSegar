/* TanamSegar - in-browser soil forecast and recommendations.

   Runs the LSTM with TensorFlow.js, so there is no backend: the page reads
   recent /Logs from Firebase, this module turns them into the model's input
   window, predicts, and converts the forecast into advice.

   No Firebase in here on purpose. The same file is loaded by analyze.html and
   by the Node parity test, which checks that a window built here matches the
   one ml/prepare_dataset.py built for training. Any difference between the two
   is training/serving skew: the model receives inputs it was never trained on
   and returns confident nonsense, with no error anywhere.

   So every step below mirrors a named step in prepare_dataset.py. Change one,
   change the other, and re-run the parity test. */

/* ---- preprocessing constants, copied from ml/prepare_dataset.py ---- */

const RAW_CH = ['temperature', 'humidity', 'ldr', 'moisture', 'soilTemp',
                'ec', 'ph', 'nitrogen', 'phosphorus', 'potassium'];

// Physically possible range. Outside it is a bad RS485 frame, not a reading.
const BOUNDS = {
  temperature: [5, 55],   humidity: [5, 100],   ldr: [0, 4095],
  moisture: [0.5, 95],    soilTemp: [5, 55],    ec: [0.02, 12],
  ph: [3, 10.5],          nitrogen: [0.5, 900], phosphorus: [0.5, 900],
  potassium: [0.5, 900]
};

// Largest believable change per minute. Soil is buffered; it cannot jump.
const MAX_RATE_PER_MIN = {
  temperature: 1.2, humidity: 6, ldr: 4095, moisture: 4, soilTemp: 0.5,
  ec: 0.6, ph: 0.4, nitrogen: 12, phosphorus: 12, potassium: 25
};

const MAX_GAP_MIN = 40;          // longest gap prepare_dataset.py interpolates
const DELTA_CH = ['moisture', 'ec', 'nitrogen', 'potassium'];
const STALE_AFTER_SEC = 30 * 60; // no forecast from data this old

const svp = t => 0.6108 * Math.exp(17.27 * t / (t + 237.3));
const isNum = v => typeof v === 'number' && Number.isFinite(v);

/* ---- loading ---- */

export async function loadForecaster(tf, base = 'model/'){
  const contract = await (await fetch(base + 'contract.json', { cache: 'no-cache' })).json();
  const model = await tf.loadLayersModel(base + 'model.json');
  return { tf, model, contract };
}

/* ---- step 2: screen bad frames (prepare_dataset.screen) ----
   rows: [{ ts, m: {channel: number|null} }] sorted by ts. Mutates m. */
function screen(rows){
  for (const c of RAW_CH){
    const [lo, hi] = BOUNDS[c];
    for (const r of rows){
      const v = r.m[c];
      if (v !== null && !(v >= lo && v <= hi)) r.m[c] = null;
    }
  }
  // out and straight back again is a spike; a one-way jump may be a real
  // watering, so it is kept. Decide on the whole pass first, then apply.
  for (const c of RAW_CH){
    const limit = MAX_RATE_PER_MIN[c];
    const idx = [];
    rows.forEach((r, i) => { if (r.m[c] !== null) idx.push(i); });
    const kill = [];
    for (let j = 1; j < idx.length - 1; j++){
      const a = rows[idx[j - 1]], b = rows[idx[j]], n = rows[idx[j + 1]];
      const dtA = Math.max(1, Math.floor((b.ts - a.ts) / 60));
      const dtB = Math.max(1, Math.floor((n.ts - b.ts) / 60));
      const up = (b.m[c] - a.m[c]) / dtA;
      const dn = (n.m[c] - b.m[c]) / dtB;
      if (Math.abs(up) > limit && Math.abs(dn) > limit && up * dn < 0) kill.push(idx[j]);
    }
    for (const i of kill) rows[i].m[c] = null;
  }
}

/* ---- step 3: resample onto the grid (prepare_dataset.resample) ---- */
function resample(rows, stepSec, firstBin, lastBin){
  const sums = new Map();
  for (const r of rows){
    const b = Math.floor(r.ts / stepSec) * stepSec;
    if (b < firstBin || b > lastBin) continue;
    let s = sums.get(b);
    if (!s){ s = {}; for (const c of RAW_CH) s[c] = [0, 0]; sums.set(b, s); }
    for (const c of RAW_CH){
      const v = r.m[c];
      if (v !== null){ s[c][0] += v; s[c][1] += 1; }
    }
  }
  const grid = [];
  for (let t = firstBin; t <= lastBin; t += stepSec){
    const s = sums.get(t), rec = { ts: t };
    for (const c of RAW_CH) rec[c] = (s && s[c][1]) ? s[c][0] / s[c][1] : null;
    grid.push(rec);
  }
  return grid;
}

/* ---- step 4: interpolate short gaps (prepare_dataset.fill_gaps) ---- */
function fillGaps(grid, gridMin){
  const maxSteps = Math.max(1, Math.floor(MAX_GAP_MIN / gridMin));
  for (const c of RAW_CH){
    const idx = [];
    grid.forEach((r, i) => { if (r[c] !== null) idx.push(i); });
    for (let j = 0; j < idx.length - 1; j++){
      const a = idx[j], b = idx[j + 1], hole = b - a - 1;
      if (hole <= 0 || hole > maxSteps) continue;
      const va = grid[a][c], vb = grid[b][c];
      for (let k = 1; k <= hole; k++) grid[a + k][c] = va + (vb - va) * k / (hole + 1);
    }
  }
}

/* ---- step 5: engineered features (prepare_dataset.add_features) ---- */
function addFeatures(grid, contract){
  const per = contract.delta_steps, tz = contract.tz_offset_sec;
  grid.forEach((r, i) => {
    const t = r.temperature, h = r.humidity;
    r.vpd = (t !== null && h !== null) ? svp(t) * (1 - h / 100) : null;
    const frac = ((r.ts + tz) % 86400) / 86400;
    r.tod_sin = Math.sin(2 * Math.PI * frac);
    r.tod_cos = Math.cos(2 * Math.PI * frac);
    for (const c of DELTA_CH){
      const j = i - per;
      r['d_' + c] = (j >= 0 && r[c] !== null && grid[j][c] !== null) ? r[c] - grid[j][c] : null;
    }
  });
}

/* Builds the model's input window ending at the last complete grid bin.
   Returns { ok:false, reason } when an honest forecast is not possible -
   stale data, a gap, or too little history - rather than padding. */
export function buildWindow(records, contract, nowSec){
  const stepSec = contract.grid_min * 60;
  const L = contract.lookback;

  if (!records.length) return { ok: false, reason: 'No readings yet.' };
  const lastRec = records[records.length - 1].ts;
  if (nowSec - lastRec > STALE_AFTER_SEC){
    return { ok: false, reason: 'Last reading was ' + Math.round((nowSec - lastRec) / 60) +
      ' min ago - the forecast needs live data. Check the ESP32.' };
  }

  const lastBin = Math.floor(nowSec / stepSec) * stepSec - stepSec;  // last complete bin
  const winStart = lastBin - (L - 1) * stepSec;
  // extra bins in front: the deltas look back delta_steps, and interpolation
  // needs a known point on each side of a gap at the window's leading edge
  const margin = contract.delta_steps + Math.floor(MAX_GAP_MIN / contract.grid_min) + 2;
  const firstBin = winStart - margin * stepSec;

  // Not a gap - the history simply does not reach back far enough yet, which
  // is the normal state for the first half-day of a collection run.
  const needFrom = winStart - contract.delta_steps * stepSec;
  if (records[0].ts > needFrom){
    const needH = (lastBin + stepSec - needFrom) / 3600;
    const haveH = Math.max(0, (lastRec - records[0].ts) / 3600);
    return { ok: false, warming: true, reason: 'The first forecast needs ' + needH.toFixed(0) +
      ' h of readings - ' + haveH.toFixed(1) + ' h so far. Check back in about ' +
      Math.max(1, Math.round(needH - haveH)) + ' h.' };
  }

  const rows = records
    .filter(r => r.ts >= firstBin - 120 && r.ts < lastBin + stepSec + 1800)
    .map(r => ({ ts: r.ts, m: Object.fromEntries(RAW_CH.map(c => [c, isNum(r.m[c]) ? r.m[c] : null])) }));

  screen(rows);
  const grid = resample(rows, stepSec, firstBin, lastBin);
  fillGaps(grid, contract.grid_min);
  addFeatures(grid, contract);

  const win = grid.slice(grid.length - L);
  const F = contract.features, nf = F.length;
  const X = new Float32Array(L * nf);
  for (let i = 0; i < L; i++){
    for (let k = 0; k < nf; k++){
      const v = win[i][F[k]];
      if (v === null || !Number.isFinite(v)){
        const ago = Math.round((lastBin - win[i].ts) / 3600 * 10) / 10;
        // a whole-record outage is the ESP32 offline; one missing channel is a sensor
        const what = RAW_CH.every(c => win[i][c] === null) ? 'no readings' : F[k] + ' missing';
        return { ok: false, reason: 'Gap in the readings about ' + ago + ' h ago (' + what +
          '). The forecast needs ' + Math.round(L * contract.grid_min / 60) + ' h of continuous data.' };
      }
      X[i * nf + k] = v;
    }
  }

  const last = win[L - 1];
  const anchor = Object.fromEntries(contract.targets.map(t => [t, last[t]]));
  return { ok: true, X, anchor, last, lastBin, issuedAt: lastBin + stepSec };
}

/* Runs the model. Each output whose test score did not beat persistence
   reports the current value instead ("no change"), flagged as such - the
   page never shows a forecast that is known to be worse than doing nothing. */
export function predict(fc, records, nowSec){
  const { tf, model, contract } = fc;
  const w = buildWindow(records, contract, nowSec);
  if (!w.ok) return w;

  const L = contract.lookback, nf = contract.features.length;
  const Xs = new Float32Array(w.X.length);
  for (let i = 0; i < L; i++)
    for (let k = 0; k < nf; k++)
      Xs[i * nf + k] = (w.X[i * nf + k] - contract.x_mu[k]) / contract.x_sd[k];

  const raw = tf.tidy(() => model.predict(tf.tensor3d(Xs, [1, L, nf])).dataSync().slice());

  const nt = contract.targets.length;
  const out = {};
  contract.horizons_h.forEach((h, hi) => {
    contract.targets.forEach((t, ti) => {
      const j = hi * nt + ti;
      const name = contract.outputs[j];
      let v = raw[j] * contract.y_sd[j] + contract.y_mu[j];
      if (contract.residual) v += w.anchor[t];
      const sk = (contract.skill || {})[name];
      const trusted = !!(sk && sk.beats_persistence);
      (out[t] = out[t] || {})[h] = {
        value: trusted ? v : w.anchor[t],
        model: v,
        source: trusted ? 'model' : 'persistence',
        skill: sk ? sk.skill_pct : null
      };
    });
  });

  return { ok: true, issuedAt: w.issuedAt, current: w.last, anchor: w.anchor, forecast: out };
}

/* ---- recommendations ----

   Absolute thresholds only where they are meaningful for this hardware:
   pH and moisture. The probe's N/P/K "mg/kg" is inferred electrically and is
   not the quantity a lab reports, so published deficiency thresholds do not
   apply to it. Nutrient advice is therefore RELATIVE: the forecast's drop
   below this pot's own reference level (its training-period average, which
   on real data is the pot's own history), combined with the direction of
   travel. Revisit once a lab soil test gives a calibration anchor. */

export const ADVICE = {
  moisture: { refill: 30, critical: 20, wet: 50 },   // % as this probe reads the compost mix
  ph:       { low: 5.5, high: 6.8, act: 7.5 },       // eggplant 5.5-6.8
  ec:       { watch: 2.0, high: 2.5 },               // dS/m, provisional until the EC scaling is verified
  nutrient: { watch: 0.15, act: 0.30 },              // fractional drop below the pot's reference
  recentFertiliseSec: 3 * 86400
};

const LABEL = { nitrogen: 'Nitrogen', phosphorus: 'Phosphorus', potassium: 'Potassium' };

export function recommend(p, contract, opts = {}){
  const A = ADVICE, f = p.forecast, cur = p.current, out = [];
  const h6 = Math.min(...contract.horizons_h), h24 = Math.max(...contract.horizons_h);
  const fmt = (v, d = 0) => v.toFixed(d);

  /* irrigation - current reading and the short-horizon forecast */
  const mNow = cur.moisture, m6 = f.moisture[h6].value, m24 = f.moisture[h24].value;
  if (mNow < A.moisture.critical){
    out.push({ level: 'act', area: 'Irrigation', title: 'Water now',
      detail: 'Moisture is ' + fmt(mNow) + '%, below the ' + A.moisture.critical + '% stress point.' });
  } else if (m6 < A.moisture.refill){
    out.push({ level: 'act', area: 'Irrigation', title: 'Water within ' + h6 + ' hours',
      detail: 'Moisture is forecast to reach ' + fmt(m6) + '% (refill point ' + A.moisture.refill + '%).' });
  } else if (m24 < A.moisture.refill){
    out.push({ level: 'watch', area: 'Irrigation', title: 'Plan to water tomorrow morning',
      detail: 'Forecast ' + fmt(m24) + '% in ' + h24 + ' h. Morning watering loses less to evaporation.' });
  } else if (mNow > A.moisture.wet){
    out.push({ level: 'ok', area: 'Irrigation', title: 'Skip watering',
      detail: 'Moisture is ' + fmt(mNow) + '% - the mix is near field capacity.' });
  } else {
    out.push({ level: 'ok', area: 'Irrigation', title: 'No watering needed',
      detail: 'Forecast stays above ' + A.moisture.refill + '% for the next ' + h24 + ' h.' });
  }

  /* salinity - high EC overrides any fertiliser advice */
  const ecNow = cur.ec, ec24 = f.ec[h24].value, ecPeak = Math.max(ecNow, ec24);
  const saltHigh = ecPeak >= A.ec.high;
  if (saltHigh){
    out.push({ level: 'act', area: 'Salinity', title: 'Hold fertiliser and flush',
      detail: 'EC ' + fmt(ecNow, 2) + ' dS/m' + (ec24 > ecNow ? ', rising to ' + fmt(ec24, 2) : '') +
        '. Water through until it drains freely to leach salts.' });
  } else if (ecPeak >= A.ec.watch){
    out.push({ level: 'watch', area: 'Salinity', title: 'Salts building up',
      detail: 'EC ' + fmt(ecNow, 2) + ' dS/m. Go lighter on the next fertiliser dose.' });
  }

  /* nutrients - relative to the pot's own reference level */
  const recentFert = opts.lastFertiliseTs && (p.issuedAt - opts.lastFertiliseTs) < A.recentFertiliseSec;
  const low = [], watch = [], recovering = [];
  for (const t of ['nitrogen', 'phosphorus', 'potassium']){
    const ref = contract.x_mu[contract.features.indexOf(t)];
    const v24 = f[t][h24].value, drop = (ref - v24) / ref;
    const falling = v24 < cur[t];
    if (drop >= A.nutrient.act && falling) low.push([t, drop]);
    else if (drop >= A.nutrient.watch && falling) watch.push([t, drop]);
    else if (drop >= A.nutrient.watch) recovering.push([t, drop]);   // low, but on the way back up
  }
  const names = list => list.map(([t]) => LABEL[t]).join(', ');
  const worst = list => Math.round(Math.max(...list.map(([, d]) => d)) * 100);

  if (low.length && !saltHigh){
    if (recentFert){
      out.push({ level: 'watch', area: 'Fertiliser', title: 'Recently fertilised - give it time',
        detail: names(low) + ' still low, but a dose was logged in the last 3 days and is still dissolving.' });
    } else {
      const kOnly = low.length === 1 && low[0][0] === 'potassium';
      out.push({ level: 'act', area: 'Fertiliser',
        title: kOnly ? 'Apply a potassium-rich feed' : 'Apply balanced fertiliser',
        detail: names(low) + ' forecast ' + worst(low) + '% below this pot\'s usual level and still falling.' });
    }
  } else if ((low.length || watch.length) && !saltHigh){
    const all = low.concat(watch);
    out.push({ level: 'watch', area: 'Fertiliser', title: names(all) + ' trending down',
      detail: 'Forecast ' + worst(all) + '% below usual within ' + h24 + ' h. No dose needed yet.' });
  } else if (recovering.length && !saltHigh){
    out.push({ level: 'ok', area: 'Fertiliser', title: names(recovering) + ' recovering',
      detail: 'Still ' + worst(recovering) + '% below usual but rising over the next ' + h24 + ' h. No dose needed.' });
  } else if (!saltHigh){
    out.push({ level: 'ok', area: 'Fertiliser', title: 'Nutrients holding',
      detail: 'N, P and K stay near this pot\'s usual levels over the next ' + h24 + ' h.' });
  }

  /* pH - current reading only; it moves less than the probe's 0.1 step over
     these horizons, so it is not forecast */
  const ph = cur.ph;
  if (ph > A.ph.act){
    out.push({ level: 'act', area: 'pH', title: 'Soil too alkaline',
      detail: 'pH ' + fmt(ph, 1) + '. Eggplant prefers ' + A.ph.low + '-' + A.ph.high + '; consider elemental sulphur.' });
  } else if (ph > A.ph.high){
    out.push({ level: 'watch', area: 'pH', title: 'pH slightly high',
      detail: 'pH ' + fmt(ph, 1) + ', just above the ' + A.ph.low + '-' + A.ph.high + ' range. No action unless it keeps rising.' });
  } else if (ph < A.ph.low){
    out.push({ level: 'act', area: 'pH', title: 'Soil too acidic',
      detail: 'pH ' + fmt(ph, 1) + '. Add agricultural lime to bring it toward ' + A.ph.low + '-' + A.ph.high + '.' });
  }

  const rank = { act: 0, watch: 1, ok: 2 };
  return out.sort((a, b) => rank[a.level] - rank[b.level]);
}
