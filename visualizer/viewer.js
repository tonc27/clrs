// Extracted from viewer.html for easier maintenance.
// Requires: pca.js (defines global PCA)

(() => {
  'use strict';

  function init() {
    // Selection + caches (used across UI and rendering)
    const selectedState = new Map();
    const scaleCache = new Map();
    // Cache for node-wise PCA basis so we don't recompute when scrubbing time
    const nodePcaCache = { key: null, pca: null };

    // Toggle state for hint correctness graphs (per probe name)
    const correctnessState = new Map();

    // Palette for categorical visualizations
    const CAT_PALETTE = [
      '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948',
      '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC',
      '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B',
      '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF'
    ];

    const elFile = document.getElementById('file');
    const elBidx = document.getElementById('bidx');
    const elFps  = document.getElementById('fps');
    const elT    = document.getElementById('t');
    const elTLabel = document.getElementById('tLabel');
    const elVCell = document.getElementById('vcell');
    const elMCell = document.getElementById('mcell');
    const elR0 = document.getElementById('r0');
    const elR1 = document.getElementById('r1');
    const elC0 = document.getElementById('c0');
    const elC1 = document.getElementById('c1');
    const elContrast = document.getElementById('contrast');
    const elContrastLabel = document.getElementById('contrastLabel');
    const elLockScale = document.getElementById('lockScale');
    const elMeta = document.getElementById('meta');
    const elTopMeta = document.getElementById('topMeta');
    const elAlgoName = document.getElementById('algoName');
    const elSummary = document.getElementById('summary');
    const elHoverText = document.getElementById('hoverText');

    // --- PCA has been removed from the UI (viewer.html). Keep the viewer working by making PCA optional.
    const HAS_PCA_UI = !!document.getElementById('showPca');

    // PCA controls (optional)
    const elShowPca = HAS_PCA_UI ? document.getElementById('showPca') : null;
    const elPcx = HAS_PCA_UI ? document.getElementById('pcx') : null;
    const elPcy = HAS_PCA_UI ? document.getElementById('pcy') : null;
    const elPcaInfo = HAS_PCA_UI ? document.getElementById('pcaInfo') : null;
    const elPcaContainer = HAS_PCA_UI ? document.getElementById('pcaContainer') : null;
    const elPcaCanvas = HAS_PCA_UI ? document.getElementById('pcaCanvas') : null;
    const elPcaTitle = HAS_PCA_UI ? document.getElementById('pcaTitle') : null;
    // Node-wise PCA controls (optional)
    const elShowPcaNodes = HAS_PCA_UI ? document.getElementById('showPcaNodes') : null;
    const elPcxNodes = HAS_PCA_UI ? document.getElementById('pcxNodes') : null;
    const elPcyNodes = HAS_PCA_UI ? document.getElementById('pcyNodes') : null;
    const elPcaNodesInfo = HAS_PCA_UI ? document.getElementById('pcaNodesInfo') : null;
    const elPcaNodesContainer = HAS_PCA_UI ? document.getElementById('pcaNodesContainer') : null;
    const elPcaNodesCanvas = HAS_PCA_UI ? document.getElementById('pcaNodesCanvas') : null;
    const elPcaNodesTitle = HAS_PCA_UI ? document.getElementById('pcaNodesTitle') : null;
    const elColorGraphNodes = HAS_PCA_UI ? document.getElementById('colorGraphNodes') : null;

    const btnPlay = document.getElementById('play');
    const btnPause = document.getElementById('pause');
    const btnStepBack = document.getElementById('stepBack');
    const btnStepForward = document.getElementById('stepForward');
    const btnAll = document.getElementById('all');
    const btnNone = document.getElementById('none');

    const elProbeList = document.getElementById('probeList');
    const elCardsContainer = document.getElementById('cardsContainer');

    // Cache PCA plot bounds so axis range stays consistent across time (per batch/PC selection)
    const pcaBoundsCache = HAS_PCA_UI ? { key: null, bounds: null } : null;
    const pcaNodesBoundsCache = HAS_PCA_UI ? { key: null, bounds: null } : null;

    // Provide stubs so non-PCA flow can still call these safely.
    const renderPCA = HAS_PCA_UI ? renderPCA_ : () => {};
    const renderPcaNodes = HAS_PCA_UI ? renderPcaNodes_ : () => {};

    function computeBoundsFromPoints(points) {
      let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
      for (const p of points){
        if (!p) continue;
        if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
        if (p.x < xmin) xmin = p.x; if (p.x > xmax) xmax = p.x;
        if (p.y < ymin) ymin = p.y; if (p.y > ymax) ymax = p.y;
      }
      if (!Number.isFinite(xmin) || !Number.isFinite(ymin)) return null;
      if (xmax === xmin) xmax = xmin + 1e-6;
      if (ymax === ymin) ymax = ymin + 1e-6;

      // Always include origin in view.
      xmin = Math.min(xmin, 0);
      xmax = Math.max(xmax, 0);
      ymin = Math.min(ymin, 0);
      ymax = Math.max(ymax, 0);

      const padX = 0.08 * (xmax - xmin);
      const padY = 0.12 * (ymax - ymin);
      xmin -= padX; xmax += padX;
      ymin -= padY; ymax += padY;

      // Keep origin visible even after padding.
      xmin = Math.min(xmin, 0);
      xmax = Math.max(xmax, 0);
      ymin = Math.min(ymin, 0);
      ymax = Math.max(ymax, 0);

      return { xmin, xmax, ymin, ymax };
    }

    function drawAxisLines(ctx, cssW, cssH, sx, sy) {
      ctx.save();
      ctx.strokeStyle = 'rgba(0,0,0,0.35)';
      ctx.lineWidth = 1;

      const x0 = sx(0);
      const y0 = sy(0);

      // y-axis (x=0)
      if (x0 >= 0 && x0 <= cssW) {
        ctx.beginPath();
        ctx.moveTo(x0 + 0.5, 10.5);
        ctx.lineTo(x0 + 0.5, cssH - 22.5);
        ctx.stroke();
      }
      // x-axis (y=0)
      if (y0 >= 0 && y0 <= cssH) {
        ctx.beginPath();
        ctx.moveTo(10.5, y0 + 0.5);
        ctx.lineTo(cssW - 10.5, y0 + 0.5);
        ctx.stroke();
      }

      // origin marker
      if (x0 >= 0 && x0 <= cssW && y0 >= 0 && y0 <= cssH) {
        ctx.fillStyle = 'rgba(0,0,0,0.55)';
        ctx.beginPath();
        ctx.arc(x0, y0, 3.5, 0, Math.PI*2);
        ctx.fill();
      }

      ctx.restore();
    }

    let algo = null;      // ONLY multi-probe format supported
    let timer = null;

    // --- FPS helpers: support decimals and "1/n" fractions, plus arrow stepping below 1.
    function parseFpsValue(v) {
      if (v == null) return NaN;
      const s = String(v).trim();
      if (!s) return NaN;
      // Accept fraction notation like "1/2", "3/4".
      const m = s.match(/^([+-]?\d+(?:\.\d+)?)\s*\/\s*([+-]?\d+(?:\.\d+)?)$/);
      if (m) {
        const num = Number(m[1]);
        const den = Number(m[2]);
        if (!Number.isFinite(num) || !Number.isFinite(den) || den === 0) return NaN;
        return num / den;
      }
      const n = Number(s);
      return Number.isFinite(n) ? n : NaN;
    }

    function formatFpsValueForInput(fps) {
      // Keep user-friendly fractional display for fps<1 (1/n), else show a number.
      if (!Number.isFinite(fps) || fps <= 0) return '';
      if (fps < 1) {
        const n = Math.max(1, Math.round(1 / fps));
        return `1/${n}`;
      }
      // For integers, show integer (avoid "6.000").
      if (Math.abs(fps - Math.round(fps)) < 1e-9) return String(Math.round(fps));
      // Otherwise, keep a concise decimal.
      return String(Number(fps.toFixed(6)));
    }

    function getFps() {
      const fps = parseFpsValue(elFps.value);
      if (!Number.isFinite(fps) || fps <= 0) return 6;
      // Keep existing upper bound behavior.
      return Math.min(60, fps);
    }

    function stepFps(delta) {
      let fps = parseFpsValue(elFps.value);
      if (!Number.isFinite(fps) || fps <= 0) fps = 6;

      if (delta > 0) {
        // Up arrow: if below 1, go from 1/n -> 1/(n-1) -> ... -> 1 -> 2 -> 3 ...
        if (fps < 1) {
          const n = Math.max(1, Math.round(1 / fps));
          if (n <= 1) fps = 2;
          else fps = 1 / (n - 1);
        } else {
          fps = Math.min(60, Math.floor(fps + 1e-9) + 1);
        }
      } else {
        // Down arrow: 3 -> 2 -> 1 -> 1/2 -> 1/3 -> 1/4 ...
        if (fps > 1) {
          fps = Math.max(1, Math.ceil(fps - 1e-9) - 1);
        } else {
          const n = Math.max(1, Math.round(1 / fps));
          fps = 1 / (n + 1);
        }
      }

      elFps.value = formatFpsValueForInput(fps);
      if (timer) play();
    }

    // Allow arrow key stepping even if the input is non-numeric (fractions).
    elFps.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        stepFps(+1);
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        stepFps(-1);
      }
    });

    // Re-enable mouse-wheel stepping like a native number input.
    // Supports fractions via stepFps(), and keeps page scroll working when not focused.
    elFps.addEventListener('wheel', (e) => {
      if (document.activeElement !== elFps) return;
      e.preventDefault();
      stepFps(e.deltaY < 0 ? +1 : -1);
    }, { passive: false });

    // When leaving the field, normalize formatting.
    elFps.addEventListener('blur', () => {
      const fps = getFps();
      elFps.value = formatFpsValueForInput(fps);
    });

    function stop() {
      if (timer) clearInterval(timer);
      timer = null;
    }

    function stepTime(delta) {
      if (!algo) return;
      stop();
      const maxT = Number(elT.max) || 0;
      let t = Number(elT.value) || 0;
      t = Math.max(0, Math.min(maxT, t + (delta | 0)));
      elT.value = String(t);
      detectAndRender();
    }

    function play() {
      stop();
      const fps = getFps();
      timer = setInterval(() => {
        const maxT = Number(elT.max);
        let t = Number(elT.value);
        t = (t >= maxT) ? 0 : (t + 1);
        elT.value = String(t);
        detectAndRender();
      }, Math.round(1000 / fps));
    }

    function clamp01(x) { return Math.max(0, Math.min(1, x)); }
    function probToGray(p, contrast=1.0) {
      p = clamp01((p - 0.5) * contrast + 0.5);
      const g = Math.round(255 * (1 - p));
      return `rgb(${g},${g},${g})`;
    }

    // Scalar-only color: white (0) -> blue (1). Used ONLY for probes with type===SCALAR.
    function probToBlue(p, contrast=1.0) {
      p = clamp01((p - 0.5) * contrast + 0.5);
      // White -> Blue: keep R,G fixed at 255*(1-p), B stays 255.
      const rg = Math.round(255 * (1 - p));
      return `rgb(${rg},${rg},255)`;
    }

    function isScalarType(typeStr) {
      if (!typeStr) return false;
      return String(typeStr).toLowerCase() === 'scalar';
    }

    // --- correctness graph helpers ---
    function hasCorrectnessSeries(p){
      if (!p || String(p.stage||'').toUpperCase()!=='HINT') return false;
      const s = p.correctness_along_time;
      if (!Array.isArray(s)) return false;
      // Accept either [T] or [B][T]
      if (s.length===0) return false;
      if (Array.isArray(s[0])) return Array.isArray(s[0]);
      return true;
    }

    function _selectCorrectnessSeriesForBatch(series, bIdx){
      // series can be [T] or [B][T]. Return [T].
      if (!Array.isArray(series)) return [];
      if (series.length===0) return [];
      if (Array.isArray(series[0])){
        const B = series.length;
        const bi = Math.max(0, Math.min(B-1, (bIdx|0)));
        const row = series[bi];
        return Array.isArray(row) ? row : [];
      }
      return series;
    }

    function drawCorrectnessGraph(canvas, series, tIdx, bIdx){
      // series is per-pred-step (typically steps 1..), values in [0,1]
      const series1D = _selectCorrectnessSeriesForBatch(series, bIdx);
      const Tfull = Array.isArray(series1D) ? series1D.length : 0;
      const Tshown = Math.max(0, Math.min(Tfull, Math.max(0, (tIdx|0))));

      const cssW = canvas.clientWidth || 540;
      const cssH = canvas.clientHeight || 110;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(cssW * dpr));
      canvas.height = Math.max(1, Math.floor(cssH * dpr));

      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.clearRect(0,0,cssW,cssH);

      // background
      ctx.fillStyle = '#fff';
      ctx.fillRect(0,0,cssW,cssH);

      // axes
      const padL=42, padR=10, padT=10, padB=22;
      const x0=padL, y0=cssH-padB;
      const x1=cssW-padR, y1=padT;

      ctx.strokeStyle = 'rgba(0,0,0,0.25)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0+0.5, y1+0.5);
      ctx.lineTo(x0+0.5, y0+0.5);
      ctx.lineTo(x1+0.5, y0+0.5);
      ctx.stroke();

      // y grid at 0, 0.5, 1
      ctx.strokeStyle = 'rgba(0,0,0,0.10)';
      ctx.fillStyle = 'rgba(0,0,0,0.60)';
      ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
      for (const gv of [0, 0.5, 1]){
        const yy = y0 - (y0-y1) * gv;
        ctx.beginPath();
        ctx.moveTo(x0, yy+0.5);
        ctx.lineTo(x1, yy+0.5);
        ctx.stroke();
        ctx.fillText(String(gv), 6, yy + 4);
      }

      // labels
      ctx.fillStyle = 'rgba(15, 12, 12, 0.65)';
      ctx.fillText('correct', 6, 12);

      // If nothing to plot yet, show a faint message.
      if (Tshown <= 0){
        ctx.fillStyle = 'rgba(0,0,0,0.45)';
        ctx.fillText('no points at current t', x0 + 6, y0 - 6);
        return;
      }

      const n = Tshown; // points 1..n shown
      const sx = (i) => {
        if (n <= 1) return x0;
        return x0 + (x1-x0) * ((i-1) / (n-1));
      };
      const sy = (v) => y0 - (y0-y1) * clamp01(Number(v) || 0);

      // line
      ctx.strokeStyle = '#4E79A7';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i=1;i<=n;i++){
        const v = series1D[i-1];
        const X = sx(i);
        const Y = sy(v);
        if (i===1) ctx.moveTo(X, Y);
        else ctx.lineTo(X, Y);
      }
      ctx.stroke();

      // points
      ctx.fillStyle = '#4E79A7';
      for (let i=1;i<=n;i++){
        const v = series1D[i-1];
        const X = sx(i);
        const Y = sy(v);
        ctx.beginPath();
        ctx.arc(X, Y, 2.8, 0, Math.PI*2);
        ctx.fill();
      }

      // current-t marker at last point
      ctx.fillStyle = '#E15759';
      {
        const X = sx(n);
        const Y = sy(series1D[n-1]);
        ctx.beginPath();
        ctx.arc(X, Y, 3.6, 0, Math.PI*2);
        ctx.fill();
      }

      // x axis label (include batch when series is per-batch)
      ctx.fillStyle = 'rgba(0,0,0,0.60)';
      const showB = Array.isArray(series) && series.length>0 && Array.isArray(series[0]);
      ctx.fillText(showB ? `b=${(bIdx|0)} t<=${tIdx}` : `t<=${tIdx}`, x1-110, cssH-6);
    }

    function getDims(arr) {
      const dims = [];
      let cur = arr;
      while (Array.isArray(cur)) {
        dims.push(cur.length);
        cur = cur[0];
      }
      return dims;
    }

    function setHoverText(s){
      elHoverText.textContent = s || "Hover any cell to see its value here.";
    }

    function isCategoricalType(typeStr) {
      if (!typeStr) return false;
      const s = String(typeStr).toLowerCase();
      // Treat MASK_ONE like categorical for rendering ([B,T,N,K] with K classes)
      return s.includes('categorical') || s.includes('mask_one');
    }

    function labelToColor(label) {
      const idx = ((label|0) % CAT_PALETTE.length + CAT_PALETTE.length) % CAT_PALETTE.length;
      return CAT_PALETTE[idx];
    }

    function argmax1D(arrK) {
      let best = 0;
      let bestV = Number(arrK[0]);
      for (let k=1;k<arrK.length;k++){
        const v = Number(arrK[k]);
        if (v > bestV) { bestV = v; best = k; }
      }
      return best;
    }

    // Robust label decoding for mismatch-style errors.
    // - If the values look like a distribution/logits vector, use argmax.
    // - If the values are already indices (or close to integers), round.
    const _IDX_EPS = 1e-6;
    function looksLikeIndexScalar(x) {
      const v = Number(x);
      if (!Number.isFinite(v)) return false;
      return Math.abs(v - Math.round(v)) <= _IDX_EPS;
    }
    function decodeBinaryMask(v) {
      // For MASK_ONE encoded as scalar/probability, treat >=0.5 as 1.
      const x = Number(v);
      if (!Number.isFinite(x)) return 0;
      return x >= 0.5 ? 1 : 0;
    }
    function decodeLabelFromVector(vecK) {
      // If vecK is effectively one-hot / probabilities / logits => argmax.
      return argmax1D(vecK);
    }

    // --- PCA axis helpers (lightweight) ---
    function niceStep(span, targetTicks) {
      if (!Number.isFinite(span) || span <= 0) return 1;
      const raw = span / Math.max(2, targetTicks);
      const pow = Math.pow(10, Math.floor(Math.log10(raw)));
      const frac = raw / pow;
      let nice;
      if (frac <= 1) nice = 1;
      else if (frac <= 2) nice = 2;
      else if (frac <= 5) nice = 5;
      else nice = 10;
      return nice * pow;
    }

    function formatTick(v) {
      if (!Number.isFinite(v)) return '';
      const av = Math.abs(v);
      if (av >= 1000) return v.toFixed(0);
      if (av >= 100) return v.toFixed(1);
      if (av >= 10) return v.toFixed(2);
      if (av >= 1) return v.toFixed(3);
      return v.toExponential(2);
    }

    function drawAxes(ctx, cssW, cssH, xmin, xmax, ymin, ymax) {
      const L = 10.5, T = 10.5, R = cssW - 10.5, B = cssH - 22.5; // leave room for x tick labels
      const w = R - L, h = B - T;
      if (!(w > 0 && h > 0) || !(xmax > xmin) || !(ymax > ymin)) return;

      const stepX = niceStep(xmax - xmin, 4);
      const stepY = niceStep(ymax - ymin, 4);
      const startX = Math.ceil(xmin / stepX) * stepX;
      const startY = Math.ceil(ymin / stepY) * stepY;

      ctx.save();
      ctx.strokeStyle = 'rgba(0,0,0,0.15)';
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.lineWidth = 1;
      ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace';

      for (let x = startX; x <= xmax + 1e-12; x += stepX) {
        const px = (x - xmin) / (xmax - xmin) * w + L;
        ctx.beginPath();
        ctx.moveTo(px, B);
        ctx.lineTo(px, B + 5);
        ctx.stroke();
        const s = formatTick(x);
        ctx.fillText(s, px - ctx.measureText(s).width / 2, B + 16);
      }

      for (let y = startY; y <= ymax + 1e-12; y += stepY) {
        const py = (1 - (y - ymin) / (ymax - ymin)) * h + T;
        ctx.beginPath();
        ctx.moveTo(L - 5, py);
        ctx.lineTo(L, py);
        ctx.stroke();
        const s = formatTick(y);
        ctx.fillText(s, 0.5, py + 3.5);
      }

      ctx.restore();
    }

    function nodeIndexToColor(i, N) {
      // Low index -> warm (red), high index -> cold (blue)
      if (N <= 1) return '#E15759';
      const t = i / (N - 1);
      const hue = 0 + 220 * t; // 0=red (warm) -> 220=blue (cold)
      return `hsl(${hue.toFixed(1)}, 80%, 45%)`;
    }

    function validTFromIsLast(bIdx) {
      const isLast = algo?.is_last;
      if (!Array.isArray(isLast) || isLast.length === 0) return null;
      const b = Math.max(0, Math.min(isLast.length - 1, bIdx));
      const row = isLast[b];
      if (!Array.isArray(row) || row.length === 0) return null;
      for (let t=0; t<row.length; t++){
        if (row[t] === true) return t + 1; // inclusive
      }
      return row.length;
    }

    function validTFromLengths(bIdx, selectedNames) {
      const probesMap = algo?.probes || {};
      const globalLengths = Array.isArray(algo?.lengths) ? algo.lengths : null;
      let best = Infinity;
      const b = Math.max(0, bIdx|0);
      for (const name of selectedNames){
        const p = probesMap[name];
        if (!p || p.error) continue;
        const lengths = Array.isArray(p.lengths) ? p.lengths : globalLengths;
        if (Array.isArray(lengths) && lengths.length > b){
          const L = Number(lengths[b]);
          if (Number.isFinite(L) && L > 0) best = Math.min(best, L);
        }
      }
      return (best !== Infinity) ? best : null;
    }

    function makeTimeVectorViewWithLengths(tensor, lengths) {
      const dims = getDims(tensor);
      const L = Array.isArray(lengths) ? lengths.length : null;

      if (dims.length === 2) {
        const [T,N]=dims;
        return { kind:'vector', dims, T, N, B:1, b:0, get:(t,j)=>tensor[t][j] };
      }
      if (dims.length === 3) {
        const [d0,d1,d2]=dims;
        // Do NOT early-classify as matrix just because T==N; rely on metadata/heuristics below.

        if (L!=null) {
          if (d0===L) {
            const B=d0,T=d1,N=d2;
            const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
            return { kind:'vector', dims, T, N, B, b, get:(t,j)=>tensor[b][t][j] };
          }
          if (d1===L) {
            const T=d0,B=d1,N=d2;
            const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
            return { kind:'vector', dims, T, N, B, b, get:(t,j)=>tensor[t][b][j] };
          }
        }
        const useBTN = (d0 <= 64 && d1 > d0);
        if (useBTN) {
          const B=d0,T=d1,N=d2;
          const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
          return { kind:'vector', dims, T, N, B, b, get:(t,j)=>tensor[b][t][j] };
        }
        const T=d0,B=d1,N=d2;
        const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
        return { kind:'vector', dims, T, N, B, b, get:(t,j)=>tensor[t][b][j] };
      }
      throw new Error('not-vector');
    }

    function makeTimeMatrixViewWithLengths(tensor, lengths) {
      const dims = getDims(tensor);
      const L = Array.isArray(lengths) ? lengths.length : null;

      if (dims.length === 2) {
        const [H,W]=dims; const T=1;
        return { kind:'matrix', dims:[T,H,W], T, H, W, B:1, b:0, get:(_t,i,j)=>tensor[i][j] };
      }
      if (dims.length === 3) {
        const [T,H,W]=dims;
        return { kind:'matrix', dims, T, H, W, B:1, b:0, get:(t,i,j)=>tensor[t][i][j] };
      }
      if (dims.length === 4) {
        const [d0,d1,H,W]=dims;
        if (L!=null) {
          if (d0===L) {
            const B=d0,T=d1;
            const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
            return { kind:'matrix', dims, T, H, W, B, b, get:(t,i,j)=>tensor[b][t][i][j] };
          }
          if (d1===L) {
            const T=d0,B=d1;
            const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
            return { kind:'matrix', dims, T, H, W, B, b, get:(t,i,j)=>tensor[t][b][i][j] };
          }
        }
        if (d0 <= 64 && d1 > d0) {
          const B=d0,T=d1;
          const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
          return { kind:'matrix', dims, T, H, W, B, b, get:(t,i,j)=>tensor[b][t][i][j] };
        }
        const T=d0,B=d1,N=d2;
        const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
        return { kind:'matrix', dims, T, N, B, b, get:(t,j)=>tensor[t][b][j] };
      }
      throw new Error('not-matrix');
    }

    // --- CATEGORICAL views ---
    // [B,T,N,K] or [T,B,N,K]
    function makeTimeCategoricalVectorView(tensor, lengths) {
      const dims = getDims(tensor);
      const L = Array.isArray(lengths) ? lengths.length : null;

      if (dims.length !== 4) throw new Error('not-cat-vector');
      const [d0,d1,d2,d3] = dims;
      const K = d3;

      if (L != null) {
        if (d0 === L) { // [B,T,N,K]
          const B=d0,T=d1,N=d2;
          const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
          return { kind:'catvec', dims, B, T, N, K, b, get:(t,j,k)=>tensor[b][t][j][k] };
        }
        if (d1 === L) { // [T,B,N,K]
          const T=d0,B=d1,N=d2;
          const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
          return { kind:'catvec', dims, B, T, N, K, b, get:(t,j,k)=>tensor[t][b][j][k] };
        }
      }
      // heuristic: assume [B,T,N,K]
      const B=d0,T=d1,N=d2;
      const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
      return { kind:'catvec', dims, B, T, N, K, b, get:(t,j,k)=>tensor[b][t][j][k] };
    }

    // [B,T,H,W,K] or [T,B,H,W,K]
    function makeTimeCategoricalMatrixView(tensor, lengths) {
      const dims = getDims(tensor);
      const L = Array.isArray(lengths) ? lengths.length : null;

      if (dims.length !== 5) throw new Error('not-cat-matrix');
      const [d0,d1,H,W,K] = dims;

      if (L != null) {
        if (d0 === L) { // [B,T,H,W,K]
          const B=d0,T=d1;
          const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
          return { kind:'catmat', dims, B, T, H, W, K, b, get:(t,i,j,k)=>tensor[b][t][i][j][k] };
        }
        if (d1 === L) { // [T,B,H,W,K]
          const T=d0,B=d1;
          const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
          return { kind:'catmat', dims, B, T, H, W, K, b, get:(t,i,j,k)=>tensor[t][b][i][j][k] };
        }
      }
      // heuristic: assume [B,T,H,W,K]
      const B=d0,T=d1;
      const b=Math.max(0,Math.min(B-1,Number(elBidx.value)||0)); elBidx.max=B-1;
      return { kind:'catmat', dims, B, T, H, W, K, b, get:(t,i,j,k)=>tensor[b][t][i][j][k] };
    }

    function finalizeScale(values){
      if (!values || values.length===0) return { bothIn01:true,min:0,max:1 };
      let minV=Infinity,maxV=-Infinity,allIn01=true;
      for (const v of values){
        if(!Number.isFinite(v)) continue;
        if(v<minV) minV=v;
        if(v>maxV) maxV=v;
        if(!(v>=0 && v<=1)) allIn01=false;
      }
      if (minV===Infinity) return {bothIn01:true,min:0,max:1};
      if (allIn01) return {bothIn01:true,min:0,max:1};
      if (maxV===minV) maxV=minV+1e-6;
      return {bothIn01:false,min:minV,max:maxV};
    }

    function computeProbeScale(probe, lengths) {
      if (!probe || !probe.true || !probe.pred) return {bothIn01:true,min:0,max:1};

      // Categorical: scale doesn't make sense for label map; return [0,1]
      if (isCategoricalType(probe.type)) {
        return {bothIn01:true, min:0, max:1};
      }

      try {
        const vTrue = makeTimeVectorViewWithLengths(probe.true, lengths);
        const vPred = makeTimeVectorViewWithLengths(probe.pred, lengths);
        const N = Math.min(vTrue.N, vPred.N);
        const T = Math.min(vTrue.T, vPred.T);
        let values = [];
        for (let t=0;t<T;t++){
          for (let j=0;j<N;j++){
            const a = Number(vTrue.get(t,j));
            const b = Number(vPred.get(t,j));
            if (Number.isFinite(a)) values.push(a);
            if (Number.isFinite(b)) values.push(b);
          }
        }
        return finalizeScale(values);
      } catch(_) {}

      try {
        const mTrue = makeTimeMatrixViewWithLengths(probe.true, lengths);
        const mPred = makeTimeMatrixViewWithLengths(probe.pred, lengths);
        const H = Math.min(mTrue.H, mPred.H);
        const W = Math.min(mTrue.W, mPred.W);
        const T = Math.min(mTrue.T, mPred.T);

        let r0 = Math.max(0, parseInt(elR0.value||'0',10));
        let r1 = Math.max(0, parseInt(elR1.value||'0',10));
        let c0 = Math.max(0, parseInt(elC0.value||'0',10));
        let c1 = Math.max(0, parseInt(elC1.value||'0',10));
        if (!Number.isFinite(r0)) r0 = 0; if (!Number.isFinite(c0)) c0 = 0;
        if (!Number.isFinite(r1) || r1===0) r1 = H; if (!Number.isFinite(c1) || c1===0) c1 = W;
        r0 = Math.min(r0, H); r1 = Math.min(Math.max(r1, r0+0), H);
        c0 = Math.min(c0, W); c1 = Math.min(Math.max(c1, c0+0), W);

        let values = [];
        for (let t=0;t<T;t++){
          for (let i=r0;i<r1;i++){
            for (let j=c0;j<c1;j++){
              const a = Number(mTrue.get(t,i,j));
              const b = Number(mPred.get(t,i,j));
              if (Number.isFinite(a)) values.push(a);
              if (Number.isFinite(b)) values.push(b);
            }
          }
        }
        return finalizeScale(values);
      } catch(_) {}

      return {bothIn01:true,min:0,max:1};
    }

    function computeProbeErrScale(probe, lengths) {
      if (!probe || !probe.true || !probe.pred) return {bothIn01:true,min:0,max:1};

      const typeStr = String(probe.type || '').toLowerCase();

      // Categorical error: mismatch is binary (already implemented elsewhere)
      if (isCategoricalType(probe.type)) {
        return {bothIn01:true, min:0, max:1};
      }

      // Pointer and MASK_ONE are rendered as numeric tensors in many cases.
      // For error, we prefer discrete mismatch to match CLRS evaluation semantics.
      // - POINTER: support either [B,T,N] indices or [B,T,N,K] distributions (rare in this viewer path).
      // - MASK_ONE: support either scalar mask [B,T,N] (0/1 or prob) OR one-hot [B,T,N,K] (handled by categorical branch above).
      const wantsMismatch = (typeStr === 'pointer' || typeStr === 'mask_one');
      if (wantsMismatch) return {bothIn01:true, min:0, max:1};

      // Default numeric error: |pred-true|
      try {
        const vTrue = makeTimeVectorViewWithLengths(probe.true, lengths);
        const vPred = makeTimeVectorViewWithLengths(probe.pred, lengths);
        const N = Math.min(vTrue.N, vPred.N);
        const T = Math.min(vTrue.T, vPred.T);
        let maxV = 0;
        for (let t=0;t<T;t++){
          for (let j=0;j<N;j++){
            const a = Number(vTrue.get(t,j));
            const b = Number(vPred.get(t,j));
            const e = Math.abs(b - a);
            if (Number.isFinite(e) && e > maxV) maxV = e;
          }
        }
        if (maxV <= 1) return {bothIn01:true,min:0,max:1};
        return {bothIn01:false,min:0,max:Math.max(maxV,1e-6)};
      } catch(_) {}

      try {
        const mTrue = makeTimeMatrixViewWithLengths(probe.true, lengths);
        const mPred = makeTimeMatrixViewWithLengths(probe.pred, lengths);
        const H = Math.min(mTrue.H, mPred.H);
        const W = Math.min(mTrue.W, mPred.W);
        const T = Math.min(mTrue.T, mPred.T);

        let r0 = Math.max(0, parseInt(elR0.value||'0',10));
        let r1 = Math.max(0, parseInt(elR1.value||'0',10));
        let c0 = Math.max(0, parseInt(elC0.value||'0',10));
        let c1 = Math.max(0, parseInt(elC1.value||'0',10));
        if (!Number.isFinite(r0)) r0 = 0; if (!Number.isFinite(c0)) c0 = 0;
        if (!Number.isFinite(r1) || r1===0) r1 = H; if (!Number.isFinite(c1) || c1===0) c1 = W;
        r0 = Math.min(r0, H); r1 = Math.min(Math.max(r1, r0+0), H);
        c0 = Math.min(c0, W); c1 = Math.min(Math.max(c1, c0+0), W);

        let maxV = 0;
        for (let t=0;t<T;t++){
          for (let i=r0;i<r1;i++){
            for (let j=c0;j<c1;j++){
              const a = Number(mTrue.get(t,i,j));
              const b = Number(mPred.get(t,i,j));
              const e = Math.abs(b - a);
              if (Number.isFinite(e) && e > maxV) maxV = e;
            }
          }
        }
        if (maxV <= 1) return {bothIn01:true,min:0,max:1};
        return {bothIn01:false,min:0,max:Math.max(maxV,1e-6)};
      } catch(_) {}

      return {bothIn01:true,min:0,max:1};
    }

    function drawVectorTo(ctx, canvas, raw, scale, cell, contrast, colorFn = probToGray) {
      const N = raw.length;
      canvas.width = Math.max(1, N * cell);
      canvas.height = Math.max(1, cell);
      ctx.clearRect(0,0,canvas.width,canvas.height);

      const bothIn01 = scale.bothIn01;
      const gmin = scale.min;
      const gden = (scale.max - scale.min) || 1;

      for (let j=0;j<N;j++){
        const val=raw[j];
        const p = (bothIn01 ? val : ((val - gmin) / gden));
        ctx.fillStyle = colorFn(p, contrast);
        ctx.fillRect(j*cell, 0, cell, cell);
      }
      ctx.strokeStyle='rgba(0,0,0,0.10)';
      for (let j=0;j<=N;j++){
        ctx.beginPath(); ctx.moveTo(j*cell+0.5,0); ctx.lineTo(j*cell+0.5,cell); ctx.stroke();
      }
      ctx.beginPath(); ctx.moveTo(0,cell+0.5); ctx.lineTo(N*cell,cell+0.5); ctx.stroke();
    }

    function drawMatrixTo(ctx, canvas, H, W, getVal, scale, cell, contrast, colorFn = probToGray) {
      canvas.width = Math.max(1, W * cell);
      canvas.height = Math.max(1, H * cell);
      ctx.clearRect(0,0,canvas.width,canvas.height);

      const bothIn01 = scale.bothIn01;
      const gmin = scale.min;
      const gden = (scale.max - scale.min) || 1;

      for (let i=0;i<H;i++){
        for (let j=0;j<W;j++){
          const v = Number(getVal(i,j));
          const p = (bothIn01 ? v : ((v - gmin) / gden));
          ctx.fillStyle = colorFn(p, contrast);
          ctx.fillRect(j*cell, i*cell, cell, cell);
        }
      }
      ctx.strokeStyle='rgba(0,0,0,0.08)';
      for (let i=0;i<=H;i++){
        ctx.beginPath(); ctx.moveTo(0,i*cell+0.5); ctx.lineTo(W*cell,i*cell+0.5); ctx.stroke();
      }
      for (let j=0;j<=W;j++){
        ctx.beginPath(); ctx.moveTo(j*cell+0.5,0); ctx.lineTo(j*cell+0.5,H*cell); ctx.stroke();
      }
    }

    function drawCategoricalVector(ctx, canvas, labels, cell) {
      const N = labels.length;
      canvas.width = Math.max(1, N * cell);
      canvas.height = Math.max(1, cell);
      ctx.clearRect(0,0,canvas.width,canvas.height);

      for (let j=0;j<N;j++){
        ctx.fillStyle = labelToColor(labels[j]);
        ctx.fillRect(j*cell, 0, cell, cell);
      }
      ctx.strokeStyle='rgba(0,0,0,0.12)';
      for (let j=0;j<=N;j++){
        ctx.beginPath(); ctx.moveTo(j*cell+0.5,0); ctx.lineTo(j*cell+0.5,cell); ctx.stroke();
      }
      ctx.beginPath(); ctx.moveTo(0,cell+0.5); ctx.lineTo(N*cell,cell+0.5); ctx.stroke();
    }

    function drawCategoricalMatrix(ctx, canvas, H, W, getLabel, cell) {
      canvas.width = Math.max(1, W * cell);
      canvas.height = Math.max(1, H * cell);
      ctx.clearRect(0,0,canvas.width,canvas.height);

      for (let i=0;i<H;i++){
        for (let j=0;j<W;j++){
          ctx.fillStyle = labelToColor(getLabel(i,j));
          ctx.fillRect(j*cell, i*cell, cell, cell);
        }
      }
      ctx.strokeStyle='rgba(0,0,0,0.10)';
      for (let i=0;i<=H;i++){
        ctx.beginPath(); ctx.moveTo(0,i*cell+0.5); ctx.lineTo(W*cell,i*cell+0.5); ctx.stroke();
      }
      for (let j=0;j<=W;j++){
        ctx.beginPath(); ctx.moveTo(j*cell+0.5,0); ctx.lineTo(j*cell+0.5,H*cell); ctx.stroke();
      }
    }

    // --- Fallback flattening (avoid "unsupported shape") ---
    function flattenFrameTo2D(frame) {
      if (!Array.isArray(frame)) {
        return { H: 1, W: 1, get: (_i,_j)=>Number(frame) };
      }
      const H = frame.length;
      if (H === 0) return { H: 0, W: 0, get: ()=>0 };
      if (!Array.isArray(frame[0])) {
        const W = frame.length;
        return { H: 1, W, get: (_i,j)=>Number(frame[j]) };
      }
      const rows = new Array(H);
      let Wmax = 0;
      for (let i=0;i<H;i++){
        const row = frame[i];
        const flat = [];
        const stack = [row];
        while (stack.length){
          const cur = stack.pop();
          if (Array.isArray(cur)){
            for (let k=cur.length-1;k>=0;k--) stack.push(cur[k]);
          } else {
            flat.push(Number(cur));
          }
        }
        rows[i] = flat;
        if (flat.length > Wmax) Wmax = flat.length;
      }
      return {
        H, W: Wmax,
        get: (i,j) => (j < rows[i].length ? rows[i][j] : 0)
      };
    }

    function attachHover(canvas, getter, name) {
      canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const info = getter();
        if (!info){ setHoverText(""); return; }
        if (info.type==='vector'){
          const j = Math.floor(x / info.cell);
          if (j < 0 || j >= info.N){ setHoverText(""); return; }
          setHoverText(`${name} • ${info.which} • vector: t=${info.t}, j=${j}, value=${String(info.get(j))}`);
        } else {
          const i = Math.floor(y / info.cell);
          const j = Math.floor(x / info.cell);
          if (i<0||j<0||i>=info.H||j>=info.W){ setHoverText(""); return; }
          setHoverText(`${name} • ${info.which} • matrix: t=${info.t}, i=${i}, j=${j}, value=${String(info.get(i,j))}`);
        }
      });
      canvas.addEventListener('mouseleave', () => { setHoverText(""); });
    }

    function buildProbeList(probesMap){
      const names = Object.keys(probesMap);
      for (const n of names){
        if (!selectedState.has(n)) selectedState.set(n, true);
        if (!correctnessState.has(n)) correctnessState.set(n, false);
      }

      elProbeList.innerHTML = '';
      for (const name of names){
        const p = probesMap[name];
        const row = document.createElement('div');
        row.className = 'probeRow';

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = !!selectedState.get(name);
        cb.addEventListener('change', () => {
          selectedState.set(name, cb.checked);
          detectAndRender();
        });

        const span = document.createElement('span');
        span.textContent = name;

        const badge = document.createElement('span');
        badge.className = 'badge';
        const typeStr = p?.type ? String(p.type) : 'probe';
        const locStr = p?.location ? String(p.location) : '';
        const stageStr = p?.stage ? String(p.stage).toUpperCase() : '';

        const parts = [];
        if (typeStr) parts.push(typeStr);
        if (locStr) parts.push(locStr);
        if (stageStr === 'OUTPUT') parts.push('OUTPUT');
        badge.textContent = parts.join(', ');

        // Per-hint correctness graph toggle (disabled if not a hint or no series)
        const corrBtn = document.createElement('input');
        corrBtn.type = 'checkbox';
        corrBtn.title = 'Show correctness graph';
        corrBtn.checked = !!correctnessState.get(name);
        corrBtn.disabled = !hasCorrectnessSeries(p);
        corrBtn.addEventListener('change', (e) => {
          e.stopPropagation();
          correctnessState.set(name, corrBtn.checked);
          detectAndRender();
        });

        const corrLabel = document.createElement('span');
        corrLabel.className = 'small mono';
        corrLabel.style.marginLeft = '6px';
        corrLabel.style.opacity = corrBtn.disabled ? '0.45' : '0.9';
        corrLabel.textContent = 'corr';

        row.appendChild(cb);
        row.appendChild(span);
        row.appendChild(badge);
        row.appendChild(corrBtn);
        row.appendChild(corrLabel);

        row.addEventListener('click', (e) => {
          if (e.target === cb || e.target === corrBtn) return;
          cb.checked = !cb.checked;
          selectedState.set(name, cb.checked);
          detectAndRender();
        });

        elProbeList.appendChild(row);
      }
    }

    function setAllProbes(checked){
      const probesMap = algo?.probes || {};
      for (const name of Object.keys(probesMap)){
        selectedState.set(name, checked);
      }
      detectAndRender();
    }

    function buildCard(name, probeInfo, isVector){
      const card = document.createElement('div');
      card.className = 'card';

      const header = document.createElement('div');
      header.className = 'cardHeader';

      const title = document.createElement('div');
      title.className = 'title';
      title.textContent = name;

      const sub = document.createElement('div');
      sub.className = 'subtitle';
      {
        const parts = [];
        if (probeInfo?.type) parts.push(String(probeInfo.type));
        if (probeInfo?.location) parts.push(String(probeInfo.location));
        if (String(probeInfo?.stage || '').toUpperCase() === 'OUTPUT') parts.push('OUTPUT');
        sub.textContent = parts.join(' • ');
      }

      header.appendChild(title);
      header.appendChild(sub);

      const body = document.createElement('div');
      body.className = isVector ? 'vectorStack' : 'matrixRow';

      function mkBlock(which){
        const block = document.createElement('div');
        block.className = 'block';

        const bt = document.createElement('div');
        bt.className = 'blockTitle mono';
        const left = document.createElement('span'); left.textContent = which;
        const right = document.createElement('span'); right.className = 'shapeNote mono';
        bt.appendChild(left); bt.appendChild(right);

        const wrap = document.createElement('div');
        wrap.className = 'canvasWrap';

        const c = document.createElement('canvas');

        wrap.appendChild(c);
        block.appendChild(bt);
        block.appendChild(wrap);

        c._hoverGetter = null;
        attachHover(c, ()=>c._hoverGetter && c._hoverGetter(), name);

        return {block, canvas:c, shapeEl:right};
      }

      const bTrue = mkBlock('true');
      const bPred = mkBlock('pred');
      const bErr  = mkBlock('err');

      body.appendChild(bTrue.block);
      body.appendChild(bPred.block);
      body.appendChild(bErr.block);

      card.appendChild(header);
      card.appendChild(body);

      card._cTrue = bTrue.canvas; card._cPred = bPred.canvas; card._cErr = bErr.canvas;
      card._sTrue = bTrue.shapeEl; card._sPred = bPred.shapeEl; card._sErr = bErr.shapeEl;

      // Hint correctness plot (inserted below probe display, hidden by default)
      const corrWrap = document.createElement('div');
      corrWrap.className = 'canvasWrap';
      corrWrap.style.marginTop = '10px';
      corrWrap.style.display = 'none';

      const corrTitle = document.createElement('div');
      corrTitle.className = 'small mono';
      corrTitle.style.margin = '2px 0 8px 2px';
      corrTitle.textContent = 'correctness (hint)';

      const corrCanvas = document.createElement('canvas');
      corrCanvas.style.width = '100%';
      corrCanvas.style.height = '110px';
      corrCanvas.style.border = '1px solid var(--border)';
      corrCanvas.style.borderRadius = '8px';
      corrCanvas.style.background = '#fff';

      corrWrap.appendChild(corrTitle);
      corrWrap.appendChild(corrCanvas);
      card.appendChild(corrWrap);

      card._corrWrap = corrWrap;
      card._corrCanvas = corrCanvas;

      return card;
    }

    function computeGlobalMaxHW(){
      const probesMap = algo?.probes || {};
      const globalLengths = Array.isArray(algo?.lengths) ? algo.lengths : null;
      let maxH = 0, maxW = 0;
      for (const name of Object.keys(probesMap)){
        const p = probesMap[name];
        if (!p || p.error) continue;
        const lengths = Array.isArray(p.lengths) ? p.lengths : globalLengths;
        try {
          const mT = makeTimeMatrixViewWithLengths(p.true, lengths);
          if (mT.H > maxH) maxH = mT.H;
          if (mT.W > maxW) maxW = mT.W;
        } catch(_) {}
        try {
          const mP = makeTimeMatrixViewWithLengths(p.pred, lengths);
          if (mP.H > maxH) maxH = mP.H;
          if (mP.W > maxW) maxW = mP.W;
        } catch(_) {}
      }
      return {maxH, maxW};
    }

    function detectAndRender(){
      if (!algo) return;

      const probesMap = algo.probes || {};
      const globalLengths = Array.isArray(algo.lengths) ? algo.lengths : null;

      const algoName = algo.algorithm ? String(algo.algorithm) : 'algorithm';
      elAlgoName.textContent = `Loaded: ${algoName} • probes=${Object.keys(probesMap).length}`;
      elTopMeta.textContent = algoName;

      const vcell = Math.max(1, Math.min(120, Number(elVCell.value) || 13));
      const mcell = Math.max(1, Math.min(80, Number(elMCell.value) || 4));
      const contrast = Number(elContrast.value) || 1;
      elContrastLabel.textContent = `contrast=${contrast.toFixed(1)}`;

      // Compute global maximum 2D shape across all matrix-like probes
      const {maxH, maxW} = computeGlobalMaxHW();
      // Set defaults for submatrix end indices if unset (0)
      if (maxH > 0) {
        elR0.max = String(maxH);
        elR1.max = String(maxH);
        const curR1 = Number(elR1.value) || 0;
        if (curR1 === 0) elR1.value = String(maxH);
      }
      if (maxW > 0) {
        elC0.max = String(maxW);
        elC1.max = String(maxW);
        const curC1 = Number(elC1.value) || 0;
        if (curC1 === 0) elC1.value = String(maxW);
      }

      buildProbeList(probesMap);

      const selectedNames = Object.keys(probesMap).filter(n => !!selectedState.get(n));

      const vectorNames = [];
      const matrixNames = [];
      const scalarNames = [];
      let effT = Infinity;
      let anyBatch=false; let allB1=true;

      const bIdx = Math.max(0, Number(elBidx.value) || 0);
      const tCapFromIsLast = validTFromIsLast(bIdx);
      const tCapFromLengths = validTFromLengths(bIdx, selectedNames);
      if (tCapFromIsLast != null) effT = Math.min(effT, tCapFromIsLast);
      else if (tCapFromLengths != null) effT = Math.min(effT, tCapFromLengths);

      // categorize
      for (const name of selectedNames){
        const p = probesMap[name];
        if (!p || p.error) continue;
        const lengths = Array.isArray(p.lengths) ? p.lengths : globalLengths;

        let T=null, B=null;
        const dimsTrue = getDims(p.true);
        const typeStr = String(p.type || '').toUpperCase();
        const locStr = String(p.location || '').toUpperCase();
        const axes = Array.isArray(p.axes) ? p.axes : Array.isArray(p.true?.axes) ? p.true.axes : null;

        // GRAPH location means a single global value per timestep (not per node/edge)
        if (locStr === 'GRAPH'){
          scalarNames.push(name);
          try {
            const s = makeTimeScalarViewWithLengths(p.true, lengths);
            T = s.T; B = s.B;
          } catch(_e){
            // fallback: infer T/B from raw dims when possible
            if (dimsTrue.length >= 2) { B = dimsTrue[0]; T = dimsTrue[1]; }
          }
          if (T != null) effT = Math.min(effT, T);
          anyBatch = anyBatch || (B != null);
          if (B != null && B > 1) allB1 = false;
          continue;
        }

        function preferVectorByMetadata(){
          if (locStr === 'NODE') return true;
          if (Array.isArray(axes)){
            const nCount = axes.reduce((a,x)=>a + (String(x).toUpperCase()==='N'), 0);
            if (nCount === 1) return true;
          }
          if (typeStr === 'MASK_ONE' && locStr !== 'EDGE') return true;
          return false;
        }

        if (isCategoricalType(p.type) && dimsTrue.length === 4 && dimsTrue[3] <= 64) {
          vectorNames.push(name);
          try {
            const cv = makeTimeCategoricalVectorView(p.true, lengths);
            T = cv.T; B = cv.B;
          } catch(_) {}
        } else {
          const forceVector = preferVectorByMetadata();
          if (forceVector) {
            try {
              const v = makeTimeVectorViewWithLengths(p.true, lengths);
              vectorNames.push(name);
              T=v.T; B=v.B;
            } catch(_) {
              try {
                const m = makeTimeMatrixViewWithLengths(p.true, lengths);
                matrixNames.push(name);
                T=m.T; B=m.B;
              } catch(_) {}
            }
          } else {
            try {
              const v = makeTimeVectorViewWithLengths(p.true, lengths);
              vectorNames.push(name);
              T=v.T; B=v.B;
            } catch(_) {
              matrixNames.push(name);
              try {
                const m = makeTimeMatrixViewWithLengths(p.true, lengths);
                T=m.T; B=m.B;
              } catch(_) {}
            }
          }
        }

        if (T != null) effT = Math.min(effT, T);
        anyBatch = anyBatch || (B != null);
        if (B != null && B > 1) allB1 = false;
      }

      if (effT === Infinity) effT = 1;

      const prevT = Number(elT.value)||0;
      elT.max = Math.max(0, effT>0 ? effT-1 : 0);
      const tIdx = Math.min(prevT, Number(elT.max));
      elT.value = String(tIdx);
      elTLabel.textContent = `t=${elT.value} / ${elT.max}`;

      elBidx.parentElement.style.visibility = (!anyBatch || allB1) ? 'hidden' : 'visible';

      elCardsContainer.innerHTML = '';

      function getScaleFor(name, probe, lengths){
        if (!elLockScale.checked) return null;
        if (!scaleCache.has(name)) scaleCache.set(name, computeProbeScale(probe, lengths));
        return scaleCache.get(name);
      }
      function getErrScaleFor(name, probe, lengths){
        if (!elLockScale.checked) return null;
        const key = name + '|err';
        if (!scaleCache.has(key)) scaleCache.set(key, computeProbeErrScale(probe, lengths));
        return scaleCache.get(key);
      }

      // scalars first (graph-level)
      for (const name of scalarNames){
        const probe = probesMap[name];
        const lengths = Array.isArray(probe.lengths) ? probe.lengths : globalLengths;

        const card = buildCard(name, probe, true);
        elCardsContainer.appendChild(card);

        card._sTrue.textContent = `shape=${JSON.stringify(getDims(probe.true))}`;
        card._sPred.textContent = `shape=${JSON.stringify(getDims(probe.pred))}`;
        const typeStr = String(probe.type || '').toLowerCase();
        const isCat = isCategoricalType(probe.type);
        const isScalar = typeStr === 'scalar';
        const colorFn = isScalar ? probToBlue : probToGray;
        card._sErr.textContent  = isCat ? 'mismatch' : '|pred-true|';

        // correctness plot (same logic)
        if (hasCorrectnessSeries(probe) && correctnessState.get(name)){
          card._corrWrap.style.display = 'block';
          try { drawCorrectnessGraph(card._corrCanvas, probe.correctness_along_time, tIdx, bIdx); } catch(_e) {}
        } else {
          card._corrWrap.style.display = 'none';
        }

        let sTrue=null, sPred=null;
        try { sTrue = makeTimeScalarViewWithLengths(probe.true, lengths); } catch(_e) {}
        try { sPred = makeTimeScalarViewWithLengths(probe.pred, lengths); } catch(_e) {}

        if (!(sTrue && sPred)){
          // fallback: render as 1x1 matrix from the current timestep
          const frameT = probe.true?.[bIdx]?.[tIdx] ?? probe.true?.[0]?.[tIdx] ?? probe.true;
          const frameP = probe.pred?.[bIdx]?.[tIdx] ?? probe.pred?.[0]?.[tIdx] ?? probe.pred;
          const tv = Number(Array.isArray(frameT) ? frameT[0] : frameT);
          const pv = Number(Array.isArray(frameP) ? frameP[0] : frameP);
          const scale = {bothIn01:true,min:0,max:1};
          const tvSafe = Number.isFinite(tv) ? tv : 0;
          const pvSafe = Number.isFinite(pv) ? pv : 0;
          const evSafe = Math.abs(pvSafe - tvSafe);

          drawMatrixTo(card._cTrue.getContext('2d'), card._cTrue, 1, 1, ()=>tvSafe, scale, 40, contrast, colorFn);
          drawMatrixTo(card._cPred.getContext('2d'), card._cPred, 1, 1, ()=>pvSafe, scale, 40, contrast, colorFn);
          drawMatrixTo(card._cErr.getContext('2d'),  card._cErr,  1, 1, ()=>evSafe, {bothIn01:true,min:0,max:1}, 40, contrast, colorFn);

          // Hover support for fallback scalar rendering
          card._cTrue._hoverGetter = ()=>({type:'matrix',which:'true',t:tIdx,H:1,W:1,cell:40,get:()=>tvSafe});
          card._cPred._hoverGetter = ()=>({type:'matrix',which:'pred',t:tIdx,H:1,W:1,cell:40,get:()=>pvSafe});
          card._cErr._hoverGetter  = ()=>({type:'matrix',which:'err', t:tIdx,H:1,W:1,cell:40,get:()=>evSafe});
          continue;
        }

        // Render as a length-1 vector so it still uses the vector layout.
        const tv = Number(sTrue.get(tIdx));
        const pv = Number(sPred.get(tIdx));
        const rawTrue = [tv];
        const rawPred = [pv];

        let rawErr;
        if (isCat){
          const lt = Math.round(tv);
          const lp = Math.round(pv);
          rawErr = [(lt === lp) ? 0 : 1];
        } else {
          rawErr = [Math.abs(pv - tv)];
        }

        const scale = {bothIn01:false,min:Math.min(tv,pv),max:Math.max(tv,pv)};
        drawVectorTo(card._cTrue.getContext('2d'), card._cTrue, rawTrue, scale, 60, contrast, colorFn);
        drawVectorTo(card._cPred.getContext('2d'), card._cPred, rawPred, scale, 60, contrast, colorFn);
        drawVectorTo(card._cErr.getContext('2d'),  card._cErr,  rawErr,  {bothIn01:true,min:0,max:1}, 60, contrast, colorFn);

        card._cTrue._hoverGetter = ()=>({type:'vector',which:'true',t:tIdx,N:1,cell:60,get:()=>tv});
        card._cPred._hoverGetter = ()=>({type:'vector',which:'pred',t:tIdx,N:1,cell:60,get:()=>pv});
        card._cErr._hoverGetter  = ()=>({type:'vector',which:'err', t:tIdx,N:1,cell:60,get:()=>rawErr[0]});
      }

      // vectors next
      for (const name of vectorNames){
        const probe = probesMap[name];
        const lengths = Array.isArray(probe.lengths) ? probe.lengths : globalLengths;

        const card = buildCard(name, probe, true);
        elCardsContainer.appendChild(card);

        card._sTrue.textContent = `shape=${JSON.stringify(getDims(probe.true))}`;
        card._sPred.textContent = `shape=${JSON.stringify(getDims(probe.pred))}`;
        card._sErr.textContent  = `|pred-true|`;

        // --- Hint correctness graph (optional; render early so no branch can skip it) ---
        if (hasCorrectnessSeries(probe) && correctnessState.get(name)){
          card._corrWrap.style.display = 'block';
          try {
            drawCorrectnessGraph(card._corrCanvas, probe.correctness_along_time, tIdx, bIdx);
          } catch (_e){
            // ignore plot errors
          }
        } else {
          card._corrWrap.style.display = 'none';
        }

        if (probe.error){
          const ctx = card._cTrue.getContext('2d');
          card._cTrue.width=520; card._cTrue.height=40;
          ctx.font='12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
          ctx.fillStyle='#333';
          ctx.fillText(`Error: ${probe.error}`, 6, 24);
          card._cPred.width=1; card._cPred.height=1;
          card._cErr.width=1; card._cErr.height=1;
          continue;
        }

        const dimsTrue = getDims(probe.true);
        const isCatVec = isCategoricalType(probe.type) && dimsTrue.length === 4 && dimsTrue[3] <= 64;

        if (isCatVec){
          let vt=null, vp=null;
          try { vt = makeTimeCategoricalVectorView(probe.true, lengths); } catch(_) {}
          try { vp = makeTimeCategoricalVectorView(probe.pred, lengths); } catch(_) {}

          if (!(vt && vp)){
            const frameT = probe.true?.[0]?.[tIdx] ?? probe.true;
            const frameP = probe.pred?.[0]?.[tIdx] ?? probe.pred;
            const fT = flattenFrameTo2D(frameT);
            const fP = flattenFrameTo2D(frameP);
            const scale = {bothIn01:true,min:0,max:1};
            drawMatrixTo(card._cTrue.getContext('2d'), card._cTrue, fT.H, fT.W, (i,j)=>fT.get(i,j), scale, mcell, contrast);
            drawMatrixTo(card._cPred.getContext('2d'), card._cPred, fP.H, fP.W, (i,j)=>fP.get(i,j), scale, mcell, contrast);
            drawMatrixTo(card._cErr.getContext('2d'),  card._cErr,  fT.H, fT.W, (i,j)=>Math.abs(fP.get(i,j)-fT.get(i,j)), {bothIn01:true,min:0,max:1}, mcell, contrast);
            continue;
          }

          const N = Math.min(vt.N, vp.N);
          const K = Math.min(vt.K, vp.K);

          const labelsTrue = new Array(N);
          const labelsPred = new Array(N);
          const mismatch = new Array(N);

          for (let j=0;j<N;j++){
            const vecT = new Array(K);
            const vecP = new Array(K);
            for (let k=0;k<K;k++){
              vecT[k] = Number(vt.get(tIdx,j,k));
              vecP[k] = Number(vp.get(tIdx,j,k));
            }
            const lt = argmax1D(vecT);
            const lp = argmax1D(vecP);
            labelsTrue[j] = lt;
            labelsPred[j] = lp;
            mismatch[j] = (lt === lp) ? 0 : 1;
          }

          drawCategoricalVector(card._cTrue.getContext('2d'), card._cTrue, labelsTrue, vcell);
          drawCategoricalVector(card._cPred.getContext('2d'), card._cPred, labelsPred, vcell);
          drawVectorTo(card._cErr.getContext('2d'), card._cErr, mismatch, {bothIn01:true,min:0,max:1}, vcell, contrast);

          card._cTrue._hoverGetter = ()=>({type:'vector',which:'true',t:tIdx,N:1,cell:vcell,get:(j)=>labelsTrue[j]});
          card._cPred._hoverGetter = ()=>({type:'vector',which:'pred',t:tIdx,N:1,cell:vcell,get:(j)=>labelsPred[j]});
          card._cErr._hoverGetter  = ()=>({type:'vector',which:'err', t:tIdx,N:1,cell:vcell,get:(j)=>mismatch[j]});

          continue;
        }

        // numeric vector
        let vTrue=null,vPred=null;
        try { vTrue = makeTimeVectorViewWithLengths(probe.true, lengths); } catch(_) {}
        try { vPred = makeTimeVectorViewWithLengths(probe.pred, lengths); } catch(_) {}

        if (!(vTrue && vPred)){
          const frameT = probe.true?.[0]?.[tIdx] ?? probe.true;
          const frameP = probe.pred?.[0]?.[tIdx] ?? probe.pred;
          const fT = flattenFrameTo2D(frameT);
          const fP = flattenFrameTo2D(frameP);
          const scale = {bothIn01:true,min:0,max:1};
          const colorFn = isScalarType(probe.type) ? probToBlue : probToGray;
          drawMatrixTo(card._cTrue.getContext('2d'), card._cTrue, fT.H, fT.W, (i,j)=>fT.get(i,j), scale, mcell, contrast, colorFn);
          drawMatrixTo(card._cPred.getContext('2d'), card._cPred, fP.H, fP.W, (i,j)=>fP.get(i,j), scale, mcell, contrast, colorFn);
          drawMatrixTo(card._cErr.getContext('2d'),  card._cErr,  fT.H, fT.W, (i,j)=>Math.abs(fP.get(i,j)-fT.get(i,j)), {bothIn01:true,min:0,max:1}, mcell, contrast, colorFn);
          continue;
        }

        const N = Math.min(vTrue.N, vPred.N);
        const rawTrue = new Array(N);
        const rawPred = new Array(N);
        for (let j=0;j<N;j++){
          rawTrue[j] = Number(vTrue.get(tIdx, j));
          rawPred[j] = Number(vPred.get(tIdx, j));
        }

        const locked = getScaleFor(name, probe, lengths);
        let scale = {bothIn01:true,min:0,max:1};
        if (locked) scale = locked;
        else {
          const tmin = Math.min(...rawTrue), tmax = Math.max(...rawTrue);
          const pmin = Math.min(...rawPred), pmax = Math.max(...rawPred);
          const bothIn01 = (tmin>=0 && tmax<=1 && pmin>=0 && pmax<=1);
          scale = bothIn01 ? {bothIn01:true,min:0,max:1} : {bothIn01:false,min:Math.min(tmin,pmin),max:Math.max(tmax,pmax)};
        }

        const colorFn = isScalarType(probe.type) ? probToBlue : probToGray;
        drawVectorTo(card._cTrue.getContext('2d'), card._cTrue, rawTrue, scale, vcell, contrast, colorFn);
        drawVectorTo(card._cPred.getContext('2d'), card._cPred, rawPred, scale, vcell, contrast, colorFn);

        const errLocked = getErrScaleFor(name, probe, lengths);
        const typeStr = String(probe.type || '').toLowerCase();
        const isPointer = (typeStr === 'pointer');
        const isMaskOne = (typeStr === 'mask_one');

        // Preserve discrete mismatch behavior for POINTER/MASK_ONE vectors.
        let rawErr;
        if (isPointer) {
          rawErr = rawPred.map((v,idx)=>{
            const tp = rawTrue[idx];
            const pp = v;
            const lt = looksLikeIndexScalar(tp) ? Math.round(tp) : Math.round(tp);
            const lp = looksLikeIndexScalar(pp) ? Math.round(pp) : Math.round(pp);
            return (lt === lp) ? 0 : 1;
          });
          card._sErr.textContent = 'mismatch';
        } else if (isMaskOne) {
          rawErr = rawPred.map((v,idx)=>{
            const lt = decodeBinaryMask(rawTrue[idx]);
            const lp = decodeBinaryMask(v);
            return (lt === lp) ? 0 : 1;
          });
          card._sErr.textContent = 'mismatch';
        } else {
          rawErr = rawPred.map((v,idx)=>Math.abs(v - rawTrue[idx]));
        }

        let errScale = {bothIn01:true,min:0,max:1};
        if (errLocked) errScale = errLocked;
        else {
          const emax = Math.max(...rawErr, 0);
          errScale = (emax <= 1) ? {bothIn01:true,min:0,max:1} : {bothIn01:false,min:0,max:Math.max(emax,1e-6)};
        }

        drawVectorTo(card._cErr.getContext('2d'), card._cErr, rawErr, errScale, vcell, contrast, colorFn);

        card._cTrue._hoverGetter = ()=>({type:'vector',which:'true',t:tIdx,N,cell:vcell,get:(j)=>rawTrue[j]});
        card._cPred._hoverGetter = ()=>({type:'vector',which:'pred',t:tIdx,N,cell:vcell,get:(j)=>rawPred[j]});
        card._cErr._hoverGetter  = ()=>({type:'vector',which:'err', t:tIdx,N,cell:vcell,get:(j)=>rawErr[j]});

        // (correctness graph already handled above)
      }

      // matrices second
      for (const name of matrixNames){
        const probe = probesMap[name];
        const lengths = Array.isArray(probe.lengths) ? probe.lengths : globalLengths;

        const card = buildCard(name, probe, false);
        elCardsContainer.appendChild(card);

        card._sTrue.textContent = `shape=${JSON.stringify(getDims(probe.true))}`;
        card._sPred.textContent = `shape=${JSON.stringify(getDims(probe.pred))}`;
        card._sErr.textContent  = `|pred-true|`;

        // --- Hint correctness graph (optional; render early so no branch can skip it) ---
        if (hasCorrectnessSeries(probe) && correctnessState.get(name)){
          card._corrWrap.style.display = 'block';
          try {
            drawCorrectnessGraph(card._corrCanvas, probe.correctness_along_time, tIdx, bIdx);
          } catch (_e){
            // ignore plot errors
          }
        } else {
          card._corrWrap.style.display = 'none';
        }

        if (probe.error){
          const ctx = card._cTrue.getContext('2d');
          card._cTrue.width=520; card._cTrue.height=40;
          ctx.font='12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
          ctx.fillStyle='#333';
          ctx.fillText(`Error: ${probe.error}`, 6, 24);
          card._cPred.width=1; card._cPred.height=1;
          card._cErr.width=1; card._cErr.height=1;
          continue;
        }

        let r0 = Math.max(0, parseInt(elR0.value||'0',10));
        let r1 = Math.max(0, parseInt(elR1.value||'0',10));
        let c0 = Math.max(0, parseInt(elC0.value||'0',10));
        let c1 = Math.max(0, parseInt(elC1.value||'0',10));

        const dimsTrue = getDims(probe.true);
        const isCatMat = isCategoricalType(probe.type) && dimsTrue.length === 5 && dimsTrue[4] <= 64;

        const t0 = tIdx;

        if (isCatMat){
          let mt=null, mp = null;
          try { mt = makeTimeCategoricalMatrixView(probe.true, lengths); } catch(_) {}
          try { mp = makeTimeCategoricalMatrixView(probe.pred, lengths); } catch(_) {}

          if (!(mt && mp)){
            const frameT = probe.true?.[0]?.[t0] ?? probe.true;
            const frameP = probe.pred?.[0]?.[t0] ?? probe.pred;
            const fT = flattenFrameTo2D(frameT);
            const fP = flattenFrameTo2D(frameP);
            const scale = {bothIn01:true,min:0,max:1};
            drawMatrixTo(card._cTrue.getContext('2d'), card._cTrue, fT.H, fT.W, (i,j)=>fT.get(i,j), scale, mcell, contrast);
            drawMatrixTo(card._cPred.getContext('2d'), card._cPred, fP.H, fP.W, (i,j)=>fP.get(i,j), scale, mcell, contrast);
            drawMatrixTo(card._cErr.getContext('2d'),  card._cErr,  fT.H, fT.W, (i,j)=>Math.abs(fP.get(i,j)-fT.get(i,j)), {bothIn01:true,min:0,max:1}, mcell, contrast);
            continue;
          }

          const H = Math.min(mt.H, mp.H);
          const W = Math.min(mt.W, mp.W);
          const K = Math.min(mt.K, mp.K);

          if (!Number.isFinite(r0)) r0 = 0; if (!Number.isFinite(c0)) c0 = 0;
          if (!Number.isFinite(r1) || r1===0) r1 = H; if (!Number.isFinite(c1) || c1===0) c1 = W;
          r0 = Math.min(r0, H); r1 = Math.min(Math.max(r1, r0), H);
          c0 = Math.min(c0, W); c1 = Math.min(Math.max(c1, c0+0), W);
          const HH = Math.max(0, r1 - r0);
          const WW = Math.max(0, c1 - c0);

          const labelsTrue = new Array(HH);
          const labelsPred = new Array(HH);
          const mismatch = new Array(HH);
          for (let i=0;i<HH;i++){
            labelsTrue[i] = new Array(WW);
            labelsPred[i] = new Array(WW);
            mismatch[i] = new Array(WW);
            for (let j=0;j<WW;j++){
              let bestT=0, bestTv=Number(mt.get(t0,r0+i,c0+j,0));
              let bestP=0, bestPv=Number(mp.get(t0,r0+i,c0+j,0));
              for (let k=1;k<K;k++){
                const tv = Number(mt.get(t0,r0+i,c0+j,k));
                if (tv > bestTv) { bestTv = tv; bestT = k; }
                const pv = Number(mp.get(t0,r0+i,c0+j,k));
                if (pv > bestPv) { bestPv = pv; bestP = k; }
              }
              labelsTrue[i][j] = bestT;
              labelsPred[i][j] = bestP;
              mismatch[i][j] = (bestT === bestP) ? 0 : 1;
            }
          }

          // BEFORE: drawCategoricalMatrix(ctx, canvas, labelsTrue, HH, WW, mcell)
          // draw using getLabel callback to match drawCategoricalMatrix signature
          drawCategoricalMatrix(card._cTrue.getContext('2d'), card._cTrue, HH, WW, (i,j)=>labelsTrue[i][j], mcell);
          drawCategoricalMatrix(card._cPred.getContext('2d'), card._cPred, HH, WW, (i,j)=>labelsPred[i][j], mcell);
          drawMatrixTo(card._cErr.getContext('2d'), card._cErr, HH, WW, (i,j)=>mismatch[i][j], {bothIn01:true,min:0,max:1}, mcell, contrast);

          card._cTrue._hoverGetter = ()=>({type:'matrix',which:'true',t:t0,H:HH,W:WW,cell:mcell,get:(i,j)=>labelsTrue[i][j]});
          card._cPred._hoverGetter = ()=>({type:'matrix',which:'pred',t:t0,H:HH,W:WW,cell:mcell,get:(i,j)=>labelsPred[i][j]});
          card._cErr._hoverGetter  = ()=>({type:'matrix',which:'err', t:t0,H:HH,W:WW,cell:mcell,get:(i,j)=>mismatch[i][j]});

          continue;
        }

        // numeric matrix
        let mTrue=null,mPred=null;
        try { mTrue = makeTimeMatrixViewWithLengths(probe.true, lengths); } catch(_) {}
        try { mPred = makeTimeMatrixViewWithLengths(probe.pred, lengths); } catch(_) {}

        if (!(mTrue && mPred)){
          const frameT = probe.true?.[0]?.[t0] ?? probe.true;
          const frameP = probe.pred?.[0]?.[t0] ?? probe.pred;
          const fT = flattenFrameTo2D(frameT);
          const fP = flattenFrameTo2D(frameP);
          const scale = {bothIn01:true,min:0,max:1};
          const colorFn = isScalarType(probe.type) ? probToBlue : probToGray;
          drawMatrixTo(card._cTrue.getContext('2d'), card._cTrue, fT.H, fT.W, (i,j)=>fT.get(i,j), scale, mcell, contrast, colorFn);
          drawMatrixTo(card._cPred.getContext('2d'), card._cPred, fP.H, fP.W, (i,j)=>fP.get(i,j), scale, mcell, contrast, colorFn);
          drawMatrixTo(card._cErr.getContext('2d'),  card._cErr,  fT.H, fT.W, (i,j)=>Math.abs(fP.get(i,j)-fT.get(i,j)), {bothIn01:true,min:0,max:1}, mcell, contrast, colorFn);
          continue;
        }

        const H = Math.min(mTrue.H, mPred.H);
        const W = Math.min(mTrue.W, mPred.W);

        if (!Number.isFinite(r0)) r0 = 0; if (!Number.isFinite(c0)) c0 = 0;
        if (!Number.isFinite(r1) || r1===0) r1 = H; if (!Number.isFinite(c1) || c1===0) c1 = W;
        r0 = Math.min(r0, H); r1 = Math.min(Math.max(r1, r0), H);
        c0 = Math.min(c0, W); c1 = Math.min(Math.max(c1, c0), W);
        const HH = Math.max(0, r1 - r0);
        const WW = Math.max(0, c1 - c0);

        const locked = getScaleFor(name, probe, lengths);
        let scale = {bothIn01:true,min:0,max:1};
        if (locked) scale = locked;
        else {
          let valsTrue = [];
          let valsPred = [];
          for (let i=r0;i<r1;i++){
            for (let j=c0;j<c1;j++){
              valsTrue.push(Number(mTrue.get(t0,i,j)));
              valsPred.push(Number(mPred.get(t0,i,j)));
            }
          }
          const tmin = Math.min(...valsTrue), tmax = Math.max(...valsTrue);
          const pmin = Math.min(...valsPred), pmax = Math.max(...valsPred);
          const bothIn01 = (tmin>=0 && tmax<=1 && pmin>=0 && pmax<=1);
          scale = bothIn01 ? {bothIn01:true,min:0,max:1} : {bothIn01:false,min:Math.min(tmin,pmin),max:Math.max(tmax,pmax)};
        }

        const colorFn = isScalarType(probe.type) ? probToBlue : probToGray;
        drawMatrixTo(card._cTrue.getContext('2d'), card._cTrue, HH, WW, (i,j)=>mTrue.get(t0,r0+i,c0+j), scale, mcell, contrast, colorFn);
        drawMatrixTo(card._cPred.getContext('2d'), card._cPred, HH, WW, (i,j)=>mPred.get(t0,r0+i,c0+j), scale, mcell, contrast, colorFn);

        const errLocked = getErrScaleFor(name, probe, lengths);
        let errScale = {bothIn01:true,min:0,max:1};
        if (errLocked) errScale = errLocked;
        else {
          let emax = 0;
          for (let i=0;i<HH;i++){
            for (let j=0;j<WW;j++){
              const e = Math.abs(Number(mPred.get(t0,r0+i,c0+j)) - Number(mTrue.get(t0,r0+i,c0+j)));
              if (Number.isFinite(e) && e > emax) emax = e;
            }
          }

          errScale = (emax <= 1) ? {bothIn01:true,min:0,max:1} : {bothIn01:false,min:0,max:Math.max(emax,1e-6)};
        }

        const typeStr = String(probe.type || '').toLowerCase();
        const isPointer = (typeStr === 'pointer');
        const isMaskOne = (typeStr === 'mask_one');

        if (isPointer) {
          card._sErr.textContent = 'mismatch';
          const errGetter = (i,j)=>{
            const tv = Number(mTrue.get(t0,r0+i,c0+j));
            const pv = Number(mPred.get(t0,r0+i,c0+j));
            if (!Number.isFinite(tv) || !Number.isFinite(pv)) return 0;
            const lt = looksLikeIndexScalar(tv) ? Math.round(tv) : Math.round(tv);
            const lp = looksLikeIndexScalar(pv) ? Math.round(pv) : Math.round(pv);
            return (lt === lp) ? 0 : 1;
          };
          drawMatrixTo(card._cErr.getContext('2d'),  card._cErr,  HH, WW, errGetter, {bothIn01:true,min:0,max:1}, mcell, contrast, colorFn);
          card._cErr._hoverGetter  = ()=>({type:'matrix',which:'err', t:t0,H:HH,W:WW,cell:mcell,get:(i,j)=>errGetter(i,j)});
          card._cTrue._hoverGetter = ()=>({type:'matrix',which:'true',t:t0,H:HH,W:WW,cell:mcell,get:(i,j)=>Number(mTrue.get(t0,r0+i,c0+j))});
          card._cPred._hoverGetter = ()=>({type:'matrix',which:'pred',t:t0,H:HH,W:WW,cell:mcell,get:(i,j)=>Number(mPred.get(t0,r0+i,c0+j))});
          continue;
        }

        if (isMaskOne) {
          card._sErr.textContent = 'mismatch';
          const errGetter = (i,j)=>{
            const tv = Number(mTrue.get(t0,r0+i,c0+j));
            const pv = Number(mPred.get(t0,r0+i,c0+j));
            if (!Number.isFinite(tv) || !Number.isFinite(pv)) return 0;
            const lt = decodeBinaryMask(tv);
            const lp = decodeBinaryMask(pv);
            return (lt === lp) ? 0 : 1;
          };
          drawMatrixTo(card._cErr.getContext('2d'),  card._cErr,  HH, WW, errGetter, {bothIn01:true,min:0,max:1}, mcell, contrast, colorFn);
          card._cErr._hoverGetter  = ()=>({type:'matrix',which:'err', t:t0,H:HH,W:WW,cell:mcell,get:(i,j)=>errGetter(i,j)});
          card._cTrue._hoverGetter = ()=>({type:'matrix',which:'true',t:t0,H:HH,W:WW,cell:mcell,get:(i,j)=>Number(mTrue.get(t0,r0+i,c0+j))});
          card._cPred._hoverGetter = ()=>({type:'matrix',which:'pred',t:t0,H:HH,W:WW,cell:mcell,get:(i,j)=>Number(mPred.get(t0,r0+i,c0+j))});
          continue;
        }

        // Default numeric error: |pred-true|
        drawMatrixTo(card._cErr.getContext('2d'),  card._cErr,  HH, WW,
          (i,j)=>Math.abs(Number(mPred.get(t0,r0+i,c0+j)) - Number(mTrue.get(t0,r0+i,c0+j))),
          errScale, mcell, contrast, colorFn);
        card._cErr._hoverGetter  = ()=>({type:'matrix',which:'err', t:t0,H:HH,W:WW,cell:mcell,get:(i,j)=>Math.abs(Number(mPred.get(t0,r0+i,c0+j)) - Number(mTrue.get(t0,r0+i,c0+j)))});
        card._cTrue._hoverGetter = ()=>({type:'matrix',which:'true',t:t0,H:HH,W:WW,cell:mcell,get:(i,j)=>Number(mTrue.get(t0,r0+i,c0+j))});
        card._cPred._hoverGetter = ()=>({type:'matrix',which:'pred',t:t0,H:HH,W:WW,cell:mcell,get:(i,j)=>Number(mPred.get(t0,r0+i,c0+j))});
      }

      elSummary.textContent = `selected=${selectedNames.length} • vectors=${vectorNames.length} • matrices=${matrixNames.length}`;

      // call PCA renders only if UI exists
      if (HAS_PCA_UI) {
        renderPCA();
        renderPcaNodes();
      }
      if (tCapFromIsLast != null) {
        elMeta.textContent = `Using per-batch time from is_last: T_valid=${tCapFromIsLast} (batch ${bIdx}).`;
      } else if (tCapFromLengths != null) {
        elMeta.textContent = `Using per-batch time from lengths: T_valid=${tCapFromLengths} (batch ${bIdx}).`;
      } else {
        elMeta.textContent = `Tip: export algo.is_last to avoid viewing padded timesteps.`;
      }
    }

    // PCA rendering
    function renderPCA_(){
      try {
        const hs = algo && algo.gnn_hidden_states && algo.gnn_hidden_states.data;
        // Accept both nested JS arrays (from JSON) and typed-array-like objects.
        const hasHS = Array.isArray(hs) || (hs && typeof hs === 'object' && typeof hs.length === 'number');
        elShowPca.disabled = !hasHS;
        if (!elShowPca.checked || !hasHS){
          elPcaContainer.style.display = 'none';
          elPcaInfo.textContent = hasHS ? 'PCA is available. Enable the checkbox to view.' : 'Hidden states not found in JSON.';
          return;
        }

        // Ensure layout size has stabilized before measuring canvas (prevents initial weird scaling)
        const cssW = elPcaCanvas.clientWidth || 600;
        if (cssW <= 1) {
          requestAnimationFrame(renderPCA);
          return;
        }

        const bIdx = Math.max(0, Number(elBidx.value) ||  0);
        let Tvalid = validTFromIsLast(bIdx);
        if (Tvalid == null) {
          const Ls = Array.isArray(algo?.lengths) ? algo.lengths : null;
          if (Ls && Ls.length > bIdx) {
            const L = Number(Ls[bIdx]);
            if (Number.isFinite(L) && L > 0) Tvalid = L;
          }
        }
        if (Tvalid == null) {
          try { Tvalid = hs[Math.min(bIdx, hs.length-1)].length; } catch(_) { Tvalid = 1; }
        }

        const tVectors = PCA.flattenBTNDToTimeVectors(hs, bIdx, Tvalid);
        const T = tVectors.length;
        if (T === 0){
          elPcaContainer.style.display = 'none';
          elPcaInfo.textContent = 'No timesteps to show.';
          return;
        }

        const K = Math.min(10, T);
        const pca = PCA.fit(tVectors, K);

        const pcx = Math.max(1, Number(elPcx.value) || 1);
        const pcy = Math.max(1, Number(elPcy.value) || 2);
        const coords = PCA.projectXY(pca, pcx, pcy);

        // Cache bounds based on all time points (NOT changing with current time)
        const boundsKey = `b=${bIdx}|T=${T}|pcx=${pcx}|pcy=${pcy}`;
        if (pcaBoundsCache.key !== boundsKey) {
          pcaBoundsCache.bounds = computeBoundsFromPoints(coords);
          pcaBoundsCache.key = boundsKey;
        }
        const bnd = pcaBoundsCache.bounds;
        if (!bnd) {
          elPcaContainer.style.display = 'none';
          elPcaInfo.textContent = 'PCA produced invalid coordinates.';
          return;
        }
        let { xmin, xmax, ymin, ymax } = bnd;

        const dpr = window.devicePixelRatio || 1;
        const cssH = elPcaCanvas.clientHeight || 320;
        elPcaCanvas.width = Math.max(1, Math.floor(cssW * dpr));
        elPcaCanvas.height = Math.max(1, Math.floor(cssH * dpr));
        const ctx = elPcaCanvas.getContext('2d');
        ctx.setTransform(dpr,0,0,dpr,0,0);
        ctx.clearRect(0,0,cssW,cssH);

        function sx(x){ return (x - xmin) / (xmax - xmin) * (cssW - 40) + 20; }
        function sy(y){ return (1 - (y - ymin) / (ymax - ymin)) * (cssH - 48) + 16; }

        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        ctx.strokeRect(10.5, 10.5, cssW-21, cssH-33);
        drawAxes(ctx, cssW, cssH, xmin, xmax, ymin, ymax);
        drawAxisLines(ctx, cssW, cssH, sx, sy);

        ctx.strokeStyle = '#4E79A7';
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started=false;
        for (let i=0;i<coords.length;i++){
          const p = coords[i];
          const X = sx(p.x), Y = sy(p.y);
          if (!started){ ctx.moveTo(X,Y); started=true; }
          else ctx.lineTo(X,Y);
        }
        ctx.stroke();

        const curT = Math.max(0, Math.min(T-1, Number(elT.value) || 0));
        for (let i=0;i<coords.length;i++){
          const p = coords[i];
          const X = sx(p.x), Y = sy(p.y);
          ctx.beginPath();
          ctx.arc(X, Y, (i===0 || i===T-1) ? 4 : 3, 0, Math.PI*2);
          if (i === curT) {
            ctx.fillStyle = '#E15759';
            ctx.fill();
          } else {
            ctx.fillStyle = (i===0) ? '#59A14F' : (i===T-1 ? '#B07AA1' : '#4E79A7');
            ctx.fill();
          }
        }

        elPcaContainer.style.display = '';
        const evx = pca.explained[Math.max(0, Math.min(pca.explained.length-1, pcx-1))] || 0;
        const evy = pca.explained[Math.max(0, Math.min(pca.explained.length-1, pcy-1))] || 0;
        elPcaInfo.textContent = `T=${T}, PCs available=${pca.K}. Var[X]=${(evx*100).toFixed(1)}%, Var[Y]=${(evy*100).toFixed(1)}%`;
        elPcaTitle.textContent = `PCA trajectory • PC${pcx} vs PC${pcy}`;
      } catch (e){
        elPcaContainer.style.display = 'none';
        elPcaInfo.textContent = `PCA error: ${e?.message || e}`;
      }
    }

    function renderPcaNodes_(){
      try {
        const hs = algo && algo.gnn_hidden_states && algo.gnn_hidden_states.data;
        const hasHS = Array.isArray(hs) || (hs && typeof hs === 'object' && typeof hs.length === 'number');
        if (!hasHS){
          elShowPcaNodes.disabled = true;
          elPcaNodesContainer.style.display = 'none';
          elPcaNodesInfo.textContent = 'Hidden states not found in JSON.';
          return;
        } else {
          elShowPcaNodes.disabled = false;
        }
        if (!elShowPcaNodes.checked){
          elPcaNodesContainer.style.display = 'none';
          return;
        }

        const cssW = elPcaNodesCanvas.clientWidth || 600;
        if (cssW <= 1) {
          requestAnimationFrame(renderPCANodes);
          return;
        }

        const bIdx = Math.max(0, Number(elBidx.value) || 0);
        let Tvalid = validTFromIsLast(bIdx);
        if (Tvalid == null) {
          const Ls = Array.isArray(algo?.lengths) ? algo.lengths : null;
          if (Ls && Ls.length > bIdx) {
            const L = Number(Ls[bIdx]);
            if (Number.isFinite(L) && L > 0) Tvalid = L;
          }
        }
        if (Tvalid == null) {
          try { Tvalid = hs[Math.min(bIdx, hs.length-1)].length; } catch(_) { Tvalid = 1; }
        }

        const bt = hs[Math.min(bIdx, hs.length-1)] || [];
        const first = bt[0] || [];
        const N = Array.isArray(first) ? first.length : 0;
        const firstND = (N>0) ? first[0] : [];
        const D = Array.isArray(firstND) ? firstND.length : 1;

        const algoName = algo && algo.algorithm ? String(algo.algorithm) : 'algo';
        const cacheKey = `${algoName}|b=${bIdx}|T=${Tvalid}|N=${N}|D=${D}`;
        if (nodePcaCache.key !== cacheKey){
          let samples;
          try {
            samples = PCA.flattenBTNDToNodeSamples(hs, bIdx, Tvalid);
          } catch(e){
            elPcaNodesContainer.style.display = 'none';
            elPcaNodesInfo.textContent = 'Failed to build node samples for PCA.';
            return;
          }
          const K = Math.min(10, Math.min(D, samples.length));
          try {
            nodePcaCache.pca = PCA.fit(samples, K);
            nodePcaCache.key = cacheKey;
            // Also reset bounds cache when PCA basis changes (new batch / sizes)
            pcaNodesBoundsCache.key = null;
            pcaNodesBoundsCache.bounds = null;
          } catch(e){
            elPcaNodesContainer.style.display = 'none';
            elPcaNodesInfo.textContent = 'PCA fitting failed.';
            return;
          }
        }

        const pca = nodePcaCache.pca;
        if (!pca){
          elPcaNodesContainer.style.display = 'none';
          elPcaNodesInfo.textContent = 'PCA basis not available.';
          return;
        }

        const pcx = Math.max(1, Number(elPcxNodes.value) || 1);
        const pcy = Math.max(1, Number(elPcyNodes.value) || 2);

        // Bounds are computed once from ALL nodes across ALL times (basis is already from all times)
        const boundsKey = `b=${bIdx}|T=${Tvalid}|pcx=${pcx}|pcy=${pcy}|N=${N}|D=${D}`;
        if (pcaNodesBoundsCache.key !== boundsKey) {
          let samples;
          try {
            samples = PCA.flattenBTNDToNodeSamples(hs, bIdx, Tvalid);
          } catch(_e) {
            samples = null;
          }
          if (samples && samples.length) {
            const allPts = PCA.projectPointsXY(pca, samples, pcx, pcy);
            pcaNodesBoundsCache.bounds = computeBoundsFromPoints(allPts);
            pcaNodesBoundsCache.key = boundsKey;
          }
        }
        const bnd = pcaNodesBoundsCache.bounds;
        if (!bnd) {
          elPcaNodesContainer.style.display = 'none';
          elPcaNodesInfo.textContent = 'PCA bounds not available.';
          return;
        }
        let { xmin, xmax, ymin, ymax } = bnd;

        const curT = Math.max(0, Math.min((Tvalid||1)-1, Number(elT.value) || 0));
        const nodes = (bt && bt[curT]) ? bt[curT] : [];
        const vectors = new Array(N);
        for (let n=0;n<N;n++){
          const h = nodes[n] || [];
          const row = new Float64Array(D);
          if (Array.isArray(h)){
            for (let d=0; d<D; d++) row[d] = Number(h[d]) || 0;
          } else {
            row[0] = Number(h) || 0;
          }
          vectors[n] = row;
        }

        const pts = PCA.projectPointsXY(pca, vectors, pcx, pcy);

        const dpr = window.devicePixelRatio || 1;
        const cssH = elPcaNodesCanvas.clientHeight || 320;
        elPcaNodesCanvas.width = Math.max(1, Math.floor(cssW * dpr));
        elPcaNodesCanvas.height = Math.max(1, Math.floor(cssH * dpr));
        const ctx = elPcaNodesCanvas.getContext('2d');
        ctx.setTransform(dpr,0,0,dpr,0,0);
        ctx.clearRect(0,0,cssW,cssH);

        function sx(x){ return (x - xmin) / (xmax - xmin) * (cssW - 40) + 20; }
        function sy(y){ return (1 - (y - ymin) / (ymax - ymin)) * (cssH - 48) + 16; }

        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        ctx.strokeRect(10.5, 10.5, cssW-21, cssH-33);
        drawAxes(ctx, cssW, cssH, xmin, xmax, ymin, ymax);
        drawAxisLines(ctx, cssW, cssH, sx, sy);

        const colorByIndex = !!elColorGraphNodes?.checked;
        for (let i=0;i<pts.length;i++){
          const p = pts[i];
          const X = sx(p.x), Y = sy(p.y);
          ctx.beginPath();
          ctx.arc(X, Y, 3, 0, Math.PI*2);
          ctx.fillStyle = colorByIndex ? nodeIndexToColor(i, pts.length) : '#4E79A7';
          ctx.fill();
        }

        elPcaNodesContainer.style.display = '';
        const evx = pca.explained[Math.max(0, Math.min(pca.explained.length-1, pcx-1))] || 0;
        const evy = pca.explained[Math.max(0, Math.min(pca.explained.length-1, pcy-1))] || 0;
        elPcaNodesInfo.textContent = `N=${N}, T_basis=${Tvalid}, PCs available=${pca.K}. Var[X]=${(evx*100).toFixed(1)}%, Var[Y]=${(evy*100).toFixed(1)}%`;
        elPcaNodesTitle.textContent = `PCA (nodes) • PC${pcx} vs PC${pcy}`;
      } catch (_e){
        elPcaNodesContainer.style.display = 'none';
        elPcaNodesInfo.textContent = 'Error rendering node-wise PCA.';
      }
    }

    // DEBUG (temporary): help diagnose missing pi_h corr plot
    // Logs once per render when corr is enabled for pi_h.
    function _dbgCorrRender(name, probe, tIdx, bIdx, card){
      try{
        if (name !== 'pi_h') return;
        if (!probe || !probe.correctness_along_time) return;
        const s = probe.correctness_along_time;
        const is2D = Array.isArray(s) && s.length>0 && Array.isArray(s[0]);
        const len1 = Array.isArray(s) ? s.length : null;
        const len2 = is2D ? (Array.isArray(s[0]) ? s[0].length : null) : null;
        const wrapDisp = card && card._corrWrap ? card._corrWrap.style.display : '<no wrap>';
        const canDisp = card && card._corrCanvas ? card._corrCanvas.style.display : '<no canvas>';
        console.log('[viz][corr] pi_h render', {stage: probe.stage, tIdx, bIdx, is2D, len1, len2, wrapDisp, canDisp});
      } catch (e) {
        // ignore
      }
    }

    // events
    btnPlay.addEventListener('click', play);
    btnPause.addEventListener('click', stop);
    btnStepBack?.addEventListener('click', () => stepTime(-1));
    btnStepForward?.addEventListener('click', () => stepTime(+1));
    btnAll.addEventListener('click', () => setAllProbes(true));
    btnNone.addEventListener('click', () => setAllProbes(false));

    elT.addEventListener('input', () => { detectAndRender(); });
    elBidx.addEventListener('change', () => { stop(); scaleCache.clear(); pcaBoundsCache.key=null; pcaBoundsCache.bounds=null; pcaNodesBoundsCache.key=null; pcaNodesBoundsCache.bounds=null; detectAndRender(); });
    // PCA controls events
    if (HAS_PCA_UI) {
      elShowPca.addEventListener('change', () => { detectAndRender(); });
      elPcx.addEventListener('change', () => { pcaBoundsCache.key=null; pcaBoundsCache.bounds=null; detectAndRender(); });
      elPcy.addEventListener('change', () => { pcaBoundsCache.key=null; pcaBoundsCache.bounds=null; detectAndRender(); });

      elShowPcaNodes.addEventListener('change', () => { detectAndRender(); });
      elPcxNodes.addEventListener('change', () => { pcaNodesBoundsCache.key=null; pcaNodesBoundsCache.bounds=null; detectAndRender(); });
      elPcyNodes.addEventListener('change', () => { pcaNodesBoundsCache.key=null; pcaNodesBoundsCache.bounds=null; detectAndRender(); });

      if (elColorGraphNodes) {
        elColorGraphNodes.addEventListener('change', () => { detectAndRender(); });
      }
    }

    elVCell.addEventListener('change', detectAndRender);
    elMCell.addEventListener('change', detectAndRender);
    elR0.addEventListener('input', detectAndRender);
    elR1.addEventListener('input', detectAndRender);
    elC0.addEventListener('input', detectAndRender);
    elC1.addEventListener('input', detectAndRender);

    elContrast.addEventListener('input', detectAndRender);
    elFps.addEventListener('change', () => { if (timer) play(); });

    elLockScale.addEventListener('change', () => { scaleCache.clear(); detectAndRender(); });

    window.addEventListener('resize', () => { detectAndRender(); });

    elFile.addEventListener('change', async () => {
      stop();
      const f = elFile.files?.[0];
      if (!f) return;
      const text = await f.text();
      let parsed = null;
      try { parsed = JSON.parse(text); } catch (_e) { alert('Invalid JSON'); return; }

      if (!(parsed && parsed.probes && typeof parsed.probes === 'object')) {
        alert('Expected multi-probe JSON: { "algorithm": "...", "probes": { ... }, "is_last": [[...]] }');
        return;
      }

      algo = parsed;

      selectedState.clear();
      scaleCache.clear();

      elT.value = "0";
      elBidx.value = "0";
      setHoverText("Hover any cell to see its value here.");

      detectAndRender();
    });

    // Rename original PCA rendering functions to avoid name conflicts with the stubs.
    function renderPCA_() {
      try {
        const hs = algo && algo.gnn_hidden_states && algo.gnn_hidden_states.data;
        // Accept both nested JS arrays (from JSON) and typed-array-like objects.
        const hasHS = Array.isArray(hs) || (hs && typeof hs === 'object' && typeof hs.length === 'number');
        elShowPca.disabled = !hasHS;
        if (!elShowPca.checked || !hasHS){
          elPcaContainer.style.display = 'none';
          elPcaInfo.textContent = hasHS ? 'PCA is available. Enable the checkbox to view.' : 'Hidden states not found in JSON.';
          return;
        }

        // Ensure layout size has stabilized before measuring canvas (prevents initial weird scaling)
        const cssW = elPcaCanvas.clientWidth || 600;
        if (cssW <= 1) {
          requestAnimationFrame(renderPCA);
          return;
        }

        const bIdx = Math.max(0, Number(elBidx.value) ||  0);
        let Tvalid = validTFromIsLast(bIdx);
        if (Tvalid == null) {
          const Ls = Array.isArray(algo?.lengths) ? algo.lengths : null;
          if (Ls && Ls.length > bIdx) {
            const L = Number(Ls[bIdx]);
            if (Number.isFinite(L) && L > 0) Tvalid = L;
          }
        }
        if (Tvalid == null) {
          try { Tvalid = hs[Math.min(bIdx, hs.length-1)].length; } catch(_) { Tvalid = 1; }
        }

        const tVectors = PCA.flattenBTNDToTimeVectors(hs, bIdx, Tvalid);
        const T = tVectors.length;
        if (T === 0){
          elPcaContainer.style.display = 'none';
          elPcaInfo.textContent = 'No timesteps to show.';
          return;
        }

        const K = Math.min(10, T);
        const pca = PCA.fit(tVectors, K);

        const pcx = Math.max(1, Number(elPcx.value) || 1);
        const pcy = Math.max(1, Number(elPcy.value) || 2);
        const coords = PCA.projectXY(pca, pcx, pcy);

        // Cache bounds based on all time points (NOT changing with current time)
        const boundsKey = `b=${bIdx}|T=${T}|pcx=${pcx}|pcy=${pcy}`;
        if (pcaBoundsCache.key !== boundsKey) {
          pcaBoundsCache.bounds = computeBoundsFromPoints(coords);
          pcaBoundsCache.key = boundsKey;
        }
        const bnd = pcaBoundsCache.bounds;
        if (!bnd) {
          elPcaContainer.style.display = 'none';
          elPcaInfo.textContent = 'PCA produced invalid coordinates.';
          return;
        }
        let { xmin, xmax, ymin, ymax } = bnd;

        const dpr = window.devicePixelRatio || 1;
        const cssH = elPcaCanvas.clientHeight || 320;
        elPcaCanvas.width = Math.max(1, Math.floor(cssW * dpr));
        elPcaCanvas.height = Math.max(1, Math.floor(cssH * dpr));
        const ctx = elPcaCanvas.getContext('2d');
        ctx.setTransform(dpr,0,0,dpr,0,0);
        ctx.clearRect(0,0,cssW,cssH);

        function sx(x){ return (x - xmin) / (xmax - xmin) * (cssW - 40) + 20; }
        function sy(y){ return (1 - (y - ymin) / (ymax - ymin)) * (cssH - 48) + 16; }

        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        ctx.strokeRect(10.5, 10.5, cssW-21, cssH-33);
        drawAxes(ctx, cssW, cssH, xmin, xmax, ymin, ymax);
        drawAxisLines(ctx, cssW, cssH, sx, sy);

        ctx.strokeStyle = '#4E79A7';
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started=false;
        for (let i=0;i<coords.length;i++){
          const p = coords[i];
          const X = sx(p.x), Y = sy(p.y);
          if (!started){ ctx.moveTo(X,Y); started=true; }
          else ctx.lineTo(X,Y);
        }
        ctx.stroke();

        const curT = Math.max(0, Math.min(T-1, Number(elT.value) || 0));
        for (let i=0;i<coords.length;i++){
          const p = coords[i];
          const X = sx(p.x), Y = sy(p.y);
          ctx.beginPath();
          ctx.arc(X, Y, (i===0 || i===T-1) ? 4 : 3, 0, Math.PI*2);
          if (i === curT) {
            ctx.fillStyle = '#E15759';
            ctx.fill();
          } else {
            ctx.fillStyle = (i===0) ? '#59A14F' : (i===T-1 ? '#B07AA1' : '#4E79A7');
            ctx.fill();
          }
        }

        elPcaContainer.style.display = '';
        const evx = pca.explained[Math.max(0, Math.min(pca.explained.length-1, pcx-1))] || 0;
        const evy = pca.explained[Math.max(0, Math.min(pca.explained.length-1, pcy-1))] || 0;
        elPcaInfo.textContent = `T=${T}, PCs available=${pca.K}. Var[X]=${(evx*100).toFixed(1)}%, Var[Y]=${(evy*100).toFixed(1)}%`;
        elPcaTitle.textContent = `PCA trajectory • PC${pcx} vs PC${pcy}`;
      } catch (e){
        elPcaContainer.style.display = 'none';
        elPcaInfo.textContent = `PCA error: ${e?.message || e}`;
      }
    }

    function renderPCANodes_(){
      try {
        const hs = algo && algo.gnn_hidden_states && algo.gnn_hidden_states.data;
        const hasHS = Array.isArray(hs) || (hs && typeof hs === 'object' && typeof hs.length === 'number');
        if (!hasHS){
          elShowPcaNodes.disabled = true;
          elPcaNodesContainer.style.display = 'none';
          elPcaNodesInfo.textContent = 'Hidden states not found in JSON.';
          return;
        } else {
          elShowPcaNodes.disabled = false;
        }
        if (!elShowPcaNodes.checked){
          elPcaNodesContainer.style.display = 'none';
          return;
        }

        const cssW = elPcaNodesCanvas.clientWidth || 600;
        if (cssW <= 1) {
          requestAnimationFrame(renderPCANodes);
          return;
        }

        const bIdx = Math.max(0, Number(elBidx.value) || 0);
        let Tvalid = validTFromIsLast(bIdx);
        if (Tvalid == null) {
          const Ls = Array.isArray(algo?.lengths) ? algo.lengths : null;
          if (Ls && Ls.length > bIdx) {
            const L = Number(Ls[bIdx]);
            if (Number.isFinite(L) && L > 0) Tvalid = L;
          }
        }
        if (Tvalid == null) {
          try { Tvalid = hs[Math.min(bIdx, hs.length-1)].length; } catch(_) { Tvalid = 1; }
        }

        const bt = hs[Math.min(bIdx, hs.length-1)] || [];
        const first = bt[0] || [];
        const N = Array.isArray(first) ? first.length : 0;
        const firstND = (N>0) ? first[0] : [];
        const D = Array.isArray(firstND) ? firstND.length : 1;

        const algoName = algo && algo.algorithm ? String(algo.algorithm) : 'algo';
        const cacheKey = `${algoName}|b=${bIdx}|T=${Tvalid}|N=${N}|D=${D}`;
        if (nodePcaCache.key !== cacheKey){
          let samples;
          try {
            samples = PCA.flattenBTNDToNodeSamples(hs, bIdx, Tvalid);
          } catch(e){
            elPcaNodesContainer.style.display = 'none';
            elPcaNodesInfo.textContent = 'Failed to build node samples for PCA.';
            return;
          }
          const K = Math.min(10, Math.min(D, samples.length));
          try {
            nodePcaCache.pca = PCA.fit(samples, K);
            nodePcaCache.key = cacheKey;
            // Also reset bounds cache when PCA basis changes (new batch / sizes)
            pcaNodesBoundsCache.key = null;
            pcaNodesBoundsCache.bounds = null;
          } catch(e){
            elPcaNodesContainer.style.display = 'none';
            elPcaNodesInfo.textContent = 'PCA fitting failed.';
            return;
          }
        }

        const pca = nodePcaCache.pca;
        if (!pca){
          elPcaNodesContainer.style.display = 'none';
          elPcaNodesInfo.textContent = 'PCA basis not available.';
          return;
        }

        const pcx = Math.max(1, Number(elPcxNodes.value) || 1);
        const pcy = Math.max(1, Number(elPcyNodes.value) || 2);

        // Bounds are computed once from ALL nodes across ALL times (basis is already from all times)
        const boundsKey = `b=${bIdx}|T=${Tvalid}|pcx=${pcx}|pcy=${pcy}|N=${N}|D=${D}`;
        if (pcaNodesBoundsCache.key !== boundsKey) {
          let samples;
          try {
            samples = PCA.flattenBTNDToNodeSamples(hs, bIdx, Tvalid);
          } catch(_e) {
            samples = null;
          }
          if (samples && samples.length) {
            const allPts = PCA.projectPointsXY(pca, samples, pcx, pcy);
            pcaNodesBoundsCache.bounds = computeBoundsFromPoints(allPts);
            pcaNodesBoundsCache.key = boundsKey;
          }
        }
        const bnd = pcaNodesBoundsCache.bounds;
        if (!bnd) {
          elPcaNodesContainer.style.display = 'none';
          elPcaNodesInfo.textContent = 'PCA bounds not available.';
          return;
        }
        let { xmin, xmax, ymin, ymax } = bnd;

        const curT = Math.max(0, Math.min((Tvalid||1)-1, Number(elT.value) || 0));
        const nodes = (bt && bt[curT]) ? bt[curT] : [];
        const vectors = new Array(N);
        for (let n=0;n<N;n++){
          const h = nodes[n] || [];
          const row = new Float64Array(D);
          if (Array.isArray(h)){
            for (let d=0; d<D; d++) row[d] = Number(h[d]) || 0;
          } else {
            row[0] = Number(h) || 0;
          }
          vectors[n] = row;
        }

        const pts = PCA.projectPointsXY(pca, vectors, pcx, pcy);

        const dpr = window.devicePixelRatio || 1;
        const cssH = elPcaNodesCanvas.clientHeight || 320;
        elPcaNodesCanvas.width = Math.max(1, Math.floor(cssW * dpr));
        elPcaNodesCanvas.height = Math.max(1, Math.floor(cssH * dpr));
        const ctx = elPcaNodesCanvas.getContext('2d');
        ctx.setTransform(dpr,0,0,dpr,0,0);
        ctx.clearRect(0,0,cssW,cssH);

        function sx(x){ return (x - xmin) / (xmax - xmin) * (cssW - 40) + 20; }
        function sy(y){ return (1 - (y - ymin) / (ymax - ymin)) * (cssH - 48) + 16; }

        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        ctx.strokeRect(10.5, 10.5, cssW-21, cssH-33);
        drawAxes(ctx, cssW, cssH, xmin, xmax, ymin, ymax);
        drawAxisLines(ctx, cssW, cssH, sx, sy);

        const colorByIndex = !!elColorGraphNodes?.checked;
        for (let i=0;i<pts.length;i++){
          const p = pts[i];
          const X = sx(p.x), Y = sy(p.y);
          ctx.beginPath();
          ctx.arc(X, Y, 3, 0, Math.PI*2);
          ctx.fillStyle = colorByIndex ? nodeIndexToColor(i, pts.length) : '#4E79A7';
          ctx.fill();
        }

        elPcaNodesContainer.style.display = '';
        const evx = pca.explained[Math.max(0, Math.min(pca.explained.length-1, pcx-1))] || 0;
        const evy = pca.explained[Math.max(0, Math.min(pca.explained.length-1, pcy-1))] || 0;
        elPcaNodesInfo.textContent = `N=${N}, T_basis=${Tvalid}, PCs available=${pca.K}. Var[X]=${(evx*100).toFixed(1)}%, Var[Y]=${(evy*100).toFixed(1)}%`;
        elPcaNodesTitle.textContent = `PCA (nodes) • PC${pcx} vs PC${pcy}`;
      } catch (_e){
        elPcaNodesContainer.style.display = 'none';
        elPcaNodesInfo.textContent = 'Error rendering node-wise PCA.';
      }
    }

    // DEBUG (temporary): help diagnose missing pi_h corr plot
    // Logs once per render when corr is enabled for pi_h.
    function _dbgCorrRender(name, probe, tIdx, bIdx, card){
      try{
        if (name !== 'pi_h') return;
        if (!probe || !probe.correctness_along_time) return;
        const s = probe.correctness_along_time;
        const is2D = Array.isArray(s) && s.length>0 && Array.isArray(s[0]);
        const len1 = Array.isArray(s) ? s.length : null;
        const len2 = is2D ? (Array.isArray(s[0]) ? s[0].length : null) : null;
        const wrapDisp = card && card._corrWrap ? card._corrWrap.style.display : '<no wrap>';
        const canDisp = card && card._corrCanvas ? card._corrCanvas.style.display : '<no canvas>';
        console.log('[viz][corr] pi_h render', {stage: probe.stage, tIdx, bIdx, is2D, len1, len2, wrapDisp, canDisp});
      } catch (e) {
        // ignore
      }
    }

    // events
    btnPlay.addEventListener('click', play);
    btnPause.addEventListener('click', stop);
    btnStepBack?.addEventListener('click', () => stepTime(-1));
    btnStepForward?.addEventListener('click', () => stepTime(+1));
    btnAll.addEventListener('click', () => setAllProbes(true));
    btnNone.addEventListener('click', () => setAllProbes(false));

    elT.addEventListener('input', () => { detectAndRender(); });
    elBidx.addEventListener('change', () => { stop(); scaleCache.clear(); pcaBoundsCache.key=null; pcaBoundsCache.bounds=null; pcaNodesBoundsCache.key=null; pcaNodesBoundsCache.bounds=null; detectAndRender(); });
    // PCA controls events
    if (HAS_PCA_UI) {
      elShowPca.addEventListener('change', () => { detectAndRender(); });
      elPcx.addEventListener('change', () => { pcaBoundsCache.key=null; pcaBoundsCache.bounds=null; detectAndRender(); });
      elPcy.addEventListener('change', () => { pcaBoundsCache.key=null; pcaBoundsCache.bounds=null; detectAndRender(); });

      elShowPcaNodes.addEventListener('change', () => { detectAndRender(); });
      elPcxNodes.addEventListener('change', () => { pcaNodesBoundsCache.key=null; pcaNodesBoundsCache.bounds=null; detectAndRender(); });
      elPcyNodes.addEventListener('change', () => { pcaNodesBoundsCache.key=null; pcaNodesBoundsCache.bounds=null; detectAndRender(); });

      if (elColorGraphNodes) {
        elColorGraphNodes.addEventListener('change', () => { detectAndRender(); });
      }
    }

    elVCell.addEventListener('change', detectAndRender);
    elMCell.addEventListener('change', detectAndRender);
    elR0.addEventListener('input', detectAndRender);
    elR1.addEventListener('input', detectAndRender);
    elC0.addEventListener('input', detectAndRender);
    elC1.addEventListener('input', detectAndRender);

    elContrast.addEventListener('input', detectAndRender);
    elFps.addEventListener('change', () => { if (timer) play(); });

    elLockScale.addEventListener('change', () => { scaleCache.clear(); detectAndRender(); });

    window.addEventListener('resize', () => { detectAndRender(); });

    elFile.addEventListener('change', async () => {
      stop();
      const f = elFile.files?.[0];
      if (!f) return;
      const text = await f.text();
      let parsed = null;
      try { parsed = JSON.parse(text); } catch (_e) { alert('Invalid JSON'); return; }

      if (!(parsed && parsed.probes && typeof parsed.probes === 'object')) {
        alert('Expected multi-probe JSON: { "algorithm": "...", "probes": { ... }, "is_last": [[...]] }');
        return;
      }

      algo = parsed;

      selectedState.clear();
      scaleCache.clear();

      elT.value = "0";
      elBidx.value = "0";
      setHoverText("Hover any cell to see its value here.");

      detectAndRender();
    });
  }

  window.CLRSViewer = window.CLRSViewer || {};
  window.CLRSViewer.init = init;

  // Auto-init
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => init());
  } else {
    init();
  }
})();
