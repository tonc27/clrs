// Simple PCA helper for CLRS viewer

(function(global){
  'use strict';

  function flattenBTNDToTimeVectors(hsBTND, bIdx, Tvalid){
    if (!Array.isArray(hsBTND) || hsBTND.length === 0)
      throw new Error('Invalid hidden states: expected [B][T][N][D]');
    const B = hsBTND.length;
    const b = Math.max(0, Math.min(B-1, bIdx|0));
    const bt = hsBTND[b];
    if (!Array.isArray(bt) || bt.length === 0)
      throw new Error('Invalid hidden states: missing time dimension');
    const T = bt.length;
    const Tcap = (Number.isFinite(Tvalid) && Tvalid>0) ? Math.min(Tvalid|0, T) : T;
    const first = bt[0];
    if (!Array.isArray(first) || first.length === 0)
      throw new Error('Invalid hidden states: missing node dimension');
    const N = first.length;
    const firstND = first[0];
    const D = Array.isArray(firstND) ? firstND.length : 1;
    const F = N * D;
    const out = new Array(Tcap);
    for (let t=0;t<Tcap;t++){
      const vec = new Float64Array(F);
      const nodes = bt[t];
      let idx = 0;
      for (let n=0;n<N;n++){
        const h = nodes[n];
        if (Array.isArray(h)){
          for (let d=0; d<D; d++) { vec[idx++] = Number(h[d]) || 0; }
        } else {
          vec[idx++] = Number(h) || 0;
        }
      }
      out[t] = vec;
    }
    return out;
  }

  function fit(timeVectors, k){
    const T = timeVectors.length;
    if (T === 0) throw new Error('No time vectors');
    const F = timeVectors[0].length;
    const K = Math.max(1, Math.min(k|0 || 2, Math.min(T, F)));

    const mean = new Float64Array(F);
    for (let t=0;t<T;t++){
      const v = timeVectors[t];
      for (let f=0; f<F; f++) mean[f] += v[f];
    }
    for (let f=0; f<F; f++) mean[f] /= T;

    // Dual algorithm selection:
    // - If T <= F: use Gram matrix G = Xc * Xc^T (T x T)  [O(T^2 F)]
    // - If F < T : use covariance C = Xc^T * Xc (F x F)   [O(T F^2)]
    if (T <= F) {
      const G = new Float64Array(T*T);
      for (let i=0;i<T;i++){
        const xi = timeVectors[i];
        for (let j=i;j<T;j++){
          const xj = timeVectors[j];
          let dot = 0;
          for (let f=0; f<F; f++){
            const a = xi[f] - mean[f];
            const b = xj[f] - mean[f];
            dot += a*b;
          }
          G[i*T + j] = dot;
          if (j !== i) G[j*T + i] = dot;
        }
      }
      const eig = symmetricEigen(G, T);
      const svals = eig.values.map(v => Math.sqrt(Math.max(0, v)));
      const idx = svals.map((s, i) => [s, i]).sort((a,b)=>b[0]-a[0]).map(x=>x[1]);
      const S = idx.map(i => svals[i]);
      const U = makeMatrix(T, T);
      for (let c=0;c<T;c++){
        const srcCol = idx[c];
        for (let r=0;r<T;r++) U[r*T + c] = eig.vectors[r*T + srcCol];
      }
      const Kuse = Math.min(K, S.length);
      const V = makeMatrix(F, Kuse);
      for (let kcol=0; kcol<Kuse; kcol++){
        const s = S[kcol];
        const inv = (s > 1e-12) ? (1.0/s) : 0.0;
        for (let f=0; f<F; f++){
          let acc = 0;
          for (let t=0; t<T; t++){
            acc += (timeVectors[t][f] - mean[f]) * U[t*T + kcol];
          }
          V[f*Kuse + kcol] = acc * inv;
        }
      }
      const scores = makeMatrix(T, Kuse);
      for (let t=0;t<T;t++){
        for (let kcol=0;kcol<Kuse;kcol++){
          scores[t*Kuse + kcol] = U[t*T + kcol] * S[kcol];
        }
      }
      const S2 = S.map(x=>x*x);
      const total = S2.reduce((a,b)=>a+b, 0) || 1;
      const explained = S2.slice(0, Kuse).map(v => v / total);
      return { mean, components: V, scores, singularValues: S.slice(0, Kuse), explained, T, F, K: Kuse };
    } else {
      const C = new Float64Array(F*F);
      for (let i=0;i<F;i++){
        for (let j=i;j<F;j++){
          let acc = 0;
          for (let t=0;t<T;t++){
            const a = timeVectors[t][i] - mean[i];
            const b = timeVectors[t][j] - mean[j];
            acc += a*b;
          }
          C[i*F + j] = acc;
          if (j !== i) C[j*F + i] = acc;
        }
      }
      const eig = symmetricEigen(C, F);
      const evals = eig.values.map(v => Math.max(0, v));
      const idx = evals.map((v, i) => [v, i]).sort((a,b)=>b[0]-a[0]).map(x=>x[1]);
      const Kuse = Math.min(K, idx.length);
      const V = makeMatrix(F, Kuse);
      const S = new Array(Kuse);
      for (let kcol=0;kcol<Kuse;kcol++){
        const srcCol = idx[kcol];
        S[kcol] = Math.sqrt(evals[srcCol]);
        for (let f=0; f<F; f++) V[f*Kuse + kcol] = eig.vectors[f*F + srcCol];
      }
      const scores = makeMatrix(T, Kuse);
      for (let t=0;t<T;t++){
        for (let kcol=0;kcol<Kuse;kcol++){
          let acc = 0;
          for (let f=0; f<F; f++){
            acc += (timeVectors[t][f] - mean[f]) * V[f*Kuse + kcol];
          }
          scores[t*Kuse + kcol] = acc;
        }
      }
      const totalVar = evals.reduce((a,b)=>a+b, 0) || 1;
      const explained = new Array(Kuse);
      for (let kcol=0;kcol<Kuse;kcol++) explained[kcol] = (S[kcol]*S[kcol]) / totalVar;
      return { mean, components: V, scores, singularValues: S, explained, T, F, K: Kuse };
    }
  }

  function projectXY(pca, pcx, pcy){
    const K = pca.K;
    let ix = Math.max(1, Math.min(K, pcx|0 || 1)) - 1;
    let iy = Math.max(1, Math.min(K, pcy|0 || 2)) - 1;
    // NOTE: don't force different PCs; allow pcx === pcy (user may want y=x).
    const T = pca.T;
    const coords = new Array(T);
    for (let t=0;t<T;t++){
      const x = pca.scores[t*K + ix];
      const y = pca.scores[t*K + iy];
      coords[t] = {t, x, y};
    }
    return coords;
  }

  function makeMatrix(R, C){ return new Float64Array(R*C); }

  // Symmetric eigen-decomposition using Jacobi rotations
  function symmetricEigen(A, N){
    const V = makeIdentity(N);
    const a = new Float64Array(A);
    const maxIter = 50 * N;
    for (let iter=0; iter<maxIter; iter++){
      let p=0, q=1; let max = Math.abs(a[0*N + 1]);
      for (let i=0;i<N;i++){
        for (let j=i+1;j<N;j++){
          const val = Math.abs(a[i*N + j]);
          if (val > max){ max = val; p=i; q=j; }
        }
      }
      if (max < 1e-10) break;
      const app = a[p*N + p];
      const aqq = a[q*N + q];
      const apq = a[p*N + q];
      const phi = 0.5 * Math.atan2(2*apq, (aqq - app));
      const c = Math.cos(phi);
      const s = Math.sin(phi);
      for (let k=0;k<N;k++){
        const aik = a[p*N + k];
        const aqk = a[q*N + k];
        a[p*N + k] = c*aik - s*aqk;
        a[q*N + k] = s*aik + c*aqk;
      }
      for (let k=0;k<N;k++){
        const kip = a[k*N + p];
        const kiq = a[k*N + q];
        a[k*N + p] = c*kip - s*kiq;
        a[k*N + q] = s*kip + c*kiq;
      }
      a[p*N + q] = a[q*N + p] = 0;
      for (let k=0;k<N;k++){
        const vkp = V[k*N + p];
        const vkq = V[k*N + q];
        V[k*N + p] = c*vkp - s*vkq;
        V[k*N + q] = s*vkp + c*vkq;
      }
    }
    const values = new Array(N);
    for (let i=0;i<N;i++) values[i] = a[i*N + i];
    return { values, vectors: V };
  }

  function makeIdentity(N){
    const M = makeMatrix(N, N);
    for (let i=0;i<N;i++) M[i*N + i] = 1;
    return M;
  }

  function flattenBTNDToNodeSamples(hsBTND, bIdx, Tvalid){
    if (!Array.isArray(hsBTND) || hsBTND.length === 0)
      throw new Error('Invalid hidden states: expected [B][T][N][D]');
    const B = hsBTND.length;
    const b = Math.max(0, Math.min(B-1, bIdx|0));
    const bt = hsBTND[b];
    if (!Array.isArray(bt) || bt.length === 0)
      throw new Error('Invalid hidden states: missing time dimension');
    const T = bt.length;
    const Tcap = (Number.isFinite(Tvalid) && Tvalid>0) ? Math.min(Tvalid|0, T) : T;
    const first = bt[0];
    if (!Array.isArray(first) || first.length === 0)
      throw new Error('Invalid hidden states: missing node dimension');
    const N = first.length;
    const firstND = first[0];
    const D = Array.isArray(firstND) ? firstND.length : 1;
    const S = Tcap * N;
    const out = new Array(S);
    let s = 0;
    for (let t=0;t<Tcap;t++){
      const nodes = bt[t];
      for (let n=0;n<N;n++){
        const row = new Float64Array(D);
        const h = nodes[n];
        if (Array.isArray(h)){
          for (let d=0; d<D; d++) row[d] = Number(h[d]) || 0;
        } else {
          row[0] = Number(h) || 0;
        }
        out[s++] = row;
      }
    }
    return out;
  }

  function projectPointsXY(pca, vectors, pcx, pcy){
    const K = pca.K;
    const F = pca.F;
    const comps = pca.components;
    let ix = Math.max(1, Math.min(K, pcx|0 || 1)) - 1;
    let iy = Math.max(1, Math.min(K, pcy|0 || 2)) - 1;
    // NOTE: don't force different PCs; allow pcx === pcy (user may want y=x).
    const coords = new Array(vectors.length);
    for (let i=0;i<vectors.length;i++){
      const v = vectors[i];
      let x=0, y=0;
      for (let f=0; f<F; f++){
        const vf = (Number(v[f]) || 0) - pca.mean[f];
        x += vf * comps[f*K + ix];
        y += vf * comps[f*K + iy];
      }
      coords[i] = {i, x, y};
    }
    return coords;
  }

  const PCA = {
    flattenBTNDToTimeVectors,
    fit,
    projectXY,
    flattenBTNDToNodeSamples,
    projectPointsXY,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = PCA;
  global.PCA = PCA;

})(typeof window !== 'undefined' ? window : globalThis);
