import React, { useEffect, useMemo, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Legend, ScatterChart, Scatter,
  ReferenceLine,
} from 'recharts';

const API = '/api';

function lambdaTag(lam) {
  return `lambda_${String(lam).replace('.', 'p')}`;
}

function useFetch(url) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    let cancelled = false;
    fetch(url)
      .then(r => r.ok ? r.json() : Promise.reject(new Error(r.statusText)))
      .then(j => { if (!cancelled) setData(j); })
      .catch(e => { if (!cancelled) setErr(e); });
    return () => { cancelled = true; };
  }, [url]);
  return { data, err };
}

// ---------------------------------------------------------------------- //
// Hero section
// ---------------------------------------------------------------------- //
function Hero() {
  return (
    <header className="hero">
      <div className="kicker">Research · Neural Architecture</div>
      <h1>
        A network that <em>prunes itself</em>, <br />one connection at a time.
      </h1>
      <p>
        Every weight in this classifier carries a learnable gate. During training,
        an L1 penalty on those gates competes with cross-entropy loss — and the
        network decides, on its own, which of its ~1.74M connections are worth
        keeping. The dashboard below tracks that negotiation across three values
        of λ.
      </p>
    </header>
  );
}

// ---------------------------------------------------------------------- //
// Summary cards
// ---------------------------------------------------------------------- //
function SummaryCards({ rows, bestTag }) {
  return (
    <section className="section">
      <div className="section-header">
        <h2>The Sweep</h2>
        <span className="note">3 runs · 12 epochs each · fixed seed</span>
      </div>
      <div className="grid-3">
        {rows.map((r) => {
          const isBest = lambdaTag(r.lambda) === bestTag;
          const activePct = 100 - r.global_sparsity_pct;
          return (
            <div key={r.lambda} className={`card ${isBest ? 'is-best' : ''}`}>
              <div className="metric-lambda">λ = {r.lambda}</div>
              <div className="metric-primary">
                {r.test_accuracy.toFixed(1)}<span style={{fontSize: '28px', color: 'var(--fg-dim)'}}>%</span>
              </div>
              <div className="metric-label">Test Accuracy</div>

              <div className="metric-secondary">
                <span className="k">Sparsity</span>
                <span className="v" style={{color: 'var(--pruned)'}}>
                  {r.global_sparsity_pct.toFixed(1)}%
                </span>
              </div>
              <div className="bar-container"><div className="bar pruned" style={{width: `${r.global_sparsity_pct}%`}} /></div>

              <div className="metric-secondary" style={{borderTop: 'none', paddingTop: '10px', marginTop: '6px'}}>
                <span className="k">Active gates</span>
                <span className="v">{r.active_gates.toLocaleString()}</span>
              </div>
              <div className="bar-container"><div className="bar" style={{width: `${activePct}%`}} /></div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------- //
// Gate distribution histogram
// ---------------------------------------------------------------------- //
function GateHistogram({ sweep }) {
  const [selectedLambda, setSelectedLambda] = useState(sweep[0].lambda);
  const entry = sweep.find(e => e.lambda === selectedLambda);
  const { bin_edges, counts, total_gates } = entry.gate_histogram;

  const data = counts.map((c, i) => ({
    bin: (bin_edges[i] + bin_edges[i + 1]) / 2,
    binLabel: bin_edges[i].toFixed(2),
    count: c,
    pruned: bin_edges[i + 1] <= 0.01,
  }));

  return (
    <div className="chart-card">
      <h3>Gate value distribution</h3>
      <div className="subtitle">
        sigmoid(score) at end of training · log y-scale
      </div>
      <div className="selector">
        {sweep.map(e => (
          <button
            key={e.lambda}
            className={e.lambda === selectedLambda ? 'active' : ''}
            onClick={() => setSelectedLambda(e.lambda)}
          >
            λ = {e.lambda}
          </button>
        ))}
      </div>
      <div style={{height: 320}}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{top: 10, right: 10, left: -10, bottom: 0}}>
            <CartesianGrid stroke="#26262b" strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="bin"
              type="number"
              domain={[0, 1]}
              tick={{fill: '#8b8b92', fontSize: 11, fontFamily: 'JetBrains Mono'}}
              stroke="#3a3a42"
              tickFormatter={v => v.toFixed(2)}
            />
            <YAxis
              scale="log"
              domain={[1, 'auto']}
              allowDataOverflow
              tick={{fill: '#8b8b92', fontSize: 11, fontFamily: 'JetBrains Mono'}}
              stroke="#3a3a42"
              tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v}
            />
            <Tooltip
              contentStyle={{background: '#16161a', border: '1px solid #3a3a42', fontFamily: 'JetBrains Mono', fontSize: 12}}
              labelFormatter={v => `gate ≈ ${v.toFixed(3)}`}
              formatter={(val) => [val.toLocaleString(), 'gates']}
            />
            <ReferenceLine x={0.01} stroke="#ff4a6e" strokeDasharray="4 2" label={{value: 'prune threshold', position: 'top', fill: '#ff4a6e', fontSize: 10, fontFamily: 'JetBrains Mono'}} />
            <Bar dataKey="count" fill="#4ac3b8">
              {data.map((d, i) => (
                <rect key={i} fill={d.pruned ? '#ff4a6e' : '#4ac3b8'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div style={{fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--fg-faint)', marginTop: 8, textAlign: 'right'}}>
        {total_gates.toLocaleString()} total gates
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------- //
// Training trajectory
// ---------------------------------------------------------------------- //
function TrainingTrajectory({ sweep }) {
  const epochs = sweep[0].history.length;
  const data = Array.from({length: epochs}, (_, i) => {
    const row = {epoch: i + 1};
    for (const entry of sweep) {
      row[`acc_${entry.lambda}`] = entry.history[i].test_accuracy;
      row[`sp_${entry.lambda}`] = entry.history[i].global_sparsity_pct;
    }
    return row;
  });

  const COLORS = ['#4ac3b8', '#ff7a45', '#ff4a6e'];

  return (
    <div className="chart-card">
      <h3>Training trajectories</h3>
      <div className="subtitle">test accuracy (solid) vs sparsity (dashed)</div>
      <div style={{height: 320}}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{top: 10, right: 16, left: -10, bottom: 0}}>
            <CartesianGrid stroke="#26262b" strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="epoch" tick={{fill: '#8b8b92', fontSize: 11, fontFamily: 'JetBrains Mono'}} stroke="#3a3a42" />
            <YAxis tick={{fill: '#8b8b92', fontSize: 11, fontFamily: 'JetBrains Mono'}} stroke="#3a3a42" domain={[0, 100]} />
            <Tooltip
              contentStyle={{background: '#16161a', border: '1px solid #3a3a42', fontFamily: 'JetBrains Mono', fontSize: 12}}
              formatter={(val, name) => [val.toFixed(2) + '%', name]}
            />
            <Legend wrapperStyle={{fontFamily: 'JetBrains Mono', fontSize: 11}} />
            {sweep.map((entry, idx) => [
              <Line key={`a${entry.lambda}`} type="monotone" dataKey={`acc_${entry.lambda}`}
                stroke={COLORS[idx]} strokeWidth={2} dot={false}
                name={`acc λ=${entry.lambda}`} />,
              <Line key={`s${entry.lambda}`} type="monotone" dataKey={`sp_${entry.lambda}`}
                stroke={COLORS[idx]} strokeWidth={1.5} strokeDasharray="4 3" dot={false}
                name={`sparsity λ=${entry.lambda}`} />,
            ])}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------- //
// Per-layer breakdown table
// ---------------------------------------------------------------------- //
function PerLayerTable({ sweep, bestTag }) {
  const rows = [];
  const layerCount = sweep[0].per_layer_sparsity.length;
  for (const entry of sweep) {
    rows.push({
      lambda: entry.lambda,
      acc: entry.test_accuracy,
      global: entry.global_sparsity_pct,
      layers: entry.per_layer_sparsity.map(l => l.sparsity_pct),
      shapes: entry.per_layer_sparsity.map(l => l.shape),
      isBest: lambdaTag(entry.lambda) === bestTag,
    });
  }
  return (
    <div className="chart-card">
      <h3>Per-layer sparsity breakdown</h3>
      <div className="subtitle">
        how many gates each PrunableLinear layer lost
      </div>
      <table className="data-table" style={{marginTop: 12}}>
        <thead>
          <tr>
            <th>λ</th>
            <th>Accuracy</th>
            <th>Global</th>
            {Array.from({length: layerCount}, (_, i) => (
              <th key={i}>Layer {i}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.lambda} className={r.isBest ? 'highlighted' : ''}>
              <td>{r.lambda}</td>
              <td>{r.acc.toFixed(2)}%</td>
              <td>{r.global.toFixed(2)}%</td>
              {r.layers.map((s, i) => (
                <td key={i}>
                  {s.toFixed(1)}%
                  <span style={{color: 'var(--fg-faint)', fontSize: 10, marginLeft: 6}}>
                    [{r.shapes[i].join('×')}]
                  </span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------- //
// Live predict
// ---------------------------------------------------------------------- //
const CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'];

function PredictPanel({ sweep, bestTag }) {
  const [image, setImage] = useState(null);
  const [pred, setPred] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tag, setTag] = useState(bestTag);

  function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setImage(reader.result);
      setPred(null); setErr(null);
    };
    reader.readAsDataURL(file);
  }

  async function runPredict() {
    if (!image) return;
    setLoading(true); setErr(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          image_base64: image,
          lambda_tag: tag.replace('lambda_', ''),
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      setPred(await res.json());
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  const sorted = pred
    ? pred.probs.map((p, i) => ({i, p, name: CLASSES[i]}))
        .sort((a, b) => b.p - a.p)
    : null;

  return (
    <section className="section">
      <div className="section-header">
        <h2>Try a live prediction</h2>
        <span className="note">any 32×32 image · models run in-browser via FastAPI</span>
      </div>
      <div className="predict-row">
        <div>
          <label className="drop-zone">
            {image
              ? <img src={image} alt="uploaded" />
              : <span className="hint">CLICK OR DROP<br/>A 32×32 IMAGE</span>
            }
            <input type="file" accept="image/*" onChange={handleFile} />
          </label>

          <div className="selector" style={{marginTop: 16}}>
            {sweep.map(e => (
              <button key={e.lambda}
                className={tag === lambdaTag(e.lambda) ? 'active' : ''}
                onClick={() => setTag(lambdaTag(e.lambda))}>
                λ={e.lambda}
              </button>
            ))}
          </div>

          <button
            disabled={!image || loading}
            onClick={runPredict}
            style={{
              width: '100%', marginTop: 12, padding: '12px',
              background: image ? 'var(--accent)' : 'var(--bg-elev)',
              color: image ? 'var(--bg)' : 'var(--fg-faint)',
              border: 'none',
              fontFamily: 'JetBrains Mono', fontSize: 12,
              letterSpacing: '0.1em',
              cursor: image ? 'pointer' : 'not-allowed',
              fontWeight: 600,
            }}>
            {loading ? 'INFERRING…' : 'RUN PREDICTION'}
          </button>
        </div>

        <div className="predictions">
          {err && <div className="error">{err}</div>}
          {!pred && !err && (
            <div style={{color: 'var(--fg-faint)', fontFamily: 'var(--font-mono)', fontSize: 12}}>
              Upload an image and hit RUN. Class probabilities will be ranked here.
            </div>
          )}
          {pred && sorted && (
            <>
              <div style={{marginBottom: 16, fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--fg-dim)'}}>
                <span style={{color: 'var(--accent)'}}>λ = {pred.lambda_used}</span>
                {' · '}
                sparsity <b style={{color: 'var(--pruned)'}}>{pred.global_sparsity_pct.toFixed(1)}%</b>
                {' · '}
                prediction <b style={{color: 'var(--fg)'}}>{pred.predicted_class}</b>
              </div>
              {sorted.slice(0, 10).map((row, i) => (
                <div key={row.i} className={`pred-row ${i === 0 ? 'top' : ''}`}>
                  <div className="label">{row.name}</div>
                  <div className="bar-wrap"><div className="bar-fill" style={{width: `${row.p * 100}%`}} /></div>
                  <div className="pct">{(row.p * 100).toFixed(1)}%</div>
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------- //
// App
// ---------------------------------------------------------------------- //
export default function App() {
  const { data: sweep, err: sweepErr } = useFetch(`${API}/sweep`);
  const { data: summary } = useFetch(`${API}/summary`);

  if (sweepErr) {
    return (
      <div className="app">
        <div className="error">
          Could not reach the API at {API}. Make sure the FastAPI server is running:
          <br/><br/>
          <code>uvicorn api.server:app --port 8000</code>
        </div>
      </div>
    );
  }
  if (!sweep || !summary) {
    return (
      <div className="app">
        <div className="loading">
          Loading sweep data <span className="dot"/><span className="dot"/><span className="dot"/>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <Hero />
      <SummaryCards rows={summary.rows} bestTag={summary.best_tag} />

      <section className="section">
        <div className="section-header">
          <h2>Dynamics</h2>
          <span className="note">how λ shapes the training trajectory</span>
        </div>
        <div className="grid-2">
          <GateHistogram sweep={sweep} />
          <TrainingTrajectory sweep={sweep} />
        </div>
      </section>

      <section className="section">
        <PerLayerTable sweep={sweep} bestTag={summary.best_tag} />
      </section>

      <PredictPanel sweep={sweep} bestTag={summary.best_tag} />

      <footer className="footer">
        <span>self-pruning-nn · PrunableLinear × {sweep[0].per_layer_sparsity.length} layers</span>
        <span>{sweep[0].gate_histogram.total_gates.toLocaleString()} gates · FastAPI · React · Recharts</span>
      </footer>
    </div>
  );
}
