import { Component, useEffect, useState } from 'react'

export class AppErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { failed: false } }
  static getDerivedStateFromError() { return { failed: true } }
  render() {
    return this.state.failed
      ? <main className="app-shell"><p className="error-message">Something went wrong, please reload</p></main>
      : this.props.children
  }
}

const API_PREFIX = '/api'

const vitalDefinitions = [
  { key: 'HR', label: 'Heart Rate', unit: 'bpm', min: 40, max: 180, step: 1, defaultValue: 80 },
  { key: 'Temp', label: 'Temperature', unit: '°C', min: 35, max: 41, step: 0.1, defaultValue: 37 },
  { key: 'Resp', label: 'Respiratory Rate', unit: '/min', min: 6, max: 50, step: 1, defaultValue: 18 },
  { key: 'MAP', label: 'Mean Arterial Pressure', unit: 'mmHg', min: 40, max: 140, step: 1, defaultValue: 75 },
  { key: 'O2Sat', label: 'O₂ Saturation', unit: '%', min: 70, max: 100, step: 1, defaultValue: 97 },
  { key: 'Lactate', label: 'Lactate', unit: 'mmol/L', min: 0, max: 15, step: 0.1, defaultValue: 1.5 },
]

const initialVitals = Object.fromEntries(
  vitalDefinitions.map(({ key, defaultValue }) => [key, defaultValue]),
)

function normalizeSampleVitals(features) {
  return Object.fromEntries(vitalDefinitions.map((vital) => {
    const precision = Number.isInteger(vital.step) ? 0 : String(vital.step).split('.')[1].length
    const rounded = Number(Number(features[vital.key]).toFixed(precision))
    const bounded = Math.min(vital.max, Math.max(vital.min, rounded))
    return [vital.key, bounded]
  }))
}

function riskStyle(riskBand) {
  if (riskBand === 'red') return { label: 'Elevated risk', tone: 'high' }
  if (riskBand === 'yellow') return { label: 'Intermediate risk', tone: 'medium' }
  return { label: 'Low risk', tone: 'low' }
}

function friendlyFeature(feature) {
  return feature.replace('ICULOS', 'ICU length of stay')
}

function changedVital(previous, current) {
  return vitalDefinitions.find((vital) => previous[vital.key] !== current[vital.key])
}

function formatDeltaPercent(delta) {
  const percent = delta * 100
  const precision = Math.abs(percent) < 0.01 ? 4 : 2
  return `${percent >= 0 ? '+' : ''}${percent.toFixed(precision)}%`
}

function subtypeLabel(cluster) {
  const isHigher = (name) => cluster.top_features.some(
    (item) => item.feature.startsWith(name) && item.description.startsWith('Higher'),
  )
  const isLower = (name) => cluster.top_features.some(
    (item) => item.feature.startsWith(name) && item.description.startsWith('Lower'),
  )
  return (isHigher('Lactate') || isHigher('Creatinine') || isLower('BaseExcess') || isLower('pH'))
    ? 'Metabolic/Renal-Stress Profile'
    : 'Metabolically Stable Profile'
}

function subtypeTag(feature) {
  const name = feature.feature.split(' (')[0]
  const isHigher = feature.description.startsWith('Higher')
  const isLower = feature.description.startsWith('Lower') || feature.description.startsWith('Reduced')
  return `${isHigher ? '↑' : isLower ? '↓' : '•'} ${name}`
}

function App() {
  const [vitals, setVitals] = useState(initialVitals)
  const [selectedModel, setSelectedModel] = useState('xgb')
  const [baseline, setBaseline] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [sampleDetails, setSampleDetails] = useState(null)
  const [isLoadingSample, setIsLoadingSample] = useState(false)
  const [isPredicting, setIsPredicting] = useState(false)
  const [error, setError] = useState('')
  const [subtypeSummary, setSubtypeSummary] = useState(null)
  const [labPanelExpanded, setLabPanelExpanded] = useState(false)
  const [windowId, setWindowId] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [modelDetailsOpen, setModelDetailsOpen] = useState(false)
  const labGroups = Array.isArray(sampleDetails?.full_lab_panel)
    ? sampleDetails.full_lab_panel
    : (sampleDetails?.full_lab_panel?.groups ?? [])

  useEffect(() => {
    let active = true
    fetch(`${API_PREFIX}/subtypes`)
      .then((response) => response.ok ? response.json() : null)
      .then((summary) => { if (active) setSubtypeSummary(summary) })
      .catch(() => { /* Detection remains available if the optional subtype artifact is absent. */ })
    return () => { active = false }
  }, [])

  useEffect(() => {
    let active = true
    fetch(`${API_PREFIX}/model-info?model=${selectedModel}`)
      .then((response) => response.ok ? response.json() : null)
      .then((info) => { if (active) setModelInfo(info) })
      .catch(() => { if (active) setModelInfo(null) })
    return () => { active = false }
  }, [selectedModel])

  const updateVital = (key, rawValue) => {
    setVitals((current) => ({ ...current, [key]: Number(rawValue) }))
    setWindowId(null) // Clear window_id when editing manually
  }

  async function loadPatient(risk = null) {
    setError('')
    setIsLoadingSample(true)
    try {
      // Keep the demo pool aligned with the model currently shown in the UI.
      // Random sampling remains random; the model only matters for high/low pools.
      const query = new URLSearchParams({ model: selectedModel })
      if (risk) query.set('risk', risk)
      const response = await fetch(`${API_PREFIX}/sample-patient?${query.toString()}`)
      if (!response.ok) throw new Error(`Could not load a sample patient (${response.status}).`)
      const sample = await response.json()
      // Browser number controls enforce their step values; raw float32 values
      // from the API need rounding before they can be submitted by the form.
      setVitals(normalizeSampleVitals(sample.features))
      setSampleDetails(sample)
      setWindowId(sample.window_id || null)
      setBaseline(null)
      setComparison(null)
      setLabPanelExpanded(false) // Collapse when loading new patient
    } catch (requestError) {
      setError(requestError.message || 'Could not reach the API. Is it running on port 8000?')
    } finally {
      setIsLoadingSample(false)
    }
  }

  async function requestPredictions(inputVitals) {
    const models = ['xgb', 'rf', 'gb', 'adaboost']
    const payload = windowId ? { ...inputVitals, window_id: windowId } : inputVitals
    const responses = await Promise.all(models.map(async (model) => {
      const response = await fetch(`${API_PREFIX}/predict?model=${model}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) throw new Error(`Prediction failed (${response.status}).`)
      return response.json()
    }))
    return Object.fromEntries(models.map((model, index) => [model, responses[index]]))
  }

  async function predict(event) {
    event.preventDefault()
    setError('')
    setIsPredicting(true)
    try {
      const results = await requestPredictions(vitals)
      setBaseline({ vitals: { ...vitals }, results })
      setComparison(null)
    } catch (requestError) {
      setError(requestError.message || 'Could not reach the prediction API.')
    } finally {
      setIsPredicting(false)
    }
  }

  async function compare() {
    const changed = changedVital(baseline.vitals, vitals)
    if (!changed) {
      setError('Adjust exactly one vital before comparing.')
      return
    }
    const changeCount = vitalDefinitions.filter((vital) => baseline.vitals[vital.key] !== vitals[vital.key]).length
    if (changeCount !== 1) {
      setError('What-if comparison supports one changed vital at a time. Reset the baseline or restore the other inputs.')
      return
    }
    setError('')
    setIsPredicting(true)
    try {
      const results = await requestPredictions(vitals)
      setComparison({ vitals: { ...vitals }, results, changed })
    } catch (requestError) {
      setError(requestError.message || 'Could not reach the prediction API.')
    } finally {
      setIsPredicting(false)
    }
  }

  function resetBaseline() {
    setBaseline(null)
    setComparison(null)
    setError('')
  }

  const displayedResults = comparison?.results || baseline?.results
  const result = displayedResults?.[selectedModel] || null

  // Use ONLY the selected model's risk score and risk band
  // Do NOT use ensemble or max across models
  const risk = result ? riskStyle(result.risk_band) : null

  const maximumShap = result
    ? Math.max(...result.top_features.map((item) => Math.abs(item.shap_value)), Number.EPSILON)
    : 1
  const activeBaseline = baseline?.results?.[selectedModel]
  const activeComparison = comparison?.results?.[selectedModel]
  const delta = activeComparison && activeBaseline ? activeComparison.risk_score - activeBaseline.risk_score : null

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="brand-mark" aria-hidden="true">✚</div>
        <div>
          <p className="eyebrow">CLINICAL DECISION SUPPORT · RESEARCH PROTOTYPE</p>
          <h1>Sepsis Early Detection <span>— Prototype</span></h1>

        </div>
        <div className="header-status"><span className="status-dot" />Model ready</div>
      </header>

      <main className="dashboard">
        <section className="intro-panel">
          <div>
            <p className="section-kicker">Patient snapshot</p>
            <h2>Estimate near-term sepsis risk</h2>
            <p>Enter six current measurements or load a held-out example patient. The model returns an estimated risk and a concise explanation.</p>
          </div>
          <div className="sample-actions">
            <button className="secondary-button" onClick={() => loadPatient('high')} disabled={isLoadingSample} type="button">
              {isLoadingSample ? <><span className="spinner small" /> Loading…</> : 'Load High-Risk Example'}
            </button>
            <button className="secondary-button" onClick={() => loadPatient('low')} disabled={isLoadingSample} type="button">Load Low-Risk Example</button>
            <button className="text-button" onClick={() => loadPatient()} disabled={isLoadingSample} type="button">Random</button>
          </div>
        </section>

        <section className="model-picker" aria-label="Model comparison control">
          <div>
            <p className="section-kicker">Model view</p>
            <strong>{modelInfo?.display_name || (selectedModel === 'xgb' ? 'XGBoost' : selectedModel === 'rf' ? 'Random Forest' : selectedModel === 'gb' ? 'Gradient Boosting' : 'AdaBoost')}</strong>
          </div>
          <label>
            Show primary explanation for
            <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
              <option value="xgb">XGBoost</option>
              <option value="rf">Random Forest</option>
              <option value="gb">Gradient Boosting</option>
              <option value="adaboost">AdaBoost</option>
            </select>
          </label>
          <p className="leakage-badge">ICU length of stay excluded (time-proxy leakage)</p>
          <p className="band-note">Risk colors use a provisional display band, not a clinical threshold.</p>
          {modelInfo && <details className="model-details" open={modelDetailsOpen} onToggle={(event) => setModelDetailsOpen(event.currentTarget.open)}>
            <summary>Model details</summary>
            <p>Test AUROC: {modelInfo?.test_metrics?.test_auroc == null ? 'Unavailable' : `${(modelInfo.test_metrics.test_auroc * 100).toFixed(2)}%`} · PR-AUC: {modelInfo?.test_metrics?.test_pr_auc == null ? 'Unavailable' : `${(modelInfo.test_metrics.test_pr_auc * 100).toFixed(2)}%`} · Base rate: {modelInfo?.base_rate == null ? 'Unavailable' : `${(modelInfo.base_rate * 100).toFixed(2)}%`}</p>
            <p>{modelInfo?.feature_count ?? 'Unavailable'} features · train/test patients: {modelInfo?.train_patient_count ?? 'Unavailable'}/{modelInfo?.test_patient_count ?? 'Unavailable'}</p>
            <pre>{JSON.stringify(modelInfo?.hyperparameters ?? {}, null, 2)}</pre>
          </details>}
        </section>

        {sampleDetails && (
          <>
            <p className="sample-note">Loaded held-out patient #{sampleDetails.patient_index} · reference label: {sampleDetails.label === 1 ? 'Sepsis' : 'No Sepsis'}</p>
            {sampleDetails.pool_note && <p className="sample-note warning">{sampleDetails.pool_note}</p>}
            <p className="sample-note info">{windowId ? 'Loaded patient: scored on full 6-hour window' : 'Manual vitals: scored on 6-vital snapshot (approximate)'}</p>
          </>
        )}

        <div className="workspace-grid">
          <section className="card input-card" aria-labelledby="input-heading">
            <div className="card-heading">
              <div className="icon-circle">⌁</div>
              <div><p className="section-kicker">Input</p><h2 id="input-heading">Current vital measurements</h2></div>
            </div>
            <form onSubmit={baseline ? (event) => { event.preventDefault(); compare() } : predict}>
              <div className="vital-grid">
                {vitalDefinitions.map((vital) => (
                  <label className="vital-control" key={vital.key}>
                    <span className="vital-label">{vital.label}</span>
                    <div className="number-entry">
                      <input
                        aria-label={vital.label}
                        type="number"
                        min={vital.min}
                        max={vital.max}
                        step={vital.step}
                        value={vitals[vital.key]}
                        onChange={(event) => updateVital(vital.key, event.target.value)}
                      />
                      <span>{vital.unit}</span>
                    </div>
                    <input
                      className="range-input"
                      type="range"
                      min={vital.min}
                      max={vital.max}
                      step={vital.step}
                      value={vitals[vital.key]}
                      onChange={(event) => updateVital(vital.key, event.target.value)}
                    />
                    <span className="range-labels"><span>{vital.min}</span><span>{vital.max}</span></span>
                  </label>
                ))}
              </div>
              <button className="primary-button" disabled={isPredicting} type="submit">
                {isPredicting ? <><span className="spinner" /> Calculating risk…</> : baseline ? 'Compare One Vital Change →' : 'Predict Sepsis Risk →'}
              </button>
            </form>
            {baseline && <button className="text-button reset-button" onClick={resetBaseline} type="button">Set a new baseline prediction</button>}
            {error && <p className="error-message" role="alert">{error}</p>}
          </section>

          {sampleDetails && labGroups.length > 0 && (
            <section className="card lab-panel-card">
              <div className="card-heading">
                <div className="icon-circle">◈</div>
                <div><p className="section-kicker">Full patient data</p><h2>Full Lab Panel</h2></div>
              </div>
              <button
                className="text-button expand-button"
                onClick={() => setLabPanelExpanded(!labPanelExpanded)}
                type="button"
              >
                {labPanelExpanded ? '▼ Collapse' : '▶ Expand full lab panel'}
              </button>
              {labPanelExpanded && (
                <>
                  <p className="lab-legend">
                    <span className="lab-legend-measured">●</span> Actually measured
                    <span className="lab-legend-imputed">○</span> Imputed/estimated
                  </p>
                  {labGroups.map((group) => (
                    <div key={group.category} className="lab-category">
                      <h3 className="lab-category-title">{group.category}</h3>
                      <div className="lab-features-grid">
                        {(group?.features ?? []).map((feature) => (
                          <div
                            key={feature.name}
                            className={`lab-feature ${feature.was_measured ? 'measured' : 'imputed'}`}
                            title={feature.was_measured ? 'Actually measured' : 'Imputed/estimated'}
                          >
                            <span className="lab-feature-name">{feature.name}</span>
                            <span className="lab-feature-value">
                              {feature.value === null
                                ? 'Not measured'
                                : feature.name === 'Gender'
                                ? feature.value === 1 ? 'Male' : 'Female'
                                : feature.name === 'Age'
                                ? Math.round(feature.value)
                                : feature.value.toFixed(2)}
                            </span>
                            <span className="lab-feature-unit">{feature.unit}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </>
              )}
            </section>
          )}

          <section className={`card results-card ${result ? 'has-result' : ''}`} aria-live="polite" aria-labelledby="result-heading">
            <div className="card-heading">
              <div className="icon-circle result-icon">✦</div>
              <div><p className="section-kicker">Model output</p><h2 id="result-heading">Risk assessment</h2></div>
            </div>
            {!result ? (
              <div className="empty-result"><div className="empty-icon">⌁</div><p>Enter measurements and select <strong>Predict Sepsis Risk</strong> to view the model assessment.</p></div>
            ) : (
              <>
                <div className="model-score-compare">
                  <div><span>XGBoost (monotonic)</span><strong>{(displayedResults.xgb.risk_score * 100).toFixed(2)}%</strong></div>
                  <div><span>Random Forest</span><strong>{(displayedResults.rf.risk_score * 100).toFixed(2)}%</strong></div>
                  <div><span>Gradient Boosting</span><strong>{(displayedResults.gb.risk_score * 100).toFixed(2)}%</strong></div>
                  <div><span>AdaBoost</span><strong>{(displayedResults.adaboost.risk_score * 100).toFixed(2)}%</strong></div>
                </div>
                {comparison && (
                  <div className={`what-if ${delta >= 0 ? 'increased' : 'decreased'}`}>
                    <p className="section-kicker">What-if comparison · {selectedModel === 'xgb' ? 'XGBoost' : selectedModel === 'rf' ? 'Random Forest' : selectedModel === 'gb' ? 'Gradient Boosting' : 'AdaBoost'}</p>
                    <strong>{comparison.changed.label}: {baseline.vitals[comparison.changed.key]} → {comparison.vitals[comparison.changed.key]}</strong>
                    <span>{(activeBaseline.risk_score * 100).toFixed(4)}% → {(activeComparison.risk_score * 100).toFixed(4)}% &nbsp; <b>{formatDeltaPercent(delta)}</b> risk</span>
                  </div>
                )}
                <div className="risk-summary">
                  <div><p className="risk-caption">ESTIMATED RISK SCORE</p><p className="risk-value">{(result.risk_score * 100).toFixed(1)}<span>%</span></p></div>
                  <div className={`risk-badge ${risk.tone}`}><span className="badge-dot" />{risk.label}</div>
                </div>
                <p className="band-note">Raw model score, not a calibrated probability. Band is a provisional display band.</p>
                <div className={`prediction-label ${risk.tone}`}>
                  <span>{risk.tone === 'high' ? '▲' : risk.tone === 'medium' ? '•' : '✓'}</span>
                  {risk.tone === 'high' ? 'Elevated Risk' : risk.tone === 'medium' ? 'Intermediate Risk' : 'Low Risk'}
                </div>
                <div className="explanation-section">
                  <div className="explanation-title"><h3>What influenced this prediction?</h3><span>SHAP values</span></div>
                  <div className="shap-chart">
                    {result.top_features.map((item) => {
                      const percentage = (Math.abs(item.shap_value) / maximumShap) * 100
                      const pushesUp = item.direction === 'increases risk'
                      return (
                        <div className="shap-row" key={item.feature}>
                          <div className="shap-name" title={item.feature}>{friendlyFeature(item.feature)}</div>
                          <div className="shap-bar-track"><div className={`shap-bar ${pushesUp ? 'up' : 'down'}`} style={{ width: `${percentage}%` }} /></div>
                          <span className={`shap-direction ${pushesUp ? 'up' : 'down'}`}>{pushesUp ? '↑ Risk' : '↓ Risk'}</span>
                        </div>
                      )
                    })}
                  </div>
                  <p className="legend"><span className="legend-up" /> Pushes risk up <span className="legend-down" /> Pushes risk down</p>
                </div>
              </>
            )}
          </section>
        </div>
        {subtypeSummary && (
          <section className="card subtype-section" aria-labelledby="subtype-heading">
            <div className="subtype-heading">
              <div className="icon-circle subtype-icon">◌</div>
              <div>
                <p className="section-kicker">Exploratory cohort view</p>
                <h2 id="subtype-heading">Patient Subtypes</h2>
                <p>Discovered via unsupervised K-means clustering (k={subtypeSummary.n_clusters}, silhouette 0.30) on 184 septic patient profiles</p>
              </div>
            </div>
            <div className="subtype-grid">
              {subtypeSummary.clusters.map((cluster) => (
                <article className="subtype-card" key={cluster.cluster_id}>
                  <p className="subtype-id">SUBTYPE {cluster.cluster_id + 1}</p>
                  <h3>{subtypeLabel(cluster)}</h3>
                  <p className="subtype-size">{cluster.size} patients</p>
                  <div className="feature-tags">
                    {cluster.top_features.slice(0, 4).map((feature) => (
                      <span key={feature.feature} className={feature.description.startsWith('Higher') ? 'feature-tag higher' : 'feature-tag lower'}>{subtypeTag(feature)}</span>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}
        <p className="model-note">Prototype note: the percentage is a calibrated probability. Green/yellow/red are cohort-relative display bands for this imbalanced prototype, not clinical decision thresholds.</p>
      </main>
      <footer>Future modules planned: mortality prediction, patient subtyping, causal inference, GAN-based data augmentation</footer>
    </div>
  )
}

export default App
