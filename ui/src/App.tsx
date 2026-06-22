import { useState, useEffect, useCallback } from 'react'
import { api, type FoodInfo, type PredictionResponse } from './api'

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const RISK_COLORS: Record<string, { bg: string; text: string; border: string; bar: string }> = {
  low:        { bg: 'bg-emerald-50',  text: 'text-emerald-700',  border: 'border-emerald-300', bar: '#10b981' },
  moderate:   { bg: 'bg-amber-50',   text: 'text-amber-700',   border: 'border-amber-300',  bar: '#f59e0b' },
  high:       { bg: 'bg-orange-50',  text: 'text-orange-700',  border: 'border-orange-300', bar: '#f97316' },
  very_high:  { bg: 'bg-red-50',     text: 'text-red-700',     border: 'border-red-300',    bar: '#ef4444' },
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function bmiCategoryFromBmi(bmi: number): string {
  if (bmi < 18.5) return 'underweight'
  if (bmi < 25) return 'normal'
  if (bmi < 30) return 'overweight'
  return 'obese'
}

function riskScore(risk: string): number {
  // Map risk label to a 0-100 gauge value
  const map: Record<string, number> = { low: 20, moderate: 50, high: 75, very_high: 95 }
  return map[risk] ?? 0
}

/* ------------------------------------------------------------------ */
/*  Components                                                         */
/* ------------------------------------------------------------------ */

function RiskGauge({ risk }: { risk: string }) {
  const score = riskScore(risk)
  const colors = RISK_COLORS[risk] ?? RISK_COLORS.moderate

  // SVG arc gauge: 180° semicircle
  const r = 80
  const cx = 100
  const cy = 95
  const circumference = Math.PI * r // half circle
  const offset = circumference * (1 - score / 100)

  return (
    <div className="flex flex-col items-center gap-2">
      <svg viewBox="0 0 200 120" className="w-48 h-32">
        {/* Background arc */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {/* Filled arc */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={colors.bar}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-700 ease-out"
        />
        {/* Score text */}
        <text x={cx} y={cy - 10} textAnchor="middle" className="text-2xl font-bold" fill={colors.bar}>
          {score}
        </text>
        <text x={cx} y={cy + 8} textAnchor="middle" className="text-xs" fill="#94a3b8">
          / 100
        </text>
      </svg>
      <span className={`px-4 py-1.5 rounded-full text-sm font-semibold border ${colors.bg} ${colors.text} ${colors.border} transition-colors duration-300`}>
        {risk.replace('_', ' ').toUpperCase()}
      </span>
    </div>
  )
}

function PortionMeter({ portion }: { portion: number }) {
  const max = 3
  const pct = Math.min((portion / max) * 100, 100)
  const barColor = portion > 1.5 ? '#22c55e' : portion > 0.8 ? '#f59e0b' : '#ef4444'

  return (
    <div className="flex flex-col items-center gap-2 w-full max-w-xs">
      <div className="text-xs text-slate-500 uppercase tracking-wide font-medium">Safe Portions</div>
      <div className="relative w-full h-4 bg-slate-200 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
      </div>
      <div className="text-3xl font-bold text-slate-800">
        {portion.toFixed(2)}
        <span className="text-sm font-normal text-slate-500 ml-1">servings</span>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Main App                                                           */
/* ------------------------------------------------------------------ */

export default function App() {
  // Dropdown data
  const [foods, setFoods] = useState<FoodInfo[]>([])
  const [regions, setRegions] = useState<string[]>([])
  // Form state
  const [age, setAge] = useState(45)
  const [bmi, setBmi] = useState(27.5)
  const [diabetes, setDiabetes] = useState(false)
  const [fasting, setFasting] = useState(false)
  const [foodIdx, setFoodIdx] = useState(0)
  const [selectedRegion, setSelectedRegion] = useState('')

  // Prediction state
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch catalog on mount
  useEffect(() => {
    Promise.all([api.foods(), api.regions()])
      .then(([f, r]) => {
        setFoods(f)
        setRegions(r)
        setSelectedRegion(r[0] ?? 'Maharashtra')
      })
      .catch(() => setError('Could not connect to API. Make sure the backend is running on :8000'))
  }, [])

  const selectedFood = foods[foodIdx]

  // Auto-select region when food changes
  useEffect(() => {
    if (selectedFood && !regions.includes(selectedRegion)) {
      setSelectedRegion(selectedFood.region)
    }
  }, [selectedFood, selectedRegion, regions])

  const predict = useCallback(async () => {
    if (!selectedFood) return
    setLoading(true)
    setError(null)
    try {
      const res = await api.predict({
        age,
        bmi,
        diabetes_status: diabetes ? 1 : 0,
        fasting_state: fasting ? 1 : 0,
        bmi_category: bmiCategoryFromBmi(bmi),
        festival: selectedFood.festival,
        region: selectedRegion,
        food_name: selectedFood.food_name,
        glycemic_index: selectedFood.glycemic_index,
        carbs_per_item_g: selectedFood.carbs_per_item_g,
        sugar_per_item_g: selectedFood.sugar_per_item_g,
        protein_per_item_g: selectedFood.protein_per_item_g,
        fat_per_item_g: selectedFood.fat_per_item_g,
        fiber_per_item_g: selectedFood.fiber_per_item_g,
        energy_per_item_kcal: selectedFood.energy_per_item_kcal,
      })
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }, [age, bmi, diabetes, fasting, selectedFood, selectedRegion])

  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-50 via-white to-emerald-50">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-md sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center text-white font-bold text-lg shadow-md">
              A
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-800 leading-tight">Aharamitra</h1>
              <p className="text-xs text-slate-500">Food Risk Intelligence</p>
            </div>
          </div>
          <span className="text-xs text-slate-400">v0.2 · ML-powered</span>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Error banner */}
        {error && !result && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm">
            ⚠️ {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
          {/* ── Left: Form (3 cols) ── */}
          <div className="lg:col-span-3 space-y-6">
            {/* Food selection */}
            <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-base font-semibold text-slate-800 mb-4">🍽️ Select Festival Food</h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-600 mb-1">Food Item</label>
                  <select
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                    value={foodIdx}
                    onChange={e => setFoodIdx(Number(e.target.value))}
                  >
                    {foods.map((f, i) => (
                      <option key={f.food_name} value={i}>{f.food_name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-600 mb-1">Region</label>
                  <select
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                    value={selectedRegion}
                    onChange={e => setSelectedRegion(e.target.value)}
                  >
                    {regions.map(r => (
                      <option key={r} value={r}>{r}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Food nutrition card */}
              {selectedFood && (
                <div className="mt-4 p-3 bg-slate-50 rounded-xl grid grid-cols-3 sm:grid-cols-5 gap-2 text-center text-xs">
                  <Nutrient label="GI" value={selectedFood.glycemic_index.toFixed(0)} />
                  <Nutrient label="Carbs" value={`${selectedFood.carbs_per_item_g.toFixed(1)}g`} />
                  <Nutrient label="Sugar" value={`${selectedFood.sugar_per_item_g.toFixed(1)}g`} />
                  <Nutrient label="Protein" value={`${selectedFood.protein_per_item_g.toFixed(1)}g`} />
                  <Nutrient label="Energy" value={`${selectedFood.energy_per_item_kcal.toFixed(0)}kcal`} />
                </div>
              )}
            </section>

            {/* User profile */}
            <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-base font-semibold text-slate-800 mb-4">👤 Your Health Profile</h2>
              <div className="space-y-5">
                {/* Age */}
                <SliderField label="Age" value={age} min={18} max={80} step={1} unit="years"
                  onChange={setAge} />

                {/* BMI */}
                <SliderField label="Body Mass Index (BMI)" value={bmi} min={16} max={42} step={0.1} unit={bmiCategoryFromBmi(bmi)}
                  onChange={setBmi} />

                {/* Toggles */}
                <div className="flex flex-wrap gap-4">
                  <Toggle label="Diabetic" checked={diabetes} onChange={setDiabetes} />
                  <Toggle label="Fasting" checked={fasting} onChange={setFasting} />
                </div>
              </div>
            </section>

            {/* Predict button */}
            <button
              onClick={predict}
              disabled={loading || foods.length === 0}
              className="w-full py-3.5 rounded-xl font-semibold text-white bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-emerald-200 transition-all duration-200 text-sm"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
                  Analyzing...
                </span>
              ) : (
                '🚀 Analyze Food Risk'
              )}
            </button>
          </div>

          {/* ── Right: Results (2 cols) ── */}
          <div className="lg:col-span-2">
            <div className="sticky top-24 space-y-6">
              {result ? (
                <>
                  {/* Risk gauge */}
                  <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 text-center">
                    <h2 className="text-sm font-semibold text-slate-600 mb-1">Glucose Spike Risk</h2>
                    <p className="text-xs text-slate-400 mb-4">{result.food_name} · {result.region}</p>
                    <RiskGauge risk={result.glucose_spike_risk} />
                    {result.confidence && (
                      <p className="mt-2 text-xs text-slate-500">Model confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong></p>
                    )}
                  </section>

                  {/* Portion meter */}
                  <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                    <PortionMeter portion={result.safe_portion_count} />
                  </section>

                  {/* Summary card */}
                  <section className="bg-gradient-to-br from-emerald-50 to-teal-50 rounded-2xl border border-emerald-200 p-5">
                    <h3 className="text-sm font-semibold text-emerald-800 mb-3">📋 Recommendation</h3>
                    <p className="text-sm text-emerald-700 leading-relaxed">
                      Based on the analysis, consuming <strong>{result.food_name}</strong> during{' '}
                      <strong>{result.festival}</strong> carries a{' '}
                      <strong className="text-red-600">{result.glucose_spike_risk.replace('_', ' ')}</strong>{' '}
                      glucose-spike risk for this health profile.
                      {result.safe_portion_count >= 1.0 && (
                        <> You can safely enjoy up to <strong>{result.safe_portion_count.toFixed(1)} portions</strong>.</>
                      )}
                      {result.safe_portion_count < 1.0 && (
                        <> Limit intake to <strong>{result.safe_portion_count.toFixed(1)} portions</strong> or consider an alternative.</>
                      )}
                    </p>
                  </section>
                </>
              ) : (
                <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-12 text-center">
                  <div className="text-4xl mb-3">🍽️</div>
                  <p className="text-sm text-slate-500">Select a food and profile, then click <strong>Analyze</strong> to see your personalized risk assessment.</p>
                </section>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-12 border-t border-slate-200 py-6 text-center text-xs text-slate-400">
        Aharamitra v0.2 · AI-Based Food Risk & Portion Intelligence · Not a substitute for medical advice
      </footer>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Small reusable components                                          */
/* ------------------------------------------------------------------ */

function SliderField({ label, value, min, max, step, unit, onChange }: {
  label: string; value: number; min: number; max: number; step: number; unit: string;
  onChange: (v: number) => void
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-xs font-medium text-slate-600">{label}</span>
        <span className="text-sm font-bold text-slate-800">{value}{step >= 1 ? '' : ''} <span className="text-xs font-normal text-slate-500">{unit}</span></span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full cursor-pointer"
      />
    </div>
  )
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={`px-4 py-2 rounded-lg text-sm font-medium border transition-all duration-200 ${
        checked
          ? 'bg-emerald-50 border-emerald-300 text-emerald-700'
          : 'bg-white border-slate-300 text-slate-500 hover:border-slate-400'
      }`}
    >
      {checked ? '✓ ' : ''}{label}
    </button>
  )
}

function Nutrient({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white rounded-lg border border-slate-200 py-2">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-sm font-semibold text-slate-800">{value}</div>
    </div>
  )
}
