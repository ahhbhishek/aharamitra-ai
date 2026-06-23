import { useState, useEffect, useCallback } from 'react'
import { api, type FoodInfo, type PredictionResponse } from './api'

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

// Risk label -> visual treatment for the verdict badge.
const RISK_STYLES: Record<string, { bg: string; text: string; border: string; emoji: string }> = {
  low:       { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-300', emoji: '🟢' },
  moderate:  { bg: 'bg-amber-50',   text: 'text-amber-700',   border: 'border-amber-300',  emoji: '🟡' },
  high:      { bg: 'bg-orange-50',  text: 'text-orange-700',  border: 'border-orange-300', emoji: '🟠' },
  very_high: { bg: 'bg-red-50',     text: 'text-red-700',     border: 'border-red-300',    emoji: '🔴' },
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/** BMI from height (cm) and weight (kg). */
function computeBmi(heightCm: number, weightKg: number): number {
  const m = heightCm / 100
  return weightKg / (m * m)
}

/** WHO BMI category label. */
function bmiCategory(bmi: number): string {
  if (bmi < 18.5) return 'underweight'
  if (bmi < 25) return 'normal'
  if (bmi < 30) return 'overweight'
  return 'obese'
}

/* ------------------------------------------------------------------ */
/*  Main App                                                           */
/* ------------------------------------------------------------------ */

export default function App() {
  // Dropdown data
  const [foods, setFoods] = useState<FoodInfo[]>([])
  const [regions, setRegions] = useState<string[]>([])
  // Festivals are derived from the foods catalog (preserves source of truth).
  const festivals = Array.from(new Set(foods.map(f => f.festival))).sort()

  // Form state — natural measurements (no raw BMI input)
  const [gender, setGender] = useState<'male' | 'female'>('male')
  const [age, setAge] = useState(45)
  const [heightCm, setHeightCm] = useState(170)
  const [weightKg, setWeightKg] = useState(78)
  const [diabetes, setDiabetes] = useState(false)
  const [fasting, setFasting] = useState(false)

  // Selection state — guided flow: festival → region → food (filtered by festival)
  const [selectedFestival, setSelectedFestival] = useState('')
  const [selectedRegion, setSelectedRegion] = useState('')
  const [selectedFoodName, setSelectedFoodName] = useState('')

  // Prediction state
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Derived BMI for the live readout (computed in browser, validated server-side)
  const bmi = computeBmi(heightCm, weightKg)

  // Fetch catalog on mount
  useEffect(() => {
    Promise.all([api.foods(), api.regions()])
      .then(([f, r]) => {
        setFoods(f)
        setRegions(r)
        // Seed defaults from the first festival in the catalog.
        const firstFestival = f[0]?.festival ?? ''
        setSelectedFestival(firstFestival)
        setSelectedRegion(r[0] ?? 'Maharashtra')
        setSelectedFoodName(f[0]?.food_name ?? '')
      })
      .catch(() => setError('Could not connect to API. Make sure the backend is running on :8000'))
  }, [])

  // Foods for the currently selected festival.
  const festivalFoods = foods.filter(f => f.festival === selectedFestival)

  const selectedFood = foods.find(f => f.food_name === selectedFoodName) ?? foods[0]

  // When the festival changes, jump to the first food of that festival.
  const onFestivalChange = (festival: string) => {
    setSelectedFestival(festival)
    const first = foods.find(f => f.festival === festival)
    if (first) setSelectedFoodName(first.food_name)
  }

  const predict = useCallback(async () => {
    if (!selectedFood) return
    setLoading(true)
    setError(null)
    try {
      const res = await api.predict({
        gender,
        age,
        height_cm: heightCm,
        weight_kg: weightKg,
        diabetes_status: diabetes ? 1 : 0,
        fasting_state: fasting ? 1 : 0,
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
  }, [gender, age, heightCm, weightKg, diabetes, fasting, selectedFood, selectedRegion])

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
            {/* Festival context selection */}
            <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-base font-semibold text-slate-800 mb-4">🎉 Choose Festival &amp; Food</h2>
              <div className="space-y-4">
                {/* Festival → drives the food list */}
                <div>
                  <label className="block text-xs font-medium text-slate-600 mb-1">Festival</label>
                  <select
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                    value={selectedFestival}
                    onChange={e => onFestivalChange(e.target.value)}
                  >
                    {festivals.map(f => (
                      <option key={f} value={f}>{f}</option>
                    ))}
                  </select>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Region */}
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
                  {/* Food — filtered to the chosen festival */}
                  <div>
                    <label className="block text-xs font-medium text-slate-600 mb-1">
                      Food {festivalFoods.length > 0 && <span className="text-slate-400">({festivalFoods.length})</span>}
                    </label>
                    <select
                      className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                      value={selectedFoodName}
                      onChange={e => setSelectedFoodName(e.target.value)}
                    >
                      {festivalFoods.map(f => (
                        <option key={f.food_name} value={f.food_name}>{f.food_name}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Food nutrition card */}
              {selectedFood && (
                <div className="mt-4 p-3 bg-slate-50 rounded-xl grid grid-cols-3 sm:grid-cols-5 gap-2 text-center text-xs">
                  <Nutrient label="Per piece" value={`${selectedFood.weight_g}g`} />
                  <Nutrient label="Carbs" value={`${selectedFood.carbs_per_item_g.toFixed(0)}g`} />
                  <Nutrient label="Sugar" value={`${selectedFood.sugar_per_item_g.toFixed(0)}g`} />
                  <Nutrient label="Protein" value={`${selectedFood.protein_per_item_g.toFixed(1)}g`} />
                  <Nutrient label="Energy" value={`${selectedFood.energy_per_item_kcal.toFixed(0)}kcal`} />
                </div>
              )}
            </section>

            {/* User profile */}
            <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-base font-semibold text-slate-800 mb-4">👤 Your Health Profile</h2>
              <div className="space-y-5">
                {/* Gender toggle */}
                <div>
                  <span className="block text-xs font-medium text-slate-600 mb-2">Gender</span>
                  <div className="inline-flex rounded-lg border border-slate-300 overflow-hidden">
                    {(['male', 'female'] as const).map(g => (
                      <button
                        key={g}
                        type="button"
                        onClick={() => setGender(g)}
                        className={`px-5 py-2 text-sm font-medium capitalize transition-colors ${
                          gender === g
                            ? 'bg-emerald-600 text-white'
                            : 'bg-white text-slate-500 hover:bg-slate-50'
                        }`}
                      >
                        {g}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Age + Height + Weight */}
                <SliderField label="Age" value={age} min={18} max={80} step={1} unit="years"
                  onChange={setAge} />
                <SliderField label="Height" value={heightCm} min={130} max={200} step={1} unit="cm"
                  onChange={setHeightCm} />
                <SliderField label="Weight" value={weightKg} min={35} max={140} step={1} unit="kg"
                  onChange={setWeightKg} />

                {/* Live BMI readout */}
                <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <span className="text-xs font-medium text-slate-500">Your BMI</span>
                  <span className="text-sm font-bold text-slate-800">
                    {bmi.toFixed(1)} <span className="text-xs font-normal text-slate-500 capitalize">({bmiCategory(bmi)})</span>
                  </span>
                </div>

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

          {/* ── Right: Result card (2 cols) ── */}
          <div className="lg:col-span-2">
            <div className="sticky top-24">
              {result ? (
                <VerdictCard result={result} />
              ) : (
                <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-12 text-center">
                  <div className="text-4xl mb-3">🍽️</div>
                  <p className="text-sm text-slate-500">Select a food and your profile, then click <strong>Analyze</strong> to see how much you can safely eat — and why.</p>
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
/*  Result card — verdict + how much + why                             */
/* ------------------------------------------------------------------ */

function VerdictCard({ result }: { result: PredictionResponse }) {
  const style = RISK_STYLES[result.glucose_spike_risk] ?? RISK_STYLES.moderate

  return (
    <section className={`rounded-2xl shadow-sm border-2 p-6 ${style.bg} ${style.border}`}>
      {/* Verdict header */}
      <div className="flex items-center gap-3 mb-1">
        <span className="text-2xl">{style.emoji}</span>
        <h2 className={`text-2xl font-extrabold ${style.text}`}>{result.verdict}</h2>
      </div>
      <p className="text-xs text-slate-500 mb-5">{result.food_name} · {result.festival}</p>

      {/* How much */}
      <div className="bg-white/70 rounded-xl p-4 mb-4">
        <p className="text-xs font-medium text-slate-500 mb-1">How much can you eat?</p>
        <p className="text-3xl font-bold text-slate-800">
          ≈ {result.safe_grams}g
          <span className="text-base font-medium text-slate-500 ml-2">({result.safe_pieces})</span>
        </p>
        <p className="text-xs text-slate-500 mt-1">
          That&apos;s about {result.sugar_g}g sugar · {result.carbs_g}g carbs · {result.energy_kcal} cal
        </p>
      </div>

      {/* Why */}
      <div className="bg-white/70 rounded-xl p-4">
        <p className="text-xs font-medium text-slate-500 mb-2">Why this matters for you</p>
        <ul className="space-y-1.5">
          {result.reasons.map((r, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
              <span className="text-slate-400 mt-0.5">•</span>
              <span>{r}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
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
        <span className="text-sm font-bold text-slate-800">
          {value} <span className="text-xs font-normal text-slate-500">{unit}</span>
        </span>
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
