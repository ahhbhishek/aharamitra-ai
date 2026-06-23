// API client for the Aharamitra FastAPI backend.

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

export interface FoodInfo {
  food_name: string
  festival: string
  region: string
  glycemic_index: number
  weight_g: number
  carbs_per_item_g: number
  sugar_per_item_g: number
  protein_per_item_g: number
  fat_per_item_g: number
  fiber_per_item_g: number
  energy_per_item_kcal: number
}

export interface PredictionResponse {
  glucose_spike_risk: string   // low | moderate | high | very_high
  verdict: string             // Enjoy | Go easy | Limit | Avoid
  safe_grams: number          // grams of this food safe to eat
  safe_pieces: string         // human-friendly: "1 piece", "1-2 pieces", ...
  sugar_g: number             // total sugar in safe_grams
  carbs_g: number             // total carbs in safe_grams
  energy_kcal: number         // total calories in safe_grams
  glycemic_load: number       // the food's GL (for the "why" section)
  reasons: string[]           // 2-3 short, personalized bullets
  food_name: string
  festival: string
  region: string
}

export interface PredictionRequest {
  // Natural profile inputs (BMI is derived server-side from height/weight).
  gender: string              // 'male' | 'female'
  age: number
  height_cm: number
  weight_kg: number
  diabetes_status: number     // 1 | 0
  fasting_state: number       // 1 | 0
  // Food + context
  festival: string
  region: string
  food_name: string
  glycemic_index: number
  carbs_per_item_g: number
  sugar_per_item_g: number
  protein_per_item_g: number
  fat_per_item_g: number
  fiber_per_item_g: number
  energy_per_item_kcal: number
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`)
  return res.json() as Promise<T>
}

export const api = {
  foods: () => getJson<FoodInfo[]>('/foods'),
  regions: () => getJson<string[]>('/regions'),
  festivals: () => getJson<string[]>('/festivals'),
  predict: async (req: PredictionRequest): Promise<PredictionResponse> => {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail ?? `predict failed: ${res.status}`)
    }
    return res.json()
  },
}
