// API client for the Aharamitra FastAPI backend.

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

export interface FoodInfo {
  food_name: string
  festival: string
  region: string
  glycemic_index: number
  carbs_per_item_g: number
  sugar_per_item_g: number
  protein_per_item_g: number
  fat_per_item_g: number
  fiber_per_item_g: number
  energy_per_item_kcal: number
}

export interface PredictionResponse {
  glucose_spike_risk: string
  risk_encoded: number
  safe_portion_count: number
  confidence: number | null
  food_name: string
  festival: string
  region: string
}

export interface PredictionRequest {
  age: number
  bmi: number
  diabetes_status: number
  fasting_state: number
  bmi_category: string
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
