// lib/types.ts
export interface Deficiency {
  nutrient: string;
  probability: number;
  evidence?: string[];
}

export interface Food {
  name: string;
  per_serving?: Record<string, number>;
  reason: string;
}

export interface Recipe {
  id: number;
  title: string;
  servings: number;
  calories_per_serving: number;
  nutrient_breakdown?: Record<string, number>;
  preparation_summary: string;
  flags?: string[];
}

export interface RecommendationResponse {
  deficiencies: Deficiency[];
  foods: Food[];
  recipes: Recipe[];
  explanation: string;
  safety_notes: string[];
  citations?: { id: string; title: string }[];
}

// Interface for the diet plan response from our backend
export interface DietPlanResponse {
  plan: string;
}
