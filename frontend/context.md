# Context: Personalised Indian Thali – Nutrition Assistant

## Problem Statement

Many people in India experience nutrient deficiencies (iron, vitamin B12,
protein, etc.) that worsen conditions like fatigue, diabetes, anemia, or
digestive issues. Existing diet apps are generic and don’t map
**diseases/symptoms → nutrient deficiencies → food items → culturally relevant
recipes**.

This project aims to build a **Personalised Indian Thali Nutrition Assistant**,
powered by an **LLM/RAG system**, that recommends **nutrient-balanced Indian
recipes** personalized for users’ health conditions.

---

## Key Objectives

1. Accept **symptoms/disease input** (e.g., fatigue, diabetes, bloating).
2. Map to **probable nutrient deficiencies** using medical/nutrition guidelines
   (ICMR, WHO).
3. Identify **foods** that can help address deficiencies.
4. Suggest **multiple recipes** (not just one) that are:
   - Nutritionally aligned
   - Indian/thali-style
   - Diverse (veg/non-veg, seasonal, affordable options)
5. Provide **explanations, safety notes, and citations** (not a medical
   diagnosis).

---

## System Architecture

### 1. Input Layer

- **User Profile**: age, gender, dietary preferences (veg/non-veg, vegan, Jain),
  allergies, location.
- **Health Input**: symptoms (free text or checklist), known conditions
  (diabetes, PCOS, anemia).

### 2. Knowledge Base (for RAG)

- Medical mappings (symptoms ↔ nutrient deficiencies).
- Indian foods and their nutrient composition (ICMR, FSSAI tables).
- Recipes database (JSON/YAML format with nutrient breakdown, regional
  categorization).

### 3. LLM + RAG Pipeline

1. **Parse symptoms** → retrieve relevant deficiencies.
2. **Retrieve foods** from food knowledge base.
3. **Generate recipe suggestions** by combining foods into culturally relevant
   Indian dishes.
4. **Rank recipes**:
   - Nutrient coverage score (how well they address deficiencies).
   - User preferences (veg/non-veg, time to cook).
   - Diversity (show at least 2–3 useful recipes).
5. **Explain reasoning** + include **citations & safety notes**.

### 4. Backend (API Layer)

- Stack: **FastAPI**.
- Exposes endpoints:
  - `/analyze_symptoms` → returns deficiencies.
  - `/get_foods` → foods rich in required nutrients.
  - `/get_recipes` → top N recipes with nutrition breakdown.
  - `/feedback` → user feedback loop for improvement.
- Integrates with vector DB (Pinecone, Weaviate, FAISS) for RAG retrieval.
- Stores user preferences/history (MongoDB or PostgreSQL).

### 5. Frontend

- Stack: **Next.js (React)**, TailwindCSS.
- Features:
  - Symptom entry (text box + checklist).
  - Recipe recommendations page (with thali visualization).
  - Nutrient breakdown chart (iron, protein, vitamins, etc.).
  - Explanations, safety notes, and citations.
  - Filters (veg/non-veg, cooking time, ingredients available).

---

## Example Response Format (API → Frontend)

```json
{
  "deficiencies": [
    {
      "nutrient": "Iron",
      "probability": 0.88,
      "evidence": ["icmr_iron_summary"]
    },
    {
      "nutrient": "Vitamin B12",
      "probability": 0.62,
      "evidence": ["who_b12_deficiency"]
    }
  ],
  "foods": [
    {
      "name": "Palak (spinach)",
      "per_serving": { "iron_mg": 3.6 },
      "reason": "High iron + vitamin C pairing improves absorption"
    },
    {
      "name": "Paneer",
      "per_serving": { "protein_g": 18 },
      "reason": "Protein + calcium support for anemia recovery"
    }
  ],
  "recipes": [
    {
      "id": 101,
      "title": "Palak Paneer Thali",
      "servings": 2,
      "calories_per_serving": 420,
      "nutrient_breakdown": { "iron_mg": 4.1, "protein_g": 22 },
      "preparation_summary": "Paneer + spinach curry; serve with bajra roti or brown rice",
      "flags": ["vegetarian"]
    },
    {
      "id": 102,
      "title": "Rajma Chawal with Salad",
      "servings": 2,
      "calories_per_serving": 480,
      "nutrient_breakdown": { "iron_mg": 3.5, "protein_g": 19 },
      "preparation_summary": "Kidney beans curry with rice and lemon salad",
      "flags": ["vegetarian", "high-fiber"]
    }
  ],
  "explanation": "Symptoms like fatigue and pallor can indicate iron and B12 deficiency. Suggested recipes include iron-rich ingredients and protein to improve absorption.",
  "safety_notes": [
    "If severe anemia symptoms, consult a physician; not a clinical diagnosis."
  ],
  "citations": [
    { "id": "icmr_iron_summary", "title": "ICMR Nutrient Guidelines" },
    { "id": "who_b12_deficiency", "title": "WHO B12 Deficiency Report" }
  ]
}
```
