import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai

from app.models.schemas import UserProfile

# --- 0. Configure Gemini API ---
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please add it.")
genai.configure(api_key=GEMINI_API_KEY)


# --- PHASE 1: DATA PREPROCESSING & MODEL LOADING (runs once on startup) ---

def load_data(file_name):
    """Loads a CSV file from the data directory."""
    try:
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', file_name)
        df = pd.read_csv(path)
        print(f"✅ Dataset '{file_name}' loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"❌ Error: The file '{file_name}' was not found at {path}")
        return pd.DataFrame()

# Load all datasets
disease_df = load_data('disease_nutrients.csv')
food_df = load_data('Food-nutrient_dataset.csv')
abbreviations_df = load_data('nutrients_abbreviations.csv')

# Load Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ SentenceTransformer model loaded.")

# --- Data Cleaning and Transformation ---

# 1. Create Nutrient Abbreviation Mapper
if not abbreviations_df.empty:
    abbreviations_df.columns = [col.strip().lower() for col in abbreviations_df.columns]
    nutrient_mapper = pd.Series(abbreviations_df['name'].values, index=abbreviations_df['code']).to_dict()
    print("✅ Nutrient abbreviation mapper created.")
else:
    nutrient_mapper = {}

# 2. Create the Scientific Nutrient Corpus and Embeddings for Mapping
if not abbreviations_df.empty:
    # This is our target list of official nutrient names
    target_nutrient_corpus = abbreviations_df['name'].dropna().unique().tolist()
    # Create embeddings for each of these official names
    nutrient_embeddings = model.encode(target_nutrient_corpus, convert_to_tensor=True)
    print(f"✅ Embeddings created for {len(target_nutrient_corpus)} target nutrients.")
else:
    target_nutrient_corpus = []
    nutrient_embeddings = torch.Tensor()

# 3. Clean and Process Food Nutrient Data
def process_food_data(df, mapper):
    if df.empty or not mapper:
        return {}
    
    df.columns = df.columns.str.lower()
    nutrient_cols = [col for col in df.columns if not col.endswith('_e') and col in mapper]
    
    food_to_nutrients = {}
    for _, row in df.iterrows():
        food_name = row['name']
        present_nutrients = set()
        for col in nutrient_cols:
            if pd.notna(row[col]) and row[col] != 0:
                present_nutrients.add(mapper[col])
        food_to_nutrients[food_name] = present_nutrients
        
    return food_to_nutrients

master_food_list = process_food_data(food_df, nutrient_mapper)
print(f"✅ Master food list created with {len(master_food_list)} items.")


# 4. Create Disease Embeddings
if not disease_df.empty:
    disease_df.columns = disease_df.columns.str.lower().str.strip()
    predefined_diseases = disease_df['disease'].dropna().unique().tolist()
    disease_embeddings = model.encode(predefined_diseases, convert_to_tensor=True)
    print(f"✅ Embeddings created for {len(predefined_diseases)} diseases.")
else:
    predefined_diseases = []
    disease_embeddings = torch.Tensor()


# --- PHASE 2: RUNTIME LOGIC (runs on each API call) ---

def find_best_disease_match(user_input: str) -> str | None:
    if not user_input or not predefined_diseases:
        return None
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embedding, disease_embeddings)
    best_match_index = torch.argmax(cosine_scores).item()
    return predefined_diseases[best_match_index]

def get_clinical_nutrients_for_disease(disease_name: str) -> set:
    match = disease_df[disease_df['disease'] == disease_name]
    if not match.empty:
        nutrients_str = match.iloc[0]['recommended_nutrients']
        cleaned_nutrients_str = re.sub(r'\s*\([^)]*\)', '', nutrients_str).lower()
        return {nutrient.strip() for nutrient in cleaned_nutrients_str.split(',')}
    return set()

# --- NEW: Intelligent Nutrient Mapping Function ---
def map_clinical_to_scientific_nutrients(clinical_nutrients: set, top_k: int = 3) -> set:
    """
    Maps clinical nutrient terms to scientific names using semantic search.
    """
    if not clinical_nutrients or not target_nutrient_corpus:
        return set()

    mapped_nutrients = set()
    # Create embeddings for the incoming clinical terms
    clinical_embeddings = model.encode(list(clinical_nutrients), convert_to_tensor=True)

    # Compute cosine similarities between clinical and scientific nutrient names
    cosine_scores = util.pytorch_cos_sim(clinical_embeddings, nutrient_embeddings)

    # For each clinical nutrient, find the top k best matches from the scientific corpus
    for i in range(len(clinical_nutrients)):
        # Get the top_k results for the i-th clinical nutrient
        top_results = torch.topk(cosine_scores[i], k=top_k)
        
        for score, idx in zip(top_results.values, top_results.indices):
            # We can set a similarity threshold to ensure quality matches
            if score > 0.5: # Confidence threshold
                mapped_nutrients.add(target_nutrient_corpus[idx])
                
    return mapped_nutrients

def filter_foods_by_nutrients(required_nutrients: set, match_threshold: float = 0.4) -> list:
    """
    Filters foods based on the scientifically mapped nutrient list.
    """
    if not required_nutrients:
        return []

    recommended_foods = []
    for food, food_nutrients in master_food_list.items():
        common_nutrients = required_nutrients.intersection(food_nutrients)
        
        match_score = len(common_nutrients) / len(required_nutrients)
        
        if match_score >= match_threshold:
            recommended_foods.append(food)
            
    return recommended_foods

async def generate_final_plan_with_gemini(profile: UserProfile, disease: str, nutrients: set, foods: list) -> str:
    # This function remains the same, it just receives better data now.
    prompt = f"""
    Act as an expert nutritionist and chef. Your task is to create a helpful, personalized, and safe dietary recommendation plan.

    **Client Profile:**
    - Age: {profile.age}
    - Gender: {profile.gender}
    - Health Condition: {disease}
    - Reported Symptoms: {', '.join(profile.symptoms)}

    **Nutritional Goal:**
    The primary goal is to recommend foods that support the management of {disease}. Based on medical data, the key nutrients to focus on are:
    - {', '.join(sorted(list(nutrients)))}

    **Recommended Food Items:**
    Based on a nutritional database analysis, the following foods are highly recommended as they contain a significant number of the required nutrients:
    - {', '.join(sorted(foods))}

    **Your Task:**
    Generate a simple, actionable, and encouraging 1-day sample meal plan (Breakfast, Lunch, Dinner).
    1.  For each meal, suggest a simple recipe name using ONLY the recommended food items listed above.
    2.  Provide a brief, one-sentence explanation for why the meal is beneficial, mentioning one or two key nutrients it provides.
    3.  IMPORTANT: Do not suggest any food item that is NOT in the 'Recommended Food Items' list.
    4.  Include a friendly introduction and a clear disclaimer at the end stating that this is not medical advice and the user should consult a doctor.

    Format the output in clean Markdown.
    """
    try:
        # --- FIXED: Updated the model name to the latest stable version ---
        model_gemini = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = await model_gemini.generate_content_async(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Error during Gemini API call: {e}")
        return "Error: Could not generate the diet plan at this time. Please try again later."


# --- Main Function Called by the API (Updated Flow) ---
async def generate_plan_logic(user_profile: UserProfile) -> str:
    print(f"\n--- New Request ---")
    print(f"User Input Disease: '{user_profile.disease}'")

    # 1. Match user input to a predefined disease
    matched_disease = find_best_disease_match(user_profile.disease)
    if not matched_disease:
        return "Could not identify a matching health condition from the provided input."
    print(f"🔍 Best Disease Match: '{matched_disease}'")

    # 2. Get the list of clinical nutrient terms for that disease
    clinical_nutrients = get_clinical_nutrients_for_disease(matched_disease)
    if not clinical_nutrients:
        return f"Found a match for '{matched_disease}', but no specific nutrient recommendations are available in the dataset."
    print(f"🌿 Clinical Nutrients Required: {', '.join(clinical_nutrients)}")

    # 3. **NEW STEP**: Intelligently map clinical terms to scientific nutrient names
    scientific_nutrients = map_clinical_to_scientific_nutrients(clinical_nutrients)
    if not scientific_nutrients:
        return f"Could not map the clinical nutrient requirements for '{matched_disease}' to specific nutrients in our food database."
    print(f"💡 Scientifically Mapped Nutrients: {', '.join(scientific_nutrients)}")

    # 4. Filter foods using the accurately mapped scientific nutrient list
    recommended_foods = filter_foods_by_nutrients(scientific_nutrients)
    if not recommended_foods:
        return f"Found nutrient requirements for '{matched_disease}', but could not find enough matching food items in the database to generate a plan."
    print(f"🍲 Recommended Foods ({len(recommended_foods)}): {', '.join(sorted(recommended_foods)[:5])}...")

    # 5. Generate the final plan with Gemini
    final_plan = await generate_final_plan_with_gemini(user_profile, matched_disease, clinical_nutrients, recommended_foods)
    print("✅ Final plan generated by Gemini.")
    
    return final_plan

