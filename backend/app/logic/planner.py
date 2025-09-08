import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai

from ..models.schemas import UserProfile

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
    target_nutrient_corpus = abbreviations_df['name'].dropna().unique().tolist()
    nutrient_embeddings = model.encode(target_nutrient_corpus, convert_to_tensor=True)
    print(f"✅ Embeddings created for {len(target_nutrient_corpus)} target nutrients.")
else:
    target_nutrient_corpus = []
    nutrient_embeddings = torch.Tensor()

# 3. Process Food Nutrient Data to store VALUES
def process_food_data_with_values(df, mapper):
    if df.empty or not mapper:
        return {}
    
    df.columns = df.columns.str.lower()
    food_to_nutrients_values = {}
    
    for _, row in df.iterrows():
        food_name = row['name']
        nutrients_with_values = {}
        for code, name in mapper.items():
            if code in df.columns and pd.notna(row[code]) and row[code] != 0:
                nutrients_with_values[name] = row[code]
        food_to_nutrients_values[food_name] = nutrients_with_values
        
    return food_to_nutrients_values

master_food_list_values = process_food_data_with_values(food_df, nutrient_mapper)
print(f"✅ Master food list with nutrient VALUES created for {len(master_food_list_values)} items.")

# 4. **UPDATED:** Calculate average for ONLY non-zero nutrient values
def calculate_nutrient_averages(df, mapper):
    if df.empty or not mapper:
        return {}
    
    df.columns = df.columns.str.lower()
    averages = {}
    for code, name in mapper.items():
        if code in df.columns:
            numeric_col = pd.to_numeric(df[code], errors='coerce')
            # Filter out NaNs and zeros before calculating the mean
            non_zero_values = numeric_col[numeric_col > 0]
            if not non_zero_values.empty:
                averages[name] = non_zero_values.mean()
    return averages

nutrient_averages = calculate_nutrient_averages(food_df, nutrient_mapper)
print(f"✅ Averages calculated for {len(nutrient_averages)} nutrients (from non-zero values only).")


# 5. Create Disease Embeddings
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
    # ... (This function remains the same)
    if not user_input or not predefined_diseases:
        return None
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embedding, disease_embeddings)
    best_match_index = torch.argmax(cosine_scores).item()
    return predefined_diseases[best_match_index]

def get_clinical_nutrients_for_disease(disease_name: str) -> set:
    # ... (This function remains the same)
    match = disease_df[disease_df['disease'] == disease_name]
    if not match.empty:
        nutrients_str = match.iloc[0]['recommended_nutrients']
        cleaned_nutrients_str = re.sub(r'\s*\([^)]*\)', '', nutrients_str).lower()
        return {nutrient.strip() for nutrient in cleaned_nutrients_str.split(',')}
    return set()

def map_clinical_to_scientific_nutrients(clinical_nutrients: set, top_k: int = 1) -> set:
    # ... (This function remains the same)
    if not clinical_nutrients or not target_nutrient_corpus:
        return set()

    mapped_nutrients = set()
    clinical_embeddings = model.encode(list(clinical_nutrients), convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(clinical_embeddings, nutrient_embeddings)

    for i in range(len(clinical_nutrients)):
        best_match_index = torch.argmax(cosine_scores[i]).item()
        mapped_nutrients.add(target_nutrient_corpus[best_match_index])
                
    return mapped_nutrients

# --- Food Ranking Logic based on Average Nutrient Values ---
def rank_foods_by_average(required_nutrients: set) -> list:
    """
    Ranks foods based on how many required nutrients they contain in "above average" amounts.
    """
    food_scores = {}
    
    for food, nutrients in master_food_list_values.items():
        score = 0
        for req_nutrient in required_nutrients:
            if req_nutrient in nutrients:
                average_amount = nutrient_averages.get(req_nutrient)
                # Ensure we have a valid average to compare against
                if average_amount is not None:
                    current_amount = nutrients[req_nutrient]
                    if current_amount > average_amount:
                        score += 1 # This food is an "above average" source for one required nutrient
        if score > 0:
            food_scores[food] = score
            
    # Sort foods by their score in descending order
    sorted_foods = sorted(food_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Return the names of the top 30 most potent foods
    return [food for food, score in sorted_foods[:30]]


async def generate_final_plan_with_gemini(profile: UserProfile, disease: str, nutrients: set, foods: list) -> str:
    # ... (This function remains the same, but now gets a higher quality food list)
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

    **Top Recommended Food Items:**
    Based on a nutritional analysis, the following foods are the most potent sources for the required nutrients (containing them in above-average amounts):
    - {', '.join(sorted(foods))}

    **Your Task:**
    Generate a simple, actionable, and encouraging 1-day sample meal plan (Breakfast, Lunch, Dinner).
    1.  For each meal, suggest a simple recipe name using ONLY the recommended food items listed above.
    2.  Provide a brief, one-sentence explanation for why the meal is beneficial, mentioning one or two key nutrients it provides.
    3.  IMPORTANT: Do not suggest any food item that is NOT in the 'Top Recommended Food Items' list.
    4.  Include a friendly introduction and a clear disclaimer at the end stating that this is not medical advice and the user should consult a doctor.

    Format the output in clean Markdown.
    """
    try:
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

    # 1. Match disease
    matched_disease = find_best_disease_match(user_profile.disease)
    if not matched_disease: return "Could not identify a matching health condition."
    print(f"🔍 Best Disease Match: '{matched_disease}'")

    # 2. Get clinical nutrients
    clinical_nutrients = get_clinical_nutrients_for_disease(matched_disease)
    if not clinical_nutrients: return f"No specific nutrient recommendations found for '{matched_disease}'."
    print(f"🌿 Clinical Nutrients Required: {', '.join(clinical_nutrients)}")

    # 3. Map to scientific nutrients
    scientific_nutrients = map_clinical_to_scientific_nutrients(clinical_nutrients)
    if not scientific_nutrients: return "Could not map clinical needs to specific nutrients."
    print(f"💡 Scientifically Mapped Nutrients: {', '.join(scientific_nutrients)}")

    # 4. **UPDATED:** Rank foods based on "above average" nutrient content
    recommended_foods = rank_foods_by_average(scientific_nutrients)
    if not recommended_foods: return f"Found nutrient requirements for '{matched_disease}', but could not find potent food sources in the database."
    print(f"🍲 Top Recommended Foods ({len(recommended_foods)}): {', '.join(sorted(recommended_foods)[:])}...")

    # 5. Generate final plan
    final_plan = await generate_final_plan_with_gemini(user_profile, matched_disease, clinical_nutrients, recommended_foods)
    print("✅ Final plan generated by Gemini.")
    
    return final_plan

