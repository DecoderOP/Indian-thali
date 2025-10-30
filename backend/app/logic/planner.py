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
# NEW: Loading the official RDA table
rda_df = load_data('Indian_RDA.csv') 

# Load Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ SentenceTransformer model loaded.")

# --- Data Cleaning and Transformation ---

# 1. Create Nutrient Abbreviation Mapper
if not abbreviations_df.empty:
    abbreviations_df.columns = [col.strip().lower() for col in abbreviations_df.columns]
    # Create a mapping from both 'code' and 'name' to the official 'name'
    # This helps standardize nutrient names from different sources
    name_to_name_map = pd.Series(abbreviations_df['name'].values, index=abbreviations_df['name'].str.lower()).to_dict()
    code_to_name_map = pd.Series(abbreviations_df['name'].values, index=abbreviations_df['code']).to_dict()
    # Combine mappers, giving preference to code mapping
    nutrient_mapper = {**{k.lower(): v for k, v in name_to_name_map.items()}, 
                       **{k.lower(): v for k, v in code_to_name_map.items()}}
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
            if code in df.columns and pd.notna(row[code]):
                try:
                    # Ensure value is numeric, converting if necessary
                    value = float(str(row[code]).replace(',', ''))
                    if value != 0:
                        # Map to the standardized name
                        nutrients_with_values[name] = value
                except ValueError:
                    continue # Skip non-numeric values
        food_to_nutrients_values[food_name] = nutrients_with_values
        
    return food_to_nutrients_values

master_food_list_values = process_food_data_with_values(food_df, nutrient_mapper)
print(f"✅ Master food list with nutrient VALUES created for {len(master_food_list_values)} items.")

# 4. Standardize RDA Table
if not rda_df.empty:
    rda_df.columns = [col.strip().lower() for col in rda_df.columns]
    # Standardize nutrient names in RDA table to match our abbreviations
    rda_df['nutrient'] = rda_df['nutrient'].str.lower().map(nutrient_mapper).fillna(rda_df['nutrient'])
    print("✅ RDA Table standardized and nutrients mapped.")
else:
    print("❌ WARNING: Indian_RDA.csv is empty or not found. Personalization will be disabled.")

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
        # Get the official name from the corpus
        official_name = target_nutrient_corpus[best_match_index]
        mapped_nutrients.add(official_name)
                
    return mapped_nutrients

# --- NEW: Gold Standard Logic (RDA Based) ---

def get_user_rda_profile(age: int, gender: str) -> dict:
    """
    Gets the personalized RDA profile for the user, standardized to milligrams (mg).
    Uses the new age groups from Indian_RDA.csv
    """
    if rda_df.empty:
        return {}

    # 1. Determine Age Group from the new RDA table
    age_group = ""
    if age <= 13: # Assuming 9-13 is the youngest in our adult-focused scope
        age_group = "9-13"
    elif 14 <= age <= 18:
        age_group = "14-18"
    elif 19 <= age <= 30:
        age_group = "19-30"
    elif 31 <= age <= 50:
        age_group = "31-50"
    elif 51 <= age <= 70:
        age_group = "51-70"
    elif age > 70:
        age_group = "> 70"
    else:
        age_group = "19-30" # Default fallback for any other case

    profile = {}
    
    # Filter by age group
    age_df = rda_df[rda_df['age_group'] == age_group]
    
    # Filter by gender (handling 'Both')
    gender_df = age_df[
        (age_df['gender'].str.lower() == gender.lower()) |
        (age_df['gender'].str.lower() == 'both')
    ]
    
    for _, row in gender_df.iterrows():
        nutrient = row['nutrient'] # Already standardized at startup
        rda = row['rda']
        
        # Handle cases where RDA might be a string with a comma (e.g., "1,100")
        if isinstance(rda, str):
            rda = float(rda.replace(',', ''))
            
        unit = str(row['unit']).lower()
        
        # Standardize all RDA values to milligrams (mg)
        standard_rda_mg = 0
        if unit == 'g':
            standard_rda_mg = rda * 1000
        elif unit == 'µg' or unit == 'mcg':
            standard_rda_mg = rda / 1000
        elif unit == 'mg':
            standard_rda_mg = rda
        elif unit == 'l': # Handle Total Water (Moisture)
            standard_rda_mg = rda * 1000000 # 1 L = 1 kg = 1,000,000 mg
        
        if standard_rda_mg > 0:
            profile[nutrient] = standard_rda_mg
            
    return profile

def rank_foods_by_rda_contribution(user_rda_profile: dict, required_nutrients: set) -> list:
    """
    Ranks foods based on the sum of their % Daily Value (%DV) contribution
    for all required nutrients.
    """
    if not user_rda_profile or not required_nutrients:
        return []

    food_scores = {}
    
    for food, nutrients_in_food in master_food_list_values.items():
        total_score = 0
        for req_nutrient in required_nutrients:
            # Check if the user has an RDA for this nutrient
            # AND the food actually contains this nutrient
            if req_nutrient in user_rda_profile and req_nutrient in nutrients_in_food:
                
                user_need_mg = user_rda_profile[req_nutrient]
                food_provides_mg = nutrients_in_food[req_nutrient]
                
                if user_need_mg > 0:
                    # Calculate %DV contribution
                    percent_dv = (food_provides_mg / user_need_mg) * 100
                    total_score += percent_dv # Add the %DV to the total score
                    
        if total_score > 0:
            food_scores[food] = total_score
            
    # Sort foods by their total contribution score, descending
    sorted_foods = sorted(food_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Return the names of the top 30 most potent foods
    return [food for food, score in sorted_foods[:30]]


async def generate_final_plan_with_gemini(profile: UserProfile, disease: str, nutrients: set, foods: list) -> str:
    # ... (Prompt is updated to reflect the new logic)
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
    Based on a nutritional analysis of your specific age and gender, the following foods are the most potent sources, contributing the highest percentage of your daily needs for the required nutrients (per 100g serving):
    - {', '.join(sorted(foods))}

    **Your Task:**
    Generate a simple, actionable, and encouraging 1-day sample meal plan (Breakfast, Lunch, Dinner).
    1.  For each meal, suggest a simple recipe name using ONLY the recommended food items listed above and by exploring the Indian Food recipies data available.
    2.  Provide a brief, one-sentence explanation for why the meal is beneficial, mentioning one or two key nutrients it provides and after that , give steps to prepare the recipie also.
    3.  IMPORTANT: Do not suggest any food item that is NOT in the 'Top Recommended Food Items' list.
    4.  Include a friendly introduction and a clear disclaimer at the end stating that this is not medical advice and the user should consult a doctor.

    Format the output in clean Markdown.
    """
    try:
        # --- FIXED: Reverting to 'gemini-pro' for better compatibility with older v1beta API endpoints ---
        model_gemini = genai.GenerativeModel('gemini-2.0-flash')
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
    matched_disease = find_best_disease_match(user_input=user_profile.disease)
    if not matched_disease: return "Could not identify a matching health condition."
    print(f"🔍 Best Disease Match: '{matched_disease}'")

    # 2. Get clinical nutrients
    clinical_nutrients = get_clinical_nutrients_for_disease(disease_name=matched_disease)
    if not clinical_nutrients: return f"No specific nutrient recommendations found for '{matched_disease}'."
    print(f"🌿 Clinical Nutrients Required: {', '.join(clinical_nutrients)}")

    # 3. Map to scientific nutrients
    scientific_nutrients = map_clinical_to_scientific_nutrients(clinical_nutrients=clinical_nutrients)
    if not scientific_nutrients: return "Could not map clinical needs to specific nutrients."
    print(f"💡 Scientifically Mapped Nutrients: {', '.join(scientific_nutrients)}")

    # 4. **NEW:** Get the user's personal RDA profile
    user_rda_profile = get_user_rda_profile(age=user_profile.age, gender=user_profile.gender)
    if not user_rda_profile:
        return "Could not determine a personalized nutritional profile for the specified age or gender."
    print(f"🔬 User RDA Profile (sample): 'Protein': {user_rda_profile.get('Protein')} mg")

    # 5. **NEW:** Rank foods based on %DV contribution
    recommended_foods = rank_foods_by_rda_contribution(user_rda_profile=user_rda_profile, required_nutrients=scientific_nutrients)
    if not recommended_foods: return f"Found nutrient requirements for '{matched_disease}', but could not find potent food sources in the database."
    print(f"🍲 Top Recommended Foods ({len(recommended_foods)}): {', '.join(sorted(recommended_foods)[:5])}...")

    # 6. Generate final plan
    final_plan = await generate_final_plan_with_gemini(profile=user_profile, disease=matched_disease, nutrients=clinical_nutrients, foods=recommended_foods)
    print("✅ Final plan generated by Gemini.")
    
    return final_plan

