import pandas as pd
import os
import re
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
from neo4j import Driver

from ..models.schemas import UserProfile

# --- 0. Configure Gemini API ---
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please add it.")
genai.configure(api_key=GEMINI_API_KEY)


# --- PHASE 1: STARTUP LOADING (runs once) ---

def load_data_for_mapping(file_name):
    """
    Loads a CSV file *only* for the AI mapping process.
    This does NOT load the main food/RDA data, which is now in the graph.
    """
    try:
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', file_name)
        df = pd.read_csv(path)
        print(f"✅ Mapping data '{file_name}' loaded.")
        return df
    except FileNotFoundError:
        print(f"❌ Error: Mapping file '{file_name}' was not found at {path}")
        return pd.DataFrame()

def startup_load_models():
    """
    Loads all AI models and lightweight mapping data into memory.
    This is called once when the FastAPI app starts.
    """
    # 1. Load Sentence Transformer Model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ SentenceTransformer model loaded.")
    
    # 2. Load disease data *only* for semantic matching
    disease_df = load_data_for_mapping('disease_nutrients.csv')
    if not disease_df.empty:
        disease_df.columns = disease_df.columns.str.lower().str.strip()
        predefined_diseases = disease_df['disease'].dropna().unique().tolist()
        disease_embeddings = model.encode(predefined_diseases, convert_to_tensor=True)
        print(f"✅ Embeddings created for {len(predefined_diseases)} diseases.")
    else:
        predefined_diseases = []
        disease_embeddings = torch.Tensor()

    # 3. Load abbreviation data *only* for semantic matching
    abbreviations_df = load_data_for_mapping('nutrients_abbreviations.csv')
    if not abbreviations_df.empty:
        abbreviations_df.columns = [col.strip().lower() for col in abbreviations_df.columns]
        target_nutrient_corpus = abbreviations_df['name'].dropna().unique().tolist()
        nutrient_embeddings = model.encode(target_nutrient_corpus, convert_to_tensor=True)
        print(f"✅ Embeddings created for {len(target_nutrient_corpus)} target nutrients.")
    else:
        target_nutrient_corpus = []
        nutrient_embeddings = torch.Tensor()

    # Return a dictionary holding all the loaded models and mapping data
    return {
        "model": model,
        "predefined_diseases": predefined_diseases,
        "disease_embeddings": disease_embeddings,
        "target_nutrient_corpus": target_nutrient_corpus,
        "nutrient_embeddings": nutrient_embeddings
    }

# --- PHASE 2: RUNTIME LOGIC (runs on each API call) ---

# --- Bridge Functions (Text to Graph) ---

def find_best_disease_match(user_input: str, ai_models: dict) -> str | None:
    """Finds the closest disease node name from user's text."""
    if (not user_input 
        or not ai_models["predefined_diseases"] 
        or ai_models["disease_embeddings"] is None):
        return None
        
    model = ai_models["model"]
    disease_embeddings = ai_models["disease_embeddings"]
    predefined_diseases = ai_models["predefined_diseases"]
    
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embedding, disease_embeddings)
    best_match_index = torch.argmax(cosine_scores).item()
    return predefined_diseases[best_match_index]

def get_clinical_nutrients_from_graph(disease_name: str, driver: Driver) -> set:
    """Gets the required clinical nutrient names for a disease from the graph."""
    with driver.session() as session:
        result = session.run("""
            MATCH (d:Disease {name: $disease})
            MATCH (d)-[:REQUIRES]->(n:Nutrient)
            RETURN n.name AS nutrient_name
        """, disease=disease_name)
        return {record["nutrient_name"] for record in result}

def map_clinical_to_scientific_nutrients(clinical_nutrients: set, ai_models: dict) -> set:
    """
    Ensures the nutrient names from the graph (which might be clinical)
    are mapped to the standardized scientific names for the RDA keys.
    *This function is kept in case of vocabulary mismatches*
    """
    if (not clinical_nutrients 
        or not ai_models["target_nutrient_corpus"] 
        or ai_models["nutrient_embeddings"] is None):
        return set()

    model = ai_models["model"]
    nutrient_embeddings = ai_models["nutrient_embeddings"]
    target_nutrient_corpus = ai_models["target_nutrient_corpus"]

    mapped_nutrients = set()
    clinical_embeddings = model.encode(list(clinical_nutrients), convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(clinical_embeddings, nutrient_embeddings)

    for i in range(len(clinical_nutrients)):
        best_match_index = torch.argmax(cosine_scores[i]).item()
        mapped_nutrients.add(target_nutrient_corpus[best_match_index])
                
    return mapped_nutrients

# --- NEW: Graph-Native Logic ---

def get_rda_key(age: int, gender: str) -> str:
    """
    Converts user age/gender into the specific property key
    from our Neo4j graph.
    """
    gender_key = gender.lower()
    age_key = ""
    
    if age <= 13: age_key = "9_13"
    elif 14 <= age <= 18: age_key = "14_18"
    elif 19 <= age <= 30: age_key = "19_30"
    elif 31 <= age <= 50: age_key = "31_50"
    elif 51 <= age <= 70: age_key = "51_70"
    elif age > 70: age_key = "gt_70"
    else: age_key = "19_30" # Default fallback
    
    # Handles "both" gender case from the RDA table
    if gender_key not in ["male", "female"]:
        gender_key = "both"

    # Property keys were created like: rda_female_19_30_mg
    return f"rda_{gender_key}_{age_key}_mg"

def rank_foods_by_rda_contribution(
    driver: Driver, 
    scientific_nutrients: set, 
    user_rda_key: str
) -> list:
    """
    THE NEW CORE: Ranks foods using a Cypher query.
    This replaces all the old Pandas logic.
    """
    with driver.session() as session:
        # Note: The $user_rda_key is safely injected into the query string.
        # This is generally safe as we control the input, but for production,
        # parameterizing this would be even better if the driver supported it.
        # We sanitize the key in get_rda_key() to prevent injection.
        query = f"""
        // 1. Start with the list of required nutrients
        UNWIND $nutrient_names AS nutrient_name
        MATCH (n:Nutrient {{name: nutrient_name}})

        // 2. Find foods that contain those nutrients
        MATCH (f:Food)-[c:CONTAINS_NUTRIENT]->(n)
        
        // 3. Get the user's specific RDA property from the nutrient node
        // We use apoc.when or CASE to handle missing RDA values gracefully
        WITH f, n, c.amount_mg AS food_provides, n.`{user_rda_key}` AS user_needs

        // 4. Calculate %DV score (only for valid RDAs)
        // and avoid division by zero
        WITH f, 
             CASE 
               WHEN user_needs IS NOT NULL AND user_needs > 0 
               THEN (food_provides / user_needs) * 100 
               ELSE 0 
             END AS percent_dv

        // 5. Sum the scores for each food
        WITH f, sum(percent_dv) AS total_score

        // 6. Return the top 30 ranked foods
        RETURN f.name AS food_name
        ORDER BY total_score DESC
        LIMIT 30
        """
        
        result = session.run(query, nutrient_names=list(scientific_nutrients))
        return [record["food_name"] for record in result]


async def generate_final_plan_with_gemini(profile: UserProfile, disease: str, nutrients: set, foods: list) -> str:
    """
    Generates a culturally relevant, Indian-style, nutrient-focused 1-day meal plan.
    Uses Gemini (gemini-2.0-flash-exp) with improved reasoning and recipe realism.
    """

    prompt = f"""
    You are an expert nutritionist and chef specializing in culturally appropriate, evidence-based Indian meal planning.
    Your task is to create a helpful, personalized, and realistic dietary recommendation plan.

    Before answering, internally (without showing your reasoning), follow this checklist:
    1. Every *main ingredient* in each recipe must come from the 'Top Recommended Food Items' list below.
    2. You may use only common Indian pantry staples: salt, water, neutral cooking oil, turmeric, cumin, coriander, chili powder, garam masala, mustard seeds, fresh cilantro, ginger, and garlic.
    3. Prefer existing Indian dishes (e.g., dal, khichdi, upma, sabzi, chilla, pulao). If necessary, adapt them using only allowed ingredients.
    4. Ensure all recipes are realistic and cookable — include measurable ingredient quantities, short cooking steps, cook time, and servings.
    5. Mention 1-2 nutrients (from the nutrient focus list) that make each meal beneficial for managing {disease}.
    6. Keep the tone friendly, encouraging, and culturally familiar.

    ---

    **Client Profile:**
    - Age: {profile.age}
    - Gender: {profile.gender}
    - Health Condition: {disease}
    - Reported Symptoms: {', '.join(profile.symptoms)}

    **Nutritional Goal:**
    Recommend foods that support the management of {disease}.
    Focus on the following nutrients:
    - {', '.join(sorted(list(nutrients)))}

    **Top Recommended Food Items (ONLY these may be used as main ingredients):**
    - {', '.join(sorted(foods))}

    **Allowed Pantry Items (small amounts only):**
    salt, water, neutral cooking oil, turmeric, cumin, coriander, chili powder, garam masala, mustard seeds, fresh cilantro, ginger, garlic

    ---

    **Your Task:**
    Generate a simple, actionable, and encouraging 1-day meal plan (Breakfast, Lunch, Dinner).

    For each meal:
    1. Use ONLY foods from the 'Top Recommended Food Items' list (plus small pantry items if needed).
    2. Give a *realistic Indian recipe name* (or "Adapted -" if modified).
    3. List measurable ingredient quantities and approximate cook time.
    4. Provide clear, short cooking steps (max 8 steps).
    5. Add a one-sentence explanation of why the meal helps manage {disease}, mentioning relevant nutrients.
    6. Suggest substitution options from the same food list (if any).
    
    After all meals, include:
    - A short "Notes & Tips" section (3 practical tips using only allowed foods).
    - A clear disclaimer: this is not medical advice and the user should consult a doctor.

    **Formatting Requirements:**
    - Use clean Markdown.
    - Use clear headings for each meal: ### Breakfast, ### Lunch, ### Dinner
    - Each recipe must be plausible, safe, and Indian in style.
    - Avoid introducing any food not listed above.

    Now create the full meal plan.
    """

    try:
        model_gemini = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = await model_gemini.generate_content_async(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Error during Gemini API call: {e}")
        return "Error: Could not generate the diet plan at this time. Please try again later."


# --- Main Function Called by the API (Updated Flow) ---
async def generate_plan_logic(
    user_profile: UserProfile, 
    neo4j_driver: Driver, 
    ai_models: dict
) -> str:
    
    print(f"\n--- New GraphRAG Request ---")
    print(f"User Input Disease: '{user_profile.disease}'")

    # 1. Match disease (Text -> Graph Node)
    matched_disease = find_best_disease_match(user_profile.disease, ai_models)
    if not matched_disease: return "Could not identify a matching health condition."
    print(f"🔍 Best Disease Match: '{matched_disease}'")

    # 2. Get clinical nutrients (Graph Query)
    clinical_nutrients = get_clinical_nutrients_from_graph(matched_disease, neo4j_driver)
    if not clinical_nutrients: return f"No specific nutrient recommendations found for '{matched_disease}'."
    print(f"🌿 Clinical Nutrients Required: {', '.join(clinical_nutrients)}")

    # 3. Map to scientific nutrients (Text -> Text Mapping)
    # This step is a "failsafe" to ensure our nutrient names are standardized
    scientific_nutrients = map_clinical_to_scientific_nutrients(clinical_nutrients, ai_models)
    if not scientific_nutrients:
        print("⚠️ Warning: Could not map nutrients, using raw clinical names.")
        scientific_nutrients = clinical_nutrients
    print(f"💡 Scientifically Mapped Nutrients: {', '.join(scientific_nutrients)}")

    # 4. Get the user's personal RDA property key (Python Logic)
    user_rda_key = get_rda_key(user_profile.age, user_profile.gender)
    print(f"🔬 User RDA Key: '{user_rda_key}'")

    # 5. **NEW CORE:** Rank foods based on %DV contribution (Graph Query)
    recommended_foods = rank_foods_by_rda_contribution(neo4j_driver, scientific_nutrients, user_rda_key)
    if not recommended_foods: return f"Found nutrient requirements for '{matched_disease}', but could not find potent food sources in the database."
    print(f"🍲 Top Recommended Foods ({len(recommended_foods)}): {', '.join(sorted(recommended_foods)[:5])}...")

    # 6. Generate final plan (LLM Call)
    final_plan = await generate_final_plan_with_gemini(user_profile, matched_disease, clinical_nutrients, recommended_foods)
    print("✅ Final plan generated by Gemini.")
    
    return final_plan

