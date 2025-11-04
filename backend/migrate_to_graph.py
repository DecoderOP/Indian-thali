import pandas as pd
import os
import re
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch

# --- CONFIGURATION ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Thali@1938" # <-- Make sure this is your correct password
load_dotenv()
# --- END CONFIGURATION ---

def load_data(file_name):
    """Loads a CSV file from the data directory."""
    try:
        path = os.path.join(os.path.dirname(__file__), 'data', file_name)
        df = pd.read_csv(path)
        print(f"✅ Dataset '{file_name}' loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"❌ Error: The file '{file_name}' was not found at {path}")
        return pd.DataFrame()

def get_nutrient_mapper(abbreviations_df):
    """Creates the nutrient name standardization mapper."""
    if abbreviations_df.empty:
        return {}
    
    abbreviations_df.columns = [col.strip().lower() for col in abbreviations_df.columns]
    name_to_name_map = pd.Series(abbreviations_df['name'].values, index=abbreviations_df['name'].str.lower()).to_dict()
    code_to_name_map = pd.Series(abbreviations_df['name'].values, index=abbreviations_df['code']).to_dict()
    nutrient_mapper = {**{k.lower(): v for k, v in name_to_name_map.items()}, 
                       **{k.lower(): v for k, v in code_to_name_map.items()}}
    print("✅ Nutrient abbreviation mapper created.")
    return nutrient_mapper

def create_constraints(driver):
    """Creates unique constraints in the graph."""
    with driver.session() as session:
        print("Applying graph constraints (FOR...REQUIRE)...")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Nutrient) REQUIRE n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:Food) REQUIRE f.name IS UNIQUE")
        print("✅ Graph constraints created (or already exist).")

def migrate_nutrients(driver, mapper, rda_df):
    """Loads all nutrients and merges them with RDA data."""
    if rda_df.empty:
        print("❌ RDA_DF is empty. Cannot migrate nutrients.")
        return
    
    print("Migrating Nutrients...")
    rda_df.columns = [col.strip().lower() for col in rda_df.columns]
    rda_df['nutrient'] = rda_df['nutrient'].str.lower().map(mapper).fillna(rda_df['nutrient'])
    all_nutrient_names = set(mapper.values()) | set(rda_df['nutrient'].unique())
    
    with driver.session() as session:
        for name in all_nutrient_names:
            if not pd.isna(name):
                session.run("MERGE (n:Nutrient {name: $name})", name=name)
        
        for _, row in rda_df.iterrows():
            gender_key = str(row['gender']).lower()
            age_key = str(row['age_group']).replace(' ', '_').replace('>', 'gt').replace('-', '_')
            unit_key = str(row['unit']).lower()
            
            rda_val = row['rda']
            if isinstance(rda_val, str):
                rda_val = float(rda_val.replace(',', ''))
            
            standard_rda_mg = 0
            if unit_key == 'g': standard_rda_mg = rda_val * 1000
            elif unit_key in ['µg', 'mcg']: standard_rda_mg = rda_val / 1000
            elif unit_key == 'mg': standard_rda_mg = rda_val
            elif unit_key == 'l': standard_rda_mg = rda_val * 1000000
            
            prop_key = f"rda_{gender_key}_{age_key}_mg"
            
            session.run(f"""
                MATCH (n:Nutrient {{name: $nutrient_name}})
                SET n.`{prop_key}` = $rda_value
            """, nutrient_name=row['nutrient'], rda_value=standard_rda_mg)
            
    print(f"✅ Migrated {len(all_nutrient_names)} nutrients and added RDA properties.")

# --- NEW: AI Mapping Function for the Migration ---
def map_clinical_to_scientific(clinical_nutrients, model, corpus, embeddings):
    """Uses AI to map clinical terms to scientific terms."""
    if not clinical_nutrients:
        return set()

    mapped_nutrients = set()
    clinical_embeddings = model.encode(list(clinical_nutrients), convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(clinical_embeddings, embeddings)

    for i in range(len(clinical_nutrients)):
        best_match_index = torch.argmax(cosine_scores[i]).item()
        mapped_nutrients.add(corpus[best_match_index])
                
    return mapped_nutrients

# --- MODIFIED: This function is now "smart" ---
def migrate_diseases(driver, disease_df, nutrient_mapper, ai_models):
    """
    Loads diseases and uses AI to link them to the correct
    scientific nutrient nodes.
    """
    if disease_df.empty:
        return
        
    print("Migrating Diseases (Smart Mode)...")
    disease_df.columns = [col.strip().lower() for col in disease_df.columns]

    with driver.session() as session:
        for _, row in disease_df.iterrows():
            disease_name = row['disease']
            if pd.isna(disease_name):
                continue
            
            session.run("MERGE (d:Disease {name: $name})", name=disease_name)
            
            nutrients_str = row['recommended_nutrients']
            if pd.isna(nutrients_str):
                continue
                
            cleaned_nutrients_str = re.sub(r'\s*\([^)]*\)', '', nutrients_str).lower()
            clinical_nutrients = {nutrient.strip() for nutrient in cleaned_nutrients_str.split(',') if nutrient.strip()}
            
            # --- This is the new "smart" step ---
            scientific_nutrients = map_clinical_to_scientific(
                clinical_nutrients,
                ai_models["model"],
                ai_models["target_nutrient_corpus"],
                ai_models["nutrient_embeddings"]
            )
            # --- End of smart step ---
            
            # Create relationships using the *correct* scientific names
            for nutrient_name in scientific_nutrients:
                session.run("""
                    MATCH (d:Disease {name: $disease})
                    MATCH (n:Nutrient {name: $nutrient_name})
                    MERGE (d)-[:REQUIRES]->(n)
                """, disease=disease_name, nutrient_name=nutrient_name)
                
    print(f"✅ Migrated {len(disease_df)} disease entries (Smart Relationships).")

def migrate_foods(driver, food_df, mapper):
    """
    Loads foods and links them to the nutrients they contain,
    storing the amount on the relationship.
    """
    if food_df.empty:
        return
        
    print("Migrating Foods...")
    food_df.columns = [col.strip().lower() for col in food_df.columns]
    
    with driver.session() as session:
        nutrient_codes_in_db = [col for col in food_df.columns if col in mapper and not col.endswith('_e')]
        
        for _, row in food_df.iterrows():
            food_name = row['name']
            if pd.isna(food_name):
                continue
            
            session.run("MERGE (f:Food {name: $name})", name=food_name)
            
            for code in nutrient_codes_in_db:
                value = row[code]
                if pd.notna(value):
                    try:
                        amount_mg = float(str(value).replace(',', ''))
                        if amount_mg == 0:
                            continue
                    except ValueError:
                        continue
                        
                    nutrient_name = mapper.get(code)
                    if not nutrient_name:
                        continue
                    
                    session.run("""
                        MATCH (f:Food {name: $food})
                        MATCH (n:Nutrient {name: $nutrient})
                        MERGE (f)-[r:CONTAINS_NUTRIENT]->(n)
                        SET r.amount_mg = $amount
                    """, food=food_name, nutrient=nutrient_name, amount=amount_mg)
                    
    print(f"✅ Migrated {len(food_df)} foods and their nutrient relationships.")


def main():
    print("--- Starting Graph Migration ETL (SMART MODE) ---")
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"✅ Successfully connected to Neo4j at {NEO4J_URI}")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        print("Please check your Neo4j credentials and that the database is running.")
        return

    # --- NEW: Load AI Models for Smart Migration ---
    print("Loading AI model for semantic mapping...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    abbreviations_df = load_data('nutrients_abbreviations.csv')
    abbreviations_df.columns = [col.strip().lower() for col in abbreviations_df.columns]
    target_nutrient_corpus = abbreviations_df['name'].dropna().unique().tolist()
    nutrient_embeddings = model.encode(target_nutrient_corpus, convert_to_tensor=True)
    
    ai_models = {
        "model": model,
        "target_nutrient_corpus": target_nutrient_corpus,
        "nutrient_embeddings": nutrient_embeddings
    }
    print("✅ AI models loaded for migration.")
    # --- End New AI Block ---

    # Load all data
    disease_df = load_data('disease_nutrients.csv')
    food_df = load_data('Food-nutrient_dataset.csv')
    rda_df = load_data('Indian_RDA.csv')
    
    # Create the nutrient mapper
    nutrient_mapper = get_nutrient_mapper(abbreviations_df)
    
    # Run migration steps
    create_constraints(driver)
    migrate_nutrients(driver, nutrient_mapper, rda_df)
    
    # --- MODIFIED: Pass AI models into disease migration ---
    migrate_diseases(driver, disease_df, nutrient_mapper, ai_models)
    
    migrate_foods(driver, food_df, nutrient_mapper)
    
    driver.close()
    
    print("--- Graph Migration ETL Complete ---")
    print("Your new, 'smart' graph is ready in Neo4j Desktop.")

if __name__ == "__main__":
    main()

