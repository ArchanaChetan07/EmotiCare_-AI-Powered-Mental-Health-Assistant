import pandas as pd
from pathlib import Path

def load_gold_data():
    base_path = Path(__file__).resolve().parents[3] / "Data_final"
    
    data = {
        "counselchat": pd.read_csv(base_path / "counselchat_gold.csv"),
        "facebook": pd.read_csv(base_path / "facebook_gold.csv"),
        "goemotions": pd.read_csv(base_path / "goemotions_gold.csv")
    }
    return data
