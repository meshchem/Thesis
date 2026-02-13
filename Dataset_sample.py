from datasets import load_dataset
import pandas as pd
import numpy as np

# Load the dataset (~22M rows, but loads efficiently as parquet)
ds = load_dataset("deadbirds/usc-x-24-us-election-parquet", split="train")
df = ds.to_pandas()

print(f"Loaded {len(df):,} tweets")
print("Sample columns:", df.columns.tolist()[:10])  # Check schema
