import pandas as pd

df = pd.read_csv("../manual_annotations/manual_annotations_300.csv")
sampled_df = df.sample(n=50, random_state=42)
sampled_df.to_csv("../manual_annotations/sampled_50_rows.csv", index=False)
