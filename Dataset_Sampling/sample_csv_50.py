import pandas as pd

df = pd.read_csv("../manual_annotations/sampled_tweets_aug_300.csv")
sampled_df = df.sample(n=50, random_state=42)
sampled_df.to_csv("../manual_annotations/subset_50.csv", index=False)
