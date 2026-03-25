import pandas as pd

# Load parquet
df = pd.read_parquet('data_aug/tweets_keyword_engagement_2000.parquet')

# # View first few rows
# print(df.head())

# # View all columns
# print(df.columns)

# # View specific columns
# print(df[['tweet_id', 'text', 'like_count']].head())

# # Get basic info
# print(df.info())

# # View one tweet in detail
# print(df.iloc[0])

# Save as CSV


df.to_csv('../manual_annotations/sample_2000.csv', index=False)

# Or save just a subset
# df[['row_id', 'post_type', 'text', 'keyword_category',  'hashtags',  'like_count', 'date']].to_csv('../LLM_annotations/sampled_tweets_2000.csv', index=False)