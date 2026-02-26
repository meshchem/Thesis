# import pandas as pd

# # Load the parquet file
# df = pd.read_parquet('data/tweets.parquet')

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

import pandas as pd

# Load parquet
df = pd.read_parquet('data_aug/tweets_keyword_engagement_1.parquet')

# Save as CSV
df.to_csv('tweets.csv', index=False)

# Or save just a subset
# df[['row_id', 'keyword_category', 'text', 'hashtags', 'post_type', 'like_count', 'date']].to_csv('tweets_aug_reduced.csv', index=False)