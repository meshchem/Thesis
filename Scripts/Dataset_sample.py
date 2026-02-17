"""
QUICK START SCRIPT - Stratified Sampling
Use this if you just want to run it quickly!
"""

from datasets import load_dataset
import pandas as pd
import random

# CONFIGURATION - CHANGE THESE
SAMPLE_SIZE = 300
STRATEGY = "post_type_engagement"  # or "hashtag_time" or "user_influence_time"

print("Loading dataset (streaming - no download)...")
dataset = load_dataset(
    "deadbirds/usc-x-24-us-election-parquet",
    split='train',
    streaming=True
)

# Simple stratification function
def get_stratum(row):
    """Classify tweet into post_type × engagement stratum"""
    
    # Post type
    if row.get('retweetedTweet', False):
        post_type = 'retweet'
    elif row.get('quotedTweet', False):
        post_type = 'quote'
    elif pd.notna(row.get('in_reply_to_status_id_str')):
        post_type = 'reply'
    else:
        post_type = 'original'
    
    # Engagement
    likes = row.get('likeCount', 0) or 0
    if likes >= 100:
        engagement = 'high'
    elif likes >= 10:
        engagement = 'medium'
    else:
        engagement = 'low'
    
    return f"{post_type}_{engagement}"

# Collect tweets by stratum
from collections import defaultdict
strata = defaultdict(list)

print("Streaming through tweets...")
for i, row in enumerate(dataset):
    if i >= 50000:  # Process first 50k tweets
        break
    
    if i % 5000 == 0:
        print(f"  Processed {i} tweets...")
    
    try:
        stratum = get_stratum(row)
        if len(strata[stratum]) < 100:  # Keep max 100 per stratum
            strata[stratum].append(row)
    except:
        continue

print(f"\nFound {len(strata)} strata")

# Sample equally from each stratum
samples_per_stratum = SAMPLE_SIZE // len(strata)
sampled = []

for stratum, tweets in strata.items():
    n = min(samples_per_stratum, len(tweets))
    sample = random.sample(tweets, n)
    for tweet in sample:
        tweet['stratum'] = stratum
    sampled.extend(sample)
    print(f"  {stratum}: sampled {n} tweets")

# Save
df = pd.DataFrame(sampled)
df.to_csv('quick_sample.csv', index=False)

print(f"\n✅ Saved {len(sampled)} tweets to quick_sample.csv")
print("\nSample distribution:")
print(df['stratum'].value_counts())