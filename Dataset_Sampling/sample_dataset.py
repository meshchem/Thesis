# Engagement-stratified tweet sampler — August 2024
# Author: Maria Meshcheryakova, 2025

import pandas as pd
from datasets import load_dataset
from datetime import datetime
import os


# CONFIGURATION

SAMPLE_SIZE = 2000
# SAMPLE_SIZE = 500
RANDOM_SEED = 45
OUTPUT_PATH = "sampled_data/tweets_engagement_2000.parquet"
# OUTPUT_PATH = "data_aug/tweets_engagement_2000.parquet"


# Dataset info
DATASET_REPO = "deadbirds/usc-x-24-us-election-parquet"

# August 2024 parts to load (pre-election period)
AUGUST_PARTS = [23, 24, 25, 26, 27, 28]

# Number of tweets to collect before sampling
MAX_COLLECT = 1800000

LANGUAGE = "en"

# Date range filter (August 2024)
DATE_START = "2024-08-01"
DATE_END = "2024-08-31"

# Engagement tiers
# Based on August dataset distribution (98.6% have < 100 likes)
ENGAGEMENT_TIERS = {
    'low':    (0, 50),      # 0-50 likes
    'medium': (51, 500),    # 51-500 likes
    'high':   (501, None),  # 500+ likes
}

# Proportion of each engagement tier in final sample
ENGAGEMENT_PROPORTIONS = {
    'low':    0.50,   # 50% (reflects majority of organic discourse)
    'medium': 0.30,   # 30%
    'high':   0.20,   # 20% (oversampled relative to pool for diversity)
}


# HELPER FUNCTIONS

def derive_post_type(row):
    # Returns 'retweet', 'quote', 'reply', or 'tweet'
    if row.get('retweetedTweet') is True:
        return 'retweet'
    elif row.get('quotedTweet') is True:
        return 'quote'
    elif row.get('in_reply_to_screen_name') is not None:
        return 'reply'
    return 'tweet'


def classify_engagement(like_count):
    # Returns 'low' (<=50), 'medium' (<=500), or 'high' (500+)
    if like_count <= 50:
        return 'low'
    elif like_count <= 500:
        return 'medium'
    else:
        return 'high'


# LOADING DATA FROM AUGUST PARTS (PARTS 23-28)

def load_august_tweets():
    # Streams August 2024 tweets from 6 evenly spaced chunks per part (36 files total)
    print("Streaming English tweets from August 2024:")
    print()
    print(f"Target parts: {AUGUST_PARTS}")
    print(f"Loading 6 evenly spaced chunks per part (36 files total)")
    print(f"Date filter: {DATE_START} to {DATE_END} (August 2024)")
    print()

    CHUNKS_PER_PART = {
        23: [1, 4, 8, 12, 16, 20],
        24: [21, 24, 28, 32, 36, 40],
        25: [41, 44, 48, 52, 56, 60],
        26: [61, 64, 68, 72, 76, 80],
        27: [81, 84, 88, 92, 96, 100],
        28: [101, 104, 107, 110, 113, 117],
    }

    data_files = []
    for part_num, chunks in CHUNKS_PER_PART.items():
        for chunk_num in chunks:
            file_path = f"part_{part_num}/aug_chunk_{chunk_num}.parquet"
            data_files.append(file_path)

    print(f"Files to load ({len(data_files)} total):")
    for part_num, chunks in CHUNKS_PER_PART.items():
        print(f"  part_{part_num}: chunks {chunks}")
    print()

    print("Loading dataset in streaming mode:")
    print("(Only downloads the specified chunks)")
    print()

    try:
        dataset = load_dataset(
            DATASET_REPO,
            data_files=data_files,
            split="train",
            streaming=True
        )
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return []

    print("Dataset loaded, streaming tweets...")
    print()

    collected = []
    total_processed = 0
    skipped_language = 0
    skipped_date = 0

    try:
        for row in dataset:
            total_processed += 1

            if total_processed % 20000 == 0:
                print(f"   Processed: {total_processed:,},  Collected: {len(collected):,} English tweets in August")

            # Language filter
            if row.get('lang', '') != LANGUAGE:
                skipped_language += 1
                continue

            # Date filter
            tweet_date = row.get('date', '')
            if not tweet_date or not (DATE_START <= tweet_date <= DATE_END):
                skipped_date += 1
                continue

            try:
                tweet = {
                    'tweet_id':               row.get('id'),
                    'date':                   row.get('date'),
                    'language':               row.get('lang'),
                    'text':                   row.get('text'),
                    'raw_content':            row.get('rawContent'),
                    'like_count':             row.get('likeCount', 0) or 0,
                    'retweet_count':          row.get('retweetCount', 0) or 0,
                    'reply_count':            row.get('replyCount', 0) or 0,
                    'quote_count':            row.get('quoteCount', 0) or 0,
                    'view_count':             row.get('viewCount', 0) or 0,
                    'retweeted_tweet':        row.get('retweetedTweet'),
                    'quoted_tweet':           row.get('quotedTweet'),
                    'in_reply_to_screen_name': row.get('in_reply_to_screen_name'),
                    'hashtags':               row.get('hashtags'),
                    'url':                    row.get('url'),
                    'user':                   row.get('user'),
                    'post_type':              derive_post_type(row),
                }
                collected.append(tweet)
            except Exception:
                continue

            if len(collected) >= MAX_COLLECT:
                print()
                print(f"Target reached: {len(collected):,} English tweets in August")
                break

    except KeyboardInterrupt:
        print()
        print("WARNING: Interrupted by user")
    except Exception as e:
        print()
        print(f"WARNING: Stream error: {e}")

    print()
    print("Streaming Completed")
    print()
    print(f"Total processed:          {total_processed:,}")
    print(f"English tweets in August: {len(collected):,}")
    print(f"Skipped (language):       {skipped_language:,}")
    print(f"Skipped (date):           {skipped_date:,}")
    print("-" * 70)
    print()

    return collected


# ENGAGEMENT-BASED STRATIFIED SAMPLING

def engagement_stratified_sample(df, n):
    # Samples n tweets stratified by engagement tier: low/medium/high at 50/30/20%
    print("Engagement-stratified sampling:")
    print()
    print(f"Target sample size: {n}")
    print()

    df['engagement_level'] = df['like_count'].apply(classify_engagement)

    print("Pool composition by engagement tier:")
    for tier in ['low', 'medium', 'high']:
        lo, hi = ENGAGEMENT_TIERS[tier]
        count = (df['engagement_level'] == tier).sum()
        pct = count / len(df) * 100
        hi_str = str(hi) if hi else '+'
        print(f"  {tier.capitalize():8} ({lo}-{hi_str} likes): {count:,} ({pct:.1f}%)")
    print("=" * 70)
    print()

    samples = []

    for tier, prop in ENGAGEMENT_PROPORTIONS.items():
        n_tier = int(n * prop)
        tier_subset = df[df['engagement_level'] == tier]

        print(f"{tier.capitalize():8} engagement target: {n_tier},  available: {len(tier_subset):,}")

        if len(tier_subset) >= n_tier:
            sampled_tier = tier_subset.sample(n=n_tier, random_state=RANDOM_SEED)
        else:
            sampled_tier = tier_subset
            print(f"Only {len(tier_subset)} {tier} engagement tweets available (target: {n_tier})")

        samples.append(sampled_tier)

    result = pd.concat(samples, ignore_index=True)

    print()
    print("Sample Composition:")
    for tier in ['low', 'medium', 'high']:
        count = (result['engagement_level'] == tier).sum()
        pct = count / len(result) * 100
        print(f"  {tier.capitalize():8} engagement: {count} ({pct:.1f}%)")

    return result


# PROCESSING AND OUTPUT

def process_sample(tweets_list):
    # Converts collected tweets to DataFrame, runs stratified sampling, adds metadata
    if not tweets_list:
        print("ERROR: No tweets collected")
        return None

    df = pd.DataFrame(tweets_list)

    print()
    print(f"August 2024 tweets: {len(df):,} tweets")
    print()
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    print("Post type distribution:")
    print(df['post_type'].value_counts())
    print()
    print("Engagement statistics (likes):")
    print(df['like_count'].describe())
    print()

    sampled = engagement_stratified_sample(df, SAMPLE_SIZE)

    sampled = sampled.reset_index(drop=True)
    sampled['row_id'] = range(1, len(sampled) + 1)

    sampled['dataset_split'] = None
    sampled['sampling_strategy'] = 'engagement_stratified'
    sampled['sampled_at'] = datetime.now()

    print()
    print("Final Sample Summary:")
    print()
    print(f"Total: {len(sampled)} tweets")
    print(f"Row IDs: 1 to {len(sampled)}")
    print(f"Date range: {sampled['date'].min()} to {sampled['date'].max()}")
    print()
    print("Post type distribution:")
    print(sampled['post_type'].value_counts())
    print()
    print("Engagement distribution:")
    print(sampled['like_count'].describe())
    print()


    return sampled


def save_parquet(df):
    # Casts column types, reorders, and writes to parquet
    df['row_id']       = df['row_id'].astype('int32')
    df['tweet_id']     = df['tweet_id'].astype('int64')
    df['date']         = pd.to_datetime(df['date'], errors='coerce')
    df['like_count']   = df['like_count'].fillna(0).astype('int32')
    df['retweet_count']= df['retweet_count'].fillna(0).astype('int32')
    df['reply_count']  = df['reply_count'].fillna(0).astype('int32')
    df['quote_count']  = df['quote_count'].fillna(0).astype('int32')
    df['view_count']   = pd.to_numeric(df['view_count'], errors='coerce').fillna(0).astype('int64')

    ordered_cols = [
        'row_id',
        'date',
        'post_type',
        'engagement_level',
        'language',
        'like_count',
        'retweet_count',
        'reply_count',
        'quote_count',
        'view_count',
        'text',
        'hashtags',
        'tweet_id',
        'retweeted_tweet',
        'quoted_tweet',
        'url',
        'user',
        'raw_content',
        'in_reply_to_screen_name',
        'dataset_split',
        'sampling_strategy',
        'sampled_at',
    ]

    df = df[ordered_cols]

    os.makedirs('data_aug', exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    file_size = os.path.getsize(OUTPUT_PATH) / 1024

    print()
    print("Saved sample to parquet:")
    print(f"Location:   {OUTPUT_PATH}")
    print(f"Size:       {file_size:.1f} KB")
    print(f"Rows:       {len(df)}")
    print(f"Row IDs:    1 to {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"All English: {(df['language'] == LANGUAGE).all()}")


# MAIN

def main():
    print()
    print("August 2024 Engagement-Stratified Sample:")
    print()
    print(f"Target sample:    {SAMPLE_SIZE} tweets")
    print(f"Source parts:     {AUGUST_PARTS}")
    print(f"Date range:       {DATE_START} to {DATE_END}")
    print(f"Collection pool:  Up to {MAX_COLLECT:,} English tweets")
    print(f"Language:         {LANGUAGE}")
    print(f"Stratification:   Engagement tiers low/med/high (50/30/20)")
    print(f"Random seed:      {RANDOM_SEED}")
    print()

    tweets = load_august_tweets()

    if len(tweets) < SAMPLE_SIZE:
        print(f"ERROR: Only {len(tweets)} tweets collected (need {SAMPLE_SIZE})")
        return

    sampled_df = process_sample(tweets)

    if sampled_df is None:
        return

    save_parquet(sampled_df)


if __name__ == "__main__":
    main()