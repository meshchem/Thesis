"""
Alternative sampling approach with stratified options.
Allows you to control the mix of tweet types and engagement levels.
"""

import pandas as pd
from datasets import load_dataset
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION - MODIFY THESE
# ============================================================================

SAMPLE_SIZE = 300
RANDOM_SEED = 42
OUTPUT_PATH = "data/tweets.parquet"

# Filters
LANGUAGE = "en"
DATE_START = None  # "2024-10-01" or None
DATE_END = None    # "2024-10-31" or None

# Sampling strategy
SAMPLING_STRATEGY = "random"  # Options: "random", "stratified_type", "stratified_engagement"

# If stratified by type, define proportions (must sum to 1.0)
TYPE_PROPORTIONS = {
    'original': 0.4,
    'retweet': 0.3,
    'quote': 0.2,
    'reply': 0.1
}

# If stratified by engagement, define threshold
ENGAGEMENT_THRESHOLD = 100  # Tweets with >100 likes = "high engagement"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def derive_post_type(row):
    """Classify tweet type."""
    if row.get('retweetedTweet') is not None:
        return 'retweet'
    elif row.get('quotedTweet') is not None:
        return 'quote'
    elif row.get('in_reply_to_screen_name') is not None:
        return 'reply'
    return 'original'

def clean_list_field(field_value):
    """Clean list fields (hashtags, mentions)."""
    if field_value is None:
        return []
    if isinstance(field_value, list):
        return field_value
    return []

def extract_user_id(user_obj):
    """Extract user ID from user object."""
    if user_obj is None:
        return None
    if isinstance(user_obj, dict):
        return user_obj.get('id')
    return None

def stream_and_collect(max_collect=5000):
    """
    Stream dataset without downloading everything.
    Collect English tweets up to max_collect limit.
    """
    print(f"Streaming dataset from HuggingFace...")
    print(f"   (This won't download 22GB - only streams what we need)\n")
    
    dataset = load_dataset(
        "deadbirds/usc-x-24-us-election-parquet",
        split="train",
        streaming=True
    )
    
    collected = []
    total_processed = 0
    skipped_language = 0
    
    for row in dataset:
        total_processed += 1
        
        # Language filter - FIXED to actually check properly
        tweet_lang = row.get('lang', '')
        if tweet_lang != LANGUAGE:
            skipped_language += 1
            continue
        
        # Date filter (optional)
        if DATE_START and DATE_END:
            tweet_date = row.get('date', '')
            if tweet_date:
                if not (DATE_START <= tweet_date <= DATE_END):
                    continue
        
        # Extract needed columns
        tweet = {
            'tweet_id': row.get('id'),
            'text': row.get('text'),
            'raw_content': row.get('rawContent'),
            'date': row.get('date'),
            'retweeted_tweet': row.get('retweetedTweet'),
            'quoted_tweet': row.get('quotedTweet'),
            'in_reply_to_screen_name': row.get('in_reply_to_screen_name'),
            'language': row.get('lang'),
            'like_count': row.get('likeCount', 0),
            'retweet_count': row.get('retweetCount', 0),
            'reply_count': row.get('replyCount', 0),
            'hashtags': clean_list_field(row.get('hashtags')),
            'mentioned_users': clean_list_field(row.get('mentionedUsers')),
            'user': row.get('user'),
            'user_id': extract_user_id(row.get('user')),
            'url': row.get('url'),
        }
        
        collected.append(tweet)
        
        # Progress
        if len(collected) % 500 == 0:
            print(f"   Collected: {len(collected)} tweets (processed {total_processed} total)")
        
        # Stop condition
        if len(collected) >= max_collect:
            break
    
    print(f"\n✓ Collection complete: {len(collected)} English tweets")
    print(f"   Processed: {total_processed} total tweets")
    print(f"   Skipped: {skipped_language} non-English tweets\n")
    return collected

def sample_random(df, n):
    """Simple random sampling."""
    print(f"📊 Strategy: Random sampling")
    return df.sample(n=min(n, len(df)), random_state=RANDOM_SEED)

def sample_stratified_type(df, n):
    """Stratified sampling by post type."""
    print(f"Strategy: Stratified by post type")
    print(f"   Target proportions: {TYPE_PROPORTIONS}\n")
    
    sampled_parts = []
    
    for post_type, proportion in TYPE_PROPORTIONS.items():
        target_n = int(n * proportion)
        subset = df[df['post_type'] == post_type]
        
        if len(subset) < target_n:
            print(f"Warning: Only! {len(subset)} '{post_type}' tweets available (target: {target_n})")
            sampled = subset
        else:
            sampled = subset.sample(n=target_n, random_state=RANDOM_SEED)
        
        sampled_parts.append(sampled)
        print(f"   {post_type:12} → {len(sampled):3} tweets")
    
    return pd.concat(sampled_parts, ignore_index=True)

# Stratified sampling by engagement level.
def sample_stratified_engagement(df, n):
    
    print(f"Strategy: Stratified by engagement (threshold: {ENGAGEMENT_THRESHOLD} likes)")
    
    df['engagement_level'] = df['like_count'].apply(
        lambda x: 'high' if x >= ENGAGEMENT_THRESHOLD else 'low'
    )
    
    high_eng = df[df['engagement_level'] == 'high']
    low_eng = df[df['engagement_level'] == 'low']
    
    # 50/50 split
    n_high = n // 2
    n_low = n - n_high
    
    print(f"   High engagement (≥{ENGAGEMENT_THRESHOLD}): {len(high_eng)} available → sampling {n_high}")
    print(f"   Low engagement (<{ENGAGEMENT_THRESHOLD}): {len(low_eng)} available → sampling {n_low}")
    
    sampled_high = high_eng.sample(n=min(n_high, len(high_eng)), random_state=RANDOM_SEED)
    sampled_low = low_eng.sample(n=min(n_low, len(low_eng)), random_state=RANDOM_SEED)
    
    result = pd.concat([sampled_high, sampled_low], ignore_index=True)
    result = result.drop(columns=['engagement_level'])
    
    return result

def perform_sampling(tweets_list):
    """Execute the chosen sampling strategy."""
    # Convert to DataFrame
    df = pd.DataFrame(tweets_list)
    
    # VERIFY: Filter out any non-English tweets that slipped through
    print(f"Before language filter: {len(df)} tweets")
    print(f"Language distribution: {df['language'].value_counts().to_dict()}")
    
    df = df[df['language'] == LANGUAGE].copy()
    print(f"After language filter: {len(df)} tweets (all {LANGUAGE})\n")
    
    # Add derived fields
    df['post_type'] = df.apply(derive_post_type, axis=1)
    
    print("="*60)
    print("DATASET COMPOSITION (before sampling)")
    print("="*60)
    print("\nPost types:")
    print(df['post_type'].value_counts())
    print(f"\nEngagement stats:")
    print(df['like_count'].describe())
    print("="*60 + "\n")
    
    # Choose sampling method
    if SAMPLING_STRATEGY == "random":
        sampled = sample_random(df, SAMPLE_SIZE)
    elif SAMPLING_STRATEGY == "stratified_type":
        sampled = sample_stratified_type(df, SAMPLE_SIZE)
    elif SAMPLING_STRATEGY == "stratified_engagement":
        sampled = sample_stratified_engagement(df, SAMPLE_SIZE)
    else:
        raise ValueError(f"Unknown strategy: {SAMPLING_STRATEGY}")
    
    # IMPORTANT: Create row_id (sequential 1, 2, 3...)
    sampled = sampled.reset_index(drop=True)
    sampled['row_id'] = range(1, len(sampled) + 1)
    
    # Add metadata
    sampled['dataset_split'] = None  # Assign later
    sampled['sampled_at'] = datetime.now()
    
    print(f"\n{'='*60}")
    print("FINAL SAMPLE")
    print("="*60)
    print(f"Total tweets: {len(sampled)}")
    print(f"Row IDs: 1 to {len(sampled)}")
    print(f"All English: {(sampled['language'] == LANGUAGE).all()}")
    print("\nPost type distribution:")
    print(sampled['post_type'].value_counts())
    print("\nEngagement distribution:")
    print(sampled['like_count'].describe())
    
    return sampled

def save_parquet(df):
    """Save to parquet with proper schema."""
    # Data type conversions
    df['row_id'] = df['row_id'].astype('int32')  # NEW: row_id
    df['tweet_id'] = df['tweet_id'].astype('int64')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['like_count'] = df['like_count'].fillna(0).astype('int32')
    df['retweet_count'] = df['retweet_count'].fillna(0).astype('int32')
    df['reply_count'] = df['reply_count'].fillna(0).astype('int32')
    
    # Column order - row_id FIRST
    ordered_cols = [
        'row_id',        # NEW: Sequential 1, 2, 3...
        'tweet_id',      # Original HuggingFace ID
        'text',  
        'date', 
        'post_type',
        'language', 
        'like_count', 
        'retweet_count', 
        'reply_count',
        'retweeted_tweet', 
        'quoted_tweet', 
        'in_reply_to_screen_name',
        'hashtags', 
        'raw_content',
        'mentioned_users', 
        'user', 
        'user_id', 
        'url',
        'dataset_split', 
        'sampled_at'
    ]
    
    df = df[ordered_cols]
    
    # Create directory
    os.makedirs('data', exist_ok=True)
    
    # Save
    df.to_parquet(OUTPUT_PATH, index=False)
    
    file_size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    
    print(f"\n{'='*60}")
    print("✓ SAVED TO PARQUET")
    print("="*60)
    print(f"Location: {OUTPUT_PATH}")
    print(f"Size: {file_size_kb:.2f} KB")
    print(f"Rows: {len(df):,}")
    print(f"Row IDs: 1 to {len(df)}")
    print(f"All English: {(df['language'] == LANGUAGE).all()}")
    print(f"Columns: {len(df.columns)}")
    print("="*60)

def main():
    print("\n" + "="*60)
    print("TWEET SAMPLER FOR LLM ANNOTATION STUDY")
    print("="*60)
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Strategy: {SAMPLING_STRATEGY}")
    print(f"Language: {LANGUAGE}")
    print(f"Random seed: {RANDOM_SEED}")
    print("="*60 + "\n")
    
    # Collect
    tweets = stream_and_collect(max_collect=SAMPLE_SIZE * 10)
    
    if len(tweets) < SAMPLE_SIZE:
        print(f"⚠️  Warning: Only {len(tweets)} tweets available, less than target {SAMPLE_SIZE}")
    
    # Sample
    sampled_df = perform_sampling(tweets)
    
    # Save
    save_parquet(sampled_df)
    
    print("\n📋 NEXT STEPS:")
    print("1. Inspect: pd.read_parquet('data/tweets.parquet')")
    print("2. Split into development/test sets")
    print("3. Start annotation process\n")

if __name__ == "__main__":
    main()