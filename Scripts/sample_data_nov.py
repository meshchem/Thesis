"""
Stratified Tweet Sampler for LLM Annotation Study - November 2024 Election Data

Loads tweets from November 2024 parts (39-44) using HuggingFace datasets library
with data_files parameter to load only specific November chunk files.

Data Source:
    - Parts 39-44 of the dataset (November 2024)
    - Uses data_files to load only november_chunk_*.parquet files
    - Streams data to avoid memory issues

Strategy:
    - Loads English tweets from November parts only
    - Stratifies by engagement level (30% high, 70% low)
    - Post type is characterized but not used for stratification
    - Produces 500-tweet sample suitable for annotation quality testing

Author: Research Project
Date: 2025
"""

import pandas as pd
from datasets import load_dataset
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_SIZE = 500
RANDOM_SEED = 42
OUTPUT_PATH = "data/tweets.parquet"

# Dataset info
DATASET_REPO = "deadbirds/usc-x-24-us-election-parquet"

# November 2024 parts to load
NOVEMBER_PARTS = [44]

# Number of tweets to collect before sampling
# Set higher to get more diverse pool
MAX_COLLECT = 650000  # Collect 100k English tweets for sampling pool

LANGUAGE = "en"

# Stratification settings
ENGAGEMENT_PROPORTIONS = {
    'high': 0.30,
    'low': 0.70
}

ENGAGEMENT_THRESHOLD = 100

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def derive_post_type(row):
    """
    Classify tweet into one of four post types.
    
    Args:
        row: Dictionary or DataFrame row containing tweet data
        
    Returns:
        str: One of 'retweet', 'quote', 'reply', or 'original'
    """
    if row.get('retweetedTweet') is not None:
        return 'retweet'
    elif row.get('quotedTweet') is not None:
        return 'quote'
    elif row.get('in_reply_to_screen_name') is not None:
        return 'reply'
    return 'original'

def clean_list_field(field_value):
    """
    Safely convert field to list.
    
    Args:
        field_value: Value that should be a list
        
    Returns:
        list: Empty list if None/invalid, otherwise the original list
    """
    if field_value is None:
        return []
    if isinstance(field_value, list):
        return field_value
    return []

# ============================================================================
# DATA LOADING FROM NOVEMBER PARTS
# ============================================================================

def load_november_tweets():
    """
    Load tweets from November 2024 parts using datasets library.
    
    Uses data_files parameter to load only November chunk files from
    parts 39-44, avoiding the need to download the entire dataset.
    
    Returns:
        list: List of tweet dictionaries
    """
    print("LOADING NOVEMBER 2024 DATA")
    print("=" * 70)
    print(f"Target parts: {NOVEMBER_PARTS}")
    print(f"Loading november_chunk_*.parquet files from each part")
    print("=" * 70)
    print()
    
    # Build list of data files to load
    data_files = []
    for part_num in NOVEMBER_PARTS:
        file_pattern = f"part_{part_num}/november_chunk_*.parquet"
        data_files.append(file_pattern)
    
    print("Data file patterns:")
    for pattern in data_files:
        print(f"  {pattern}")
    print()
    
    print("Loading dataset in streaming mode...")
    print("This will download only the November chunks")
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
    
    try:
        for row in dataset:
            total_processed += 1
            
            # Progress indicator
            if total_processed % 5000 == 0:
                print(f"   Processed: {total_processed:,} | Collected: {len(collected):,} English tweets")
            
            # Language filter
            tweet_lang = row.get('lang', '')
            if tweet_lang != LANGUAGE:
                skipped_language += 1
                continue
            
            # Extract tweet data
            try:
                tweet = {
                    'tweet_id': row.get('id'),
                    'date': row.get('date'),
                    'language': row.get('lang'),
                    'text': row.get('text'),
                    'raw_content': row.get('rawContent'),
                    'like_count': row.get('likeCount', 0) or 0,
                    'retweet_count': row.get('retweetCount', 0) or 0,
                    'reply_count': row.get('replyCount', 0) or 0,
                    'retweeted_tweet': row.get('retweetedTweet'),
                    'quoted_tweet': row.get('quotedTweet'),
                    'in_reply_to_screen_name': row.get('in_reply_to_screen_name'),
                    'hashtags': clean_list_field(row.get('hashtags')),
                    'url': row.get('url'),
                    'user': row.get('user'),
                    'post_type': derive_post_type(row),
                }
                
                collected.append(tweet)
                
            except Exception as e:
                continue
            
            # Stop when we have enough
            if len(collected) >= MAX_COLLECT:
                print()
                print(f"Target reached: {len(collected):,} English tweets collected")
                break
    
    except KeyboardInterrupt:
        print()
        print()
        print("WARNING: Interrupted by user")
    except Exception as e:
        print()
        print(f"WARNING: Stream error: {e}")
    
    print()
    print("=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Total processed: {total_processed:,}")
    print(f"English tweets collected: {len(collected):,}")
    print(f"Non-English skipped: {skipped_language:,}")
    print("=" * 70)
    print()
    
    return collected

# ============================================================================
# STRATIFIED SAMPLING
# ============================================================================

def stratified_sample(df, n):
    """
    Perform stratified sampling by engagement level.
    
    Args:
        df (pd.DataFrame): DataFrame containing tweets
        n (int): Total number of tweets to sample
        
    Returns:
        pd.DataFrame: Stratified sample
    """
    print("STRATIFIED SAMPLING")
    print("=" * 70)
    print(f"Target sample size: {n}")
    print()
    print("Engagement stratification:")
    for level, proportion in ENGAGEMENT_PROPORTIONS.items():
        target_n = int(n * proportion)
        print(f"  {level:5} engagement -> {proportion*100:5.1f}% ({target_n:3d} tweets)")
    print()
    print(f"Engagement threshold: {ENGAGEMENT_THRESHOLD} likes")
    print(f"  High engagement: >= {ENGAGEMENT_THRESHOLD} likes")
    print(f"  Low engagement:  <  {ENGAGEMENT_THRESHOLD} likes")
    print("=" * 70)
    print()
    
    # Classify by engagement
    df['engagement_level'] = df['like_count'].apply(
        lambda x: 'high' if x >= ENGAGEMENT_THRESHOLD else 'low'
    )
    
    # Calculate targets
    n_high = int(n * ENGAGEMENT_PROPORTIONS['high'])
    n_low = n - n_high
    
    # Split by engagement
    high_eng = df[df['engagement_level'] == 'high']
    low_eng = df[df['engagement_level'] == 'low']
    
    print(f"Available tweets:")
    print(f"  High engagement: {len(high_eng):,} tweets")
    print(f"  Low engagement:  {len(low_eng):,} tweets")
    print()
    
    # Sample from each level
    if len(high_eng) >= n_high:
        high_sample = high_eng.sample(n=n_high, random_state=RANDOM_SEED)
        print(f"Sampled {n_high} high engagement tweets")
    else:
        high_sample = high_eng
        print(f"WARNING: Only {len(high_eng)} high engagement tweets (target: {n_high})")
    
    if len(low_eng) >= n_low:
        low_sample = low_eng.sample(n=n_low, random_state=RANDOM_SEED)
        print(f"Sampled {n_low} low engagement tweets")
    else:
        low_sample = low_eng
        print(f"WARNING: Only {len(low_eng)} low engagement tweets (target: {n_low})")
    
    # Combine
    result = pd.concat([high_sample, low_sample], ignore_index=True)
    result = result.drop(columns=['engagement_level'])
    
    print()
    print(f"Final sample: {len(result)} tweets")
    print("=" * 70)
    
    return result

# ============================================================================
# PROCESSING AND OUTPUT
# ============================================================================

def process_sample(tweets_list):
    """
    Convert collected tweets to DataFrame and apply sampling.
    
    Args:
        tweets_list (list): List of tweet dictionaries
        
    Returns:
        pd.DataFrame: Processed sample, or None if insufficient data
    """
    if len(tweets_list) == 0:
        print("ERROR: No tweets collected")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(tweets_list)
    
    print("=" * 70)
    print(f"NOVEMBER TWEET POOL: {len(df):,} tweets")
    print("=" * 70)
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    print("Post type distribution:")
    print(df['post_type'].value_counts())
    print()
    print("Engagement statistics (likes):")
    print(df['like_count'].describe())
    print("=" * 70)
    print()
    
    # Perform stratified sampling
    sampled = stratified_sample(df, SAMPLE_SIZE)
    
    # Create sequential row IDs
    sampled = sampled.reset_index(drop=True)
    sampled['row_id'] = range(1, len(sampled) + 1)
    
    # Add metadata
    sampled['dataset_split'] = None
    sampled['sampling_strategy'] = 'stratified_engagement'
    sampled['sampled_at'] = datetime.now()
    
    print()
    print("=" * 70)
    print("FINAL SAMPLE")
    print("=" * 70)
    print(f"Total: {len(sampled)} tweets")
    print(f"Row IDs: 1 to {len(sampled)}")
    print()
    print("Post type distribution:")
    print(sampled['post_type'].value_counts())
    print()
    print("Engagement distribution:")
    print(sampled['like_count'].describe())
    print("=" * 70)
    
    return sampled

def save_parquet(df):
    """
    Save DataFrame to parquet with annotation-friendly column ordering.
    
    Args:
        df (pd.DataFrame): DataFrame to save
    """
    # Data types
    df['row_id'] = df['row_id'].astype('int32')
    df['tweet_id'] = df['tweet_id'].astype('int64')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['like_count'] = df['like_count'].fillna(0).astype('int32')
    df['retweet_count'] = df['retweet_count'].fillna(0).astype('int32')
    df['reply_count'] = df['reply_count'].fillna(0).astype('int32')
    
    # Column order
    ordered_cols = [
        'row_id',
        'tweet_id',
        'date',
        'post_type',
        'language',
        'like_count',
        'retweet_count',
        'reply_count',
        'text',
        'raw_content',
        'hashtags',
        'retweeted_tweet',
        'quoted_tweet',
        'url',
        'user',
        'dataset_split',
        'sampling_strategy',
        'sampled_at'
    ]
    
    df = df[ordered_cols]
    
    # Save
    os.makedirs('data', exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    
    file_size = os.path.getsize(OUTPUT_PATH) / 1024
    
    print()
    print("=" * 70)
    print("SAVED TO PARQUET")
    print("=" * 70)
    print(f"Location: {OUTPUT_PATH}")
    print(f"Size: {file_size:.1f} KB")
    print(f"Rows: {len(df)}")
    print(f"Row IDs: 1 to {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"All English: {(df['language'] == LANGUAGE).all()}")
    print("=" * 70)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main execution function.
    
    Workflow:
        1. Load November parts (39-44) using data_files parameter
        2. Stream and collect English tweets
        3. Apply stratified sampling by engagement (30% high, 70% low)
        4. Save 500-tweet sample to parquet
    """
    print()
    print("=" * 70)
    print("NOVEMBER 2024 ELECTION TWEET SAMPLER")
    print("=" * 70)
    print(f"Target sample: {SAMPLE_SIZE} tweets")
    print(f"Source parts: {NOVEMBER_PARTS}")
    print(f"Collection pool: Up to {MAX_COLLECT:,} English tweets")
    print(f"Language: {LANGUAGE}")
    print(f"Stratification: Engagement (30% high, 70% low)")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 70)
    print()
    
    # Load November data
    tweets = load_november_tweets()
    
    if len(tweets) < SAMPLE_SIZE:
        print(f"ERROR: Only {len(tweets)} tweets collected (need {SAMPLE_SIZE})")
        print()
        print("Possible solutions:")
        print(f"1. Increase MAX_COLLECT (currently {MAX_COLLECT:,})")
        print("2. Reduce SAMPLE_SIZE")
        print("3. Check internet connection")
        return
    
    # Process and sample
    sampled_df = process_sample(tweets)
    
    if sampled_df is None:
        return
    
    # Save
    save_parquet(sampled_df)
    
    print()
    print("NEXT STEPS:")
    print("1. Inspect sample: python view_parquet.py")
    print("2. Assign development/test splits")
    print("3. Begin annotation")
    print()

if __name__ == "__main__":
    main()





# def stratified_sample(df, n):
#     """
#     Perform stratified sampling by engagement level.
    
#     Args:
#         df (pd.DataFrame): DataFrame containing tweets
#         n (int): Total number of tweets to sample
        
#     Returns:
#         pd.DataFrame: Stratified sample
#     """

#     print("STRATIFIED SAMPLING")
#     print("=" * 70)
#     print(f"Target sample size: {n}")
#     print()
#     print("Engagement stratification:")
#     for level, proportion in ENGAGEMENT_PROPORTIONS.items():
#         target_n = int(n * proportion)
#         print(f"  {level:5} engagement -> {proportion*100:5.1f}% ({target_n:3d} tweets)")
#     print()
#     print(f"Engagement threshold: {ENGAGEMENT_THRESHOLD} likes")
#     print(f"  High engagement: >= {ENGAGEMENT_THRESHOLD} likes")
#     print(f"  Low engagement:  <  {ENGAGEMENT_THRESHOLD} likes")
#     print("=" * 70)
#     print()
    
#     # Classify by engagement
#     df['engagement_level'] = df['like_count'].apply(
#         lambda x: 'high' if x >= ENGAGEMENT_THRESHOLD else 'low'
#     )
    
#     # Calculate targets
#     n_high = int(n * ENGAGEMENT_PROPORTIONS['high'])
#     n_low = n - n_high
    
#     # Split by engagement
#     high_eng = df[df['engagement_level'] == 'high']
#     low_eng = df[df['engagement_level'] == 'low']
    
#     print(f"Available tweets:")
#     print(f"  High engagement: {len(high_eng):,} tweets")
#     print(f"  Low engagement:  {len(low_eng):,} tweets")
#     print()
    
#     # Sample from each level
#     if len(high_eng) >= n_high:
#         high_sample = high_eng.sample(n=n_high, random_state=RANDOM_SEED)
#         print(f"Sampled {n_high} high engagement tweets")
#     else:
#         high_sample = high_eng
#         print(f"WARNING: Only {len(high_eng)} high engagement tweets (target: {n_high})")
    
#     if len(low_eng) >= n_low:
#         low_sample = low_eng.sample(n=n_low, random_state=RANDOM_SEED)
#         print(f"Sampled {n_low} low engagement tweets")
#     else:
#         low_sample = low_eng
#         print(f"WARNING: Only {len(low_eng)} low engagement tweets (target: {n_low})")
    
#     # Combine
#     result = pd.concat([high_sample, low_sample], ignore_index=True)
#     result = result.drop(columns=['engagement_level'])
    
#     print()
#     print(f"Final sample: {len(result)} tweets")
#     print("=" * 70)
    
#     return result
    
# def random_sample(df, n):
#     """
#     Perform random sampling without stratification.
#     Still reports engagement metrics for the sample.
    
#     Args:
#         df (pd.DataFrame): DataFrame containing tweets
#         n (int): Total number of tweets to sample
        
#     Returns:
#         pd.DataFrame: Random sample
#     """
#     print("RANDOM SAMPLING")
#     print("=" * 70)
#     print(f"Target sample size: {n}")
#     print(f"Available tweets: {len(df):,}")
#     print()
    
#     # Classify by engagement for reporting purposes
#     df['engagement_level'] = df['like_count'].apply(
#         lambda x: 'high' if x >= ENGAGEMENT_THRESHOLD else 'low'
#     )
    
#     # Show pool composition
#     high_count = (df['engagement_level'] == 'high').sum()
#     low_count = (df['engagement_level'] == 'low').sum()
#     print("Pool composition:")
#     print(f"  High engagement (>= {ENGAGEMENT_THRESHOLD} likes): {high_count:,} tweets ({high_count/len(df)*100:.1f}%)")
#     print(f"  Low engagement  (<  {ENGAGEMENT_THRESHOLD} likes): {low_count:,} tweets ({low_count/len(df)*100:.1f}%)")
#     print("=" * 70)
#     print()
    
#     # Random sample
#     if len(df) >= n:
#         result = df.sample(n=n, random_state=RANDOM_SEED)
#         print(f"Sampled {n} tweets randomly")
#     else:
#         result = df
#         print(f"WARNING: Only {len(df)} tweets available (target: {n})")
    
#     # Report sample composition
#     sample_high = (result['engagement_level'] == 'high').sum()
#     sample_low = (result['engagement_level'] == 'low').sum()
    
#     print()
#     print("Sample composition:")
#     print(f"  High engagement: {sample_high} tweets ({sample_high/len(result)*100:.1f}%)")
#     print(f"  Low engagement:  {sample_low} tweets ({sample_low/len(result)*100:.1f}%)")
    
#     # Drop the temporary engagement_level column
#     result = result.drop(columns=['engagement_level'])
    
#     print()
#     print(f"Final sample: {len(result)} tweets")
#     print("=" * 70)
    
#     return result