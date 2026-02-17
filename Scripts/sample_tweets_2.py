"""
Stratified Tweet Sampler for LLM Annotation Study

This module samples tweets from the US 2024 Election dataset with stratification
by engagement level to ensure diverse representation for annotation.

Strategy:
    - Collects English tweets from dataset stream
    - Stratifies by engagement level (30% high, 70% low)
    - Post type is characterized but not used for stratification
    - Produces balanced sample suitable for annotation quality testing

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

SAMPLE_SIZE = 300
RANDOM_SEED = 42
OUTPUT_PATH = "data/tweets.parquet"

# Collection settings
MAX_COLLECT = 50000  # Number of English tweets to collect before sampling
LANGUAGE = "en"

# Stratification settings - ENGAGEMENT ONLY
ENGAGEMENT_PROPORTIONS = {
    'high': 0.30,  # 30% high engagement (>= 100 likes)
    'low': 0.70    # 70% low engagement (< 100 likes)
}

# Engagement threshold
ENGAGEMENT_THRESHOLD = 100

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def derive_post_type(row):
    """
    Classify tweet into one of four post types based on content structure.
    
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
    Safely convert field to list, handling None and non-list values.
    
    Args:
        field_value: Value that should be a list (e.g., hashtags)
        
    Returns:
        list: Empty list if None/invalid, otherwise the original list
    """
    if field_value is None:
        return []
    if isinstance(field_value, list):
        return field_value
    return []

# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_tweets():
    """
    Stream dataset and collect English tweets.
    
    Collects MAX_COLLECT English tweets from the dataset stream.
    Post type is derived for characterization but not used for sampling.
    
    Returns:
        list: List of dictionaries, each containing one tweet's data
    """
    print("STREAMING DATASET")
    print("=" * 70)
    print(f"Collecting {MAX_COLLECT:,} English tweets from dataset")
    print(f"Language filter: {LANGUAGE}")
    print("=" * 70)
    print()
    
    print("Loading dataset in streaming mode...")
    dataset = load_dataset(
        "deadbirds/usc-x-24-us-election-parquet",
        split="train",
        streaming=True
    )
    
    collected = []
    total_processed = 0
    skipped_language = 0
    
    print(f"Collecting English tweets...")
    print()
    
    try:
        for row in dataset:
            total_processed += 1
            
            # Progress indicator every 5000 tweets
            if total_processed % 5000 == 0:
                print(f"   Processed: {total_processed:,} | Collected: {len(collected):,}")
            
            # Apply language filter
            tweet_lang = row.get('lang', '')
            if tweet_lang != LANGUAGE:
                skipped_language += 1
                continue
            
            # Extract relevant fields from row
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
                    'post_type': derive_post_type(row),  # For characterization only
                }
                
                collected.append(tweet)
                
            except Exception as e:
                # Skip malformed tweets
                continue
            
            # Stop when target reached
            if len(collected) >= MAX_COLLECT:
                print()
                print(f"Target reached: {len(collected):,} English tweets")
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
    Perform stratified sampling by engagement level only.
    
    Samples 30% high engagement (>= 100 likes) and 70% low engagement (< 100 likes).
    Post type is preserved as metadata but not used for stratification.
    
    Args:
        df (pd.DataFrame): DataFrame containing all collected tweets
        n (int): Total number of tweets to sample
        
    Returns:
        pd.DataFrame: Stratified sample of tweets
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
    
    # Classify tweets by engagement level
    df['engagement_level'] = df['like_count'].apply(
        lambda x: 'high' if x >= ENGAGEMENT_THRESHOLD else 'low'
    )
    
    # Calculate targets
    n_high = int(n * ENGAGEMENT_PROPORTIONS['high'])
    n_low = n - n_high  # Ensures we get exactly n tweets
    
    # Split by engagement
    high_eng = df[df['engagement_level'] == 'high']
    low_eng = df[df['engagement_level'] == 'low']
    
    print(f"Available tweets:")
    print(f"  High engagement: {len(high_eng):,} tweets available")
    print(f"  Low engagement:  {len(low_eng):,} tweets available")
    print()
    
    # Sample from each engagement level
    if len(high_eng) >= n_high:
        high_sample = high_eng.sample(n=n_high, random_state=RANDOM_SEED)
        print(f"Sampled {n_high} high engagement tweets")
    else:
        high_sample = high_eng
        print(f"WARNING: Only {len(high_eng)} high engagement tweets available (target: {n_high})")
    
    if len(low_eng) >= n_low:
        low_sample = low_eng.sample(n=n_low, random_state=RANDOM_SEED)
        print(f"Sampled {n_low} low engagement tweets")
    else:
        low_sample = low_eng
        print(f"WARNING: Only {len(low_eng)} low engagement tweets available (target: {n_low})")
    
    # Combine samples
    result = pd.concat([high_sample, low_sample], ignore_index=True)
    
    # Remove temporary engagement_level column
    result = result.drop(columns=['engagement_level'])
    
    print()
    print(f"Final sample: {len(result)} tweets")
    print("=" * 70)
    
    return result

def process_sample(tweets_list):
    """
    Convert collected tweets to DataFrame and apply stratified sampling.
    
    Args:
        tweets_list (list): List of tweet dictionaries
        
    Returns:
        pd.DataFrame: Processed and sampled DataFrame, or None if insufficient data
    """
    if len(tweets_list) == 0:
        print("ERROR: No tweets collected")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(tweets_list)
    
    # post_type already derived during collection, no need to recalculate
    
    print("=" * 70)
    print(f"COLLECTED POOL: {len(df):,} tweets")
    print("=" * 70)
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    print("Post type distribution in collected pool:")
    print(df['post_type'].value_counts())
    print()
    print("Engagement statistics (likes):")
    print(df['like_count'].describe())
    print("=" * 70)
    print()
    
    # Perform stratified sampling
    sampled = stratified_sample(df, SAMPLE_SIZE)
    
    # Create sequential row IDs (primary key for annotation)
    sampled = sampled.reset_index(drop=True)
    sampled['row_id'] = range(1, len(sampled) + 1)
    
    # Add metadata fields
    sampled['dataset_split'] = None  # To be assigned later (dev/test)
    sampled['sampling_strategy'] = 'stratified_engagement'
    sampled['sampled_at'] = datetime.now()
    
    print()
    print("=" * 70)
    print("FINAL SAMPLE")
    print("=" * 70)
    print(f"Total tweets: {len(sampled)}")
    print(f"Row IDs: 1 to {len(sampled)}")
    print(f"Date range: {sampled['date'].min()} to {sampled['date'].max()}")
    print()
    print("Post type distribution in sample:")
    print(sampled['post_type'].value_counts())
    print()
    print("Engagement distribution in sample:")
    print(sampled['like_count'].describe())
    
    # Calculate hashtag usage statistics
    has_hashtags = sampled['hashtags'].apply(
        lambda x: len(x) > 0 if isinstance(x, list) else False
    ).sum()
    print()
    print("Hashtag usage:")
    print(f"  Tweets with hashtags: {has_hashtags} "
          f"({has_hashtags/len(sampled)*100:.1f}%)")
    print("=" * 70)
    
    return sampled

# ============================================================================
# OUTPUT
# ============================================================================

def save_parquet(df):
    """
    Save DataFrame to parquet file with annotation-friendly column ordering.
    
    Columns are ordered to prioritize metadata fields (IDs, dates, types)
    before content fields (text, hashtags) for easier manual inspection.
    
    Args:
        df (pd.DataFrame): DataFrame to save
    """
    # Convert to appropriate data types
    df['row_id'] = df['row_id'].astype('int32')
    df['tweet_id'] = df['tweet_id'].astype('int64')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['like_count'] = df['like_count'].fillna(0).astype('int32')
    df['retweet_count'] = df['retweet_count'].fillna(0).astype('int32')
    df['reply_count'] = df['reply_count'].fillna(0).astype('int32')
    
    # Define column order - metadata first, content second
    ordered_cols = [
        'row_id',           # Primary key for annotation tables
        'tweet_id',         # Original HuggingFace dataset ID
        'date',             # Timestamp
        'post_type',        # original/retweet/quote/reply
        'language',         # Language code (all 'en')
        'like_count',       # Engagement metrics
        'retweet_count',
        'reply_count',
        'text',             # Main tweet content
        'raw_content',      # Unprocessed content
        'hashtags',         # List of hashtags
        'retweeted_tweet',  # Original tweet content (if retweet)
        'quoted_tweet',     # Original tweet content (if quote)
        'url',              # Tweet URL
        'user',             # User metadata object
        'dataset_split',    # dev/test assignment (null initially)
        'sampling_strategy', # Method used for sampling
        'sampled_at'        # Timestamp of sampling
    ]
    
    df = df[ordered_cols]
    
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to parquet format
    df.to_parquet(OUTPUT_PATH, index=False)
    
    # Calculate file size for reporting
    file_size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    
    print()
    print("=" * 70)
    print("SAVED TO PARQUET")
    print("=" * 70)
    print(f"Location: {OUTPUT_PATH}")
    print(f"Size: {file_size_kb:.1f} KB")
    print(f"Rows: {len(df)}")
    print(f"Row IDs: 1 to {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"All English: {(df['language'] == LANGUAGE).all()}")
    print("=" * 70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    
    Workflow:
        1. Stream dataset and collect English tweets
        2. Apply stratified sampling by engagement level (30% high, 70% low)
        3. Characterize post types in final sample
        4. Save final sample to parquet file
    """
    print()
    print("=" * 70)
    print("STRATIFIED TWEET SAMPLER FOR ANNOTATION STUDY")
    print("=" * 70)
    print(f"Target sample size: {SAMPLE_SIZE}")
    print(f"Collection pool: {MAX_COLLECT:,} English tweets")
    print(f"Language filter: {LANGUAGE}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Stratification: Engagement level only (30% high, 70% low)")
    print(f"Post type: Characterized but not stratified")
    print("=" * 70)
    print()
    
    # Step 1: Collect tweets from stream
    tweets = collect_tweets()
    
    if len(tweets) < SAMPLE_SIZE:
        print()
        print(f"ERROR: Only collected {len(tweets)} tweets (need {SAMPLE_SIZE})")
        print("Increase MAX_COLLECT or reduce SAMPLE_SIZE")
        return
    
    # Step 2: Apply stratified sampling
    sampled_df = process_sample(tweets)
    
    if sampled_df is None:
        return
    
    # Step 3: Save output
    save_parquet(sampled_df)
    
    # print()
    # print("NEXT STEPS:")
    # print("1. Inspect sample: python view_parquet.py")
    # print("2. Assign development/test splits")
    # print("3. Begin annotation process")
    # print()

if __name__ == "__main__":
    main()