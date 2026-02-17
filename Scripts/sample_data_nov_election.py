"""
Stratified Tweet Sampler for LLM Annotation Study - November 2024 Election Period

Loads tweets from the election period (Oct 30 - Nov 10, 2024) - 5 days before
and 5 days after Election Day (Nov 5, 2024).

Data Source:
    - Part 44 of the dataset (November 2024)
    - Uses data_files to load only specific chunks: 112, 113
    - Filters to election period: Oct 30 - Nov 10
    - Streams data to avoid memory issues

Strategy:
    - Loads English tweets from election period only, between dates 
    - Random sampling (dataset is one that captures public discourse)
    - Post type and engagement is characterized but not used for stratification
    - Produces 500-tweet sample suitable for annotation quality testing

Author: Maria Meshcheryakova
Date: 2026
"""

import pandas as pd
from datasets import load_dataset
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_SIZE = 500
RANDOM_SEED = 45
OUTPUT_PATH = "data/tweets_aug.parquet"

# Dataset info
DATASET_REPO = "deadbirds/usc-x-24-us-election-parquet"

# November 2024 parts to load
NOVEMBER_PARTS = [44]
# November 2024 parts to load
SEPTEMBER_PARTS = [29, 30, 31, 32, 33, 34, 35]
OCTOBER_PARTS = [36, 37, 38]
AUGUST_PARTS = [23, 24, 25, 26, 27, 28]


# Specific chunks to load (contain election period data)
CHUNKS_TO_LOAD = [112, 113]

# Number of tweets to collect before sampling
MAX_COLLECT = 100000   # only 26,027 tweets that fit the election da and language criteria 

LANGUAGE = "en"

# Date range filter - Election period (Nov 5, 2024)
# 5 days before and 5 days after (inclusive)
DATE_START = "2024-10-30"  # 5 days before election
DATE_END = "2024-11-10"     # 5 days after election 


ENGAGEMENT_THRESHOLD = 100

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# debug (sample was not picking up hashtags)
def has_hashtags_check(hashtags):
    """
    Check if a tweet has hashtags, handling various data formats.
    
    Args:
        hashtags: Could be None, list, string, etc.
        
    Returns:
        bool: True if hashtags exist
    """
    if hashtags is None:
        return False
    if isinstance(hashtags, list):
        return len(hashtags) > 0
    if isinstance(hashtags, str):
        # Could be empty string or string representation like "[]"
        return hashtags not in ['', '[]', 'None']
    return False

def derive_post_type(row):
    """
    Classify tweet into one of four post types.
    
    Args:
        row: Dictionary or DataFrame row containing tweet data
        
    Returns:
        str: One of 'retweet', 'quote', 'reply', or 'tweet'
    """
    if row.get('retweetedTweet') is True:
        return 'retweet'
    elif row.get('quotedTweet') is True:
        return 'quote'
    elif row.get('in_reply_to_screen_name') is not None:
        return 'reply'
    return 'tweet'

# ============================================================================
# DATA LOADING FROM NOVEMBER PARTS
# ============================================================================

def load_november_tweets():
    """
    Load tweets from November 2024 parts using datasets library.
    
    Uses data_files parameter to load only specific November chunk files (112-113)
    from part 44, which contain the election period data.
    Filters to election period (Oct 30 - Nov 10).
    
    Returns:
        list: List of tweet dictionaries
    """
    print("LOADING NOVEMBER 2024 DATA")
    print("=" * 70)
    print(f"Target parts: {NOVEMBER_PARTS}")
    # print(f"Loading specific chunks: {CHUNKS_TO_LOAD}")
    print(f"Date filter: {DATE_START} to {DATE_END} (election period)")
    print("=" * 70)
    print()
    
    # Build list of specific data files to load (chunks 112-113)
    data_files = []
    for part_num in NOVEMBER_PARTS:
        for chunk_num in CHUNKS_TO_LOAD:
            file_path = f"part_{part_num}/november_chunk_{chunk_num}.parquet"
            data_files.append(file_path)
    
    print("Specific data files to load:")
    for filepath in data_files:
        print(f"  {filepath}")
    print()
    
    print("Loading dataset in streaming mode...")
    print("Only downloads the specified chunks")
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
            
            # Progress indicator
            if total_processed % 20000 == 0:
                print(f"   Processed: {total_processed:,} | Collected: {len(collected):,} English tweets in election period")
            
            # Language filter
            tweet_lang = row.get('lang', '')
            if tweet_lang != LANGUAGE:
                skipped_language += 1
                continue
            
            # Date filter - Election period only
            tweet_date = row.get('date', '')
            if tweet_date:
                if not (DATE_START <= tweet_date <= DATE_END):
                    skipped_date += 1
                    continue
            else:
                # Skip tweets without dates
                skipped_date += 1
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
                    'quote_count': row.get('quoteCount', 0) or 0,
                    'view_count': row.get('viewCount', 0) or 0,
                    'retweeted_tweet': row.get('retweetedTweet'),
                    'quoted_tweet': row.get('quotedTweet'),
                    'in_reply_to_screen_name': row.get('in_reply_to_screen_name'),
                    'hashtags': row.get('hashtags'),
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
                print(f"Target reached: {len(collected):,} English tweets in election period")
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
    print(f"English tweets in election period: {len(collected):,}")
    print(f"Skipped (language): {skipped_language:,}")
    print(f"Skipped (date): {skipped_date:,}")
    print("=" * 70)
    print()
    
    return collected

# ============================================================================
# Random Sampling
# ============================================================================


def random_sample(df, n):
    """
    Perform random sampling without stratification.
    Ensures at least 20% of sample has hashtags.
    Still reports engagement metrics for the sample.
    
    Args:
        df (pd.DataFrame): DataFrame containing tweets
        n (int): Total number of tweets to sample
        
    Returns:
        pd.DataFrame: Random sample
    """
    print("RANDOM SAMPLING")
    print("=" * 70)
    print(f"Target sample size: {n}")
    print(f"Available tweets: {len(df):,}")
    print()
    
    # Classify by engagement for reporting purposes
    df['engagement_level'] = df['like_count'].apply(
        lambda x: 'high' if x >= ENGAGEMENT_THRESHOLD else 'low'
    )
    
    # Classify by hashtag presence
    df['has_hashtags'] = df['hashtags'].apply(has_hashtags_check)
    
    # Debug: Check what we're getting
    total_with_hashtags = df['has_hashtags'].sum()
    print(f"DEBUG: Found {total_with_hashtags} tweets with hashtags in pool")
    if total_with_hashtags > 0:
        sample_with_tags = df[df['has_hashtags'] == True].head(2)
        for idx, row in sample_with_tags.iterrows():
            print(f"  Example - Tweet ID: {row['tweet_id']}")
            print(f"           Type: {type(row['hashtags'])}")
            print(f"           Value: {row['hashtags'][:100] if isinstance(row['hashtags'], str) else row['hashtags']}")
    print()
    
    # Show pool composition
    high_count = (df['engagement_level'] == 'high').sum()
    low_count = (df['engagement_level'] == 'low').sum()
    hashtag_count = df['has_hashtags'].sum()
    
    print("Pool composition:")
    print(f"  High engagement (>= {ENGAGEMENT_THRESHOLD} likes): {high_count:,} tweets ({high_count/len(df)*100:.1f}%)")
    print(f"  Low engagement  (<  {ENGAGEMENT_THRESHOLD} likes): {low_count:,} tweets ({low_count/len(df)*100:.1f}%)")
    print(f"  With hashtags: {hashtag_count:,} tweets ({hashtag_count/len(df)*100:.1f}%)")
    print("=" * 70)
    print()
    
    # Ensure at least 20% have hashtags
    min_hashtags = int(n * 0.20)
    
    # Split by hashtag presence
    with_hashtags = df[df['has_hashtags'] == True]
    without_hashtags = df[df['has_hashtags'] == False]
    
    # Sample to ensure minimum hashtags
    if len(with_hashtags) >= min_hashtags:
        sample_with_tags = with_hashtags.sample(n=min_hashtags, random_state=RANDOM_SEED)
        remaining = n - min_hashtags
        
        if len(without_hashtags) >= remaining:
            sample_without_tags = without_hashtags.sample(n=remaining, random_state=RANDOM_SEED)
        else:
            # If not enough without hashtags, sample more with hashtags
            sample_without_tags = without_hashtags
            additional_needed = remaining - len(sample_without_tags)
            # Get additional from hashtag tweets (excluding already sampled)
            remaining_with_tags = with_hashtags.drop(sample_with_tags.index)
            if len(remaining_with_tags) >= additional_needed:
                additional_tags = remaining_with_tags.sample(n=additional_needed, random_state=RANDOM_SEED)
                sample_with_tags = pd.concat([sample_with_tags, additional_tags], ignore_index=True)
        
        result = pd.concat([sample_with_tags, sample_without_tags], ignore_index=True)
    else:
        # Not enough hashtag tweets, take all and fill rest randomly
        sample_with_tags = with_hashtags
        remaining = n - len(sample_with_tags)
        if len(without_hashtags) >= remaining:
            sample_without_tags = without_hashtags.sample(n=remaining, random_state=RANDOM_SEED)
        else:
            sample_without_tags = without_hashtags
        result = pd.concat([sample_with_tags, sample_without_tags], ignore_index=True)
    
    print(f"Sampled {len(result)} tweets randomly")
    
    # Report sample composition
    sample_high = (result['engagement_level'] == 'high').sum()
    sample_low = (result['engagement_level'] == 'low').sum()
    sample_hashtags = result['has_hashtags'].sum()
    
    print()
    print("Sample composition:")
    print(f"  High engagement: {sample_high} tweets ({sample_high/len(result)*100:.1f}%)")
    print(f"  Low engagement:  {sample_low} tweets ({sample_low/len(result)*100:.1f}%)")
    print(f"  With hashtags:   {sample_hashtags} tweets ({sample_hashtags/len(result)*100:.1f}%)")
    
    # Drop the temporary columns
    result = result.drop(columns=['engagement_level', 'has_hashtags'])
    
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
    print(f"ELECTION PERIOD TWEET POOL: {len(df):,} tweets")
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
    sampled = random_sample(df, SAMPLE_SIZE)
    # sampled = stratified_sample(df, SAMPLE_SIZE)
    
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
    df['quote_count'] = df['quote_count'].fillna(0).astype('int32')
    # view_count can have non-numeric values, convert safely
    df['view_count'] = pd.to_numeric(df['view_count'], errors='coerce').fillna(0).astype('int64')
    
    # Column order - only include columns that exist in the dataframe
    ordered_cols = [
        'row_id',
        'date',
        'post_type',
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
        1. Load November part 44, chunks 111-113 using data_files parameter
        2. Stream and collect English tweets from election period
        3. Apply stratified sampling by engagement (30% high, 70% low)
        4. Save 500-tweet sample to parquet
    """
    print()
    print("=" * 70)
    print("NOVEMBER 2024 ELECTION TWEET SAMPLER")
    print("=" * 70)
    print(f"Target sample: {SAMPLE_SIZE} tweets")
    print(f"Source parts: {NOVEMBER_PARTS}")
    print(f"Chunks to load: {CHUNKS_TO_LOAD}")
    print(f"Date range: {DATE_START} to {DATE_END} (election period)")
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
        return
    
    # Process and sample
    sampled_df = process_sample(tweets)
    
    if sampled_df is None:
        return
    
    # Save
    save_parquet(sampled_df)


if __name__ == "__main__":
    main()