"""
Stratified Tweet Sampler for LLM Annotation Study - August 2024 Pre-Election Period

Loads tweets from August 2024 to provide pre-election political discourse
with better ideological balance than the post-election November period.

Data Source:
    - Parts 23-28 of the dataset (August 2024)
    - Uses data_files to load all august_chunk_*.parquet files
    - Filters to August 2024 date range (Aug 1-31)
    - Streams data to avoid memory issues

Strategy:
    - Loads English tweets from August 2024
    - Topic-diversified sampling using keyword category (40% Rep-related, 40% Dem-related, 20% Neutral)
    - Keywords capture political subject matter, not author affiliation
    - Post type is characterized but not used for stratification
    - Produces 500-tweet sample suitable for annotation quality testing
    - Aligns with Jia et al. (2024) methodology: engagement-based stratified sampling

Author: Maria Meshcheryakova
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
RANDOM_SEED = 45
OUTPUT_PATH = "data_aug/tweets_keyword_engagement_1.parquet"

# Dataset info
DATASET_REPO = "deadbirds/usc-x-24-us-election-parquet"

# August 2024 parts to load (pre-election period)
AUGUST_PARTS = [23, 24, 25, 26, 27, 28]

# Load all chunks (stream all august_chunk_*.parquet files)

# Number of tweets to collect before sampling
MAX_COLLECT = 1800000

LANGUAGE = "en"

# Date range filter - Full August 2024
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
    'low':    0.50,   # 50% - reflects majority of organic discourse
    'medium': 0.30,   # 30%
    'high':   0.20,   # 20% - oversampled relative to pool for diversity
}

# Political leaning detection (keyword-based)
# Based on top hashtags from dataset paper + common political terms
REPUBLICAN_KEYWORDS = {
    'hashtags': [
        # Top hashtags from dataset
        'MAGA', 'Trump2024', 'Trump', 'DonaldTrump', 'GOP', 'TRUMPTRAIN',
        'Trump2024ToSaveAmerica', 'FJB', 'RNC', 
        # Additional common Republican hashtags
        'KAG', 'Republican', 'Conservative', 'RedWave', 'SaveAmerica',
        'TRUMPWON', 'StopTheSteal', 'AmericaFirst', 'MAGA2024',
        'TrumpVance2024', 'VoteRed'
    ],
    'text': [
        'trump', 'maga', 'republican', 'gop', 'conservative', 'red wave',
        'donald trump', 'fjb', 'rnc'
    ]
}

DEMOCRAT_KEYWORDS = {
    'hashtags': [
        # Top hashtags from dataset
        'BidenHarris2024', 'Biden', 'Biden2024', 'JoeBiden', 'KamalaHarris',
        'Democrats', 'VoteBlue', 'VoteBlue2024',
        # Additional common Democrat hashtags
        'Harris2024', 'Kamala', 'Democratic', 'BlueWave', 'Liberal', 
        'Progressive', 'DemCast', 'BidenHarris', 'DemVoice1', 'POTUS',
        'VoteBlueToSaveAmerica', 'ProChoice', 'VoteBlue2024ToSaveAmerica'
    ],
    'text': [
        'kamala', 'harris', 'democrat', 'biden', 'liberal', 'blue wave',
        'joe biden', 'kamala harris'
    ]
}

# Keyword category target proportions
# Note: categories reflect political subject matter (keyword presence),
# not the author's political affiliation - e.g. "Trump is a felon" is 
# classified as 'republican' due to keyword match, not author ideology
POLITICAL_BALANCE = {
    'republican': 0.40,  # Tweets containing Republican-related keywords
    'democrat': 0.40,    # Tweets containing Democrat-related keywords
    'neutral': 0.20      # Tweets with no matching political keywords
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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


def classify_keyword_category(row):
    """
    Classify tweet political leaning based on hashtags and text content.
    
    Uses keyword matching as a proxy for political ideology.
    Priority: hashtags > text content
    
    Args:
        row: DataFrame row with 'text' and 'hashtags' fields
        
    Returns:
        str: 'republican', 'democrat', or 'neutral'
    """
    text_lower = str(row.get('text', '')).lower()
    hashtags = row.get('hashtags', [])
    
    # Extract hashtag text
    hashtag_texts = []
    if isinstance(hashtags, list):
        for tag in hashtags:
            if isinstance(tag, dict) and 'text' in tag:
                hashtag_texts.append(tag['text'].lower())
    
    # Check hashtags first (stronger signal)
    rep_hashtag_matches = sum(1 for kw in REPUBLICAN_KEYWORDS['hashtags'] 
                              if kw.lower() in hashtag_texts)
    dem_hashtag_matches = sum(1 for kw in DEMOCRAT_KEYWORDS['hashtags'] 
                              if kw.lower() in hashtag_texts)
    
    if rep_hashtag_matches > dem_hashtag_matches:
        return 'republican'
    elif dem_hashtag_matches > rep_hashtag_matches:
        return 'democrat'
    
    # Check text content as fallback
    rep_text_matches = sum(1 for kw in REPUBLICAN_KEYWORDS['text'] 
                          if kw in text_lower)
    dem_text_matches = sum(1 for kw in DEMOCRAT_KEYWORDS['text'] 
                          if kw in text_lower)
    
    if rep_text_matches > dem_text_matches:
        return 'republican'
    elif dem_text_matches > rep_text_matches:
        return 'democrat'
    
    return 'neutral'


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
# DATA LOADING FROM AUGUST PARTS
# ============================================================================

def load_august_tweets():
    """
    Load tweets from August 2024 parts using datasets library.
    
    Uses data_files parameter to load 6 evenly spaced chunks per part,
    matching the same loading strategy as the November loader.
    Filters to August 2024 date range.
    
    Returns:
        list: List of tweet dictionaries
    """
    print("LOADING AUGUST 2024 DATA")
    print("=" * 70)
    print(f"Target parts: {AUGUST_PARTS}")
    print(f"Loading 6 evenly spaced chunks per part (36 files total)")
    print(f"Date filter: {DATE_START} to {DATE_END} (August 2024)")
    print("=" * 70)
    print()

    # 6 evenly spaced chunks per part
    CHUNKS_PER_PART = {
        23: [1, 4, 8, 12, 16, 20],
        24: [21, 24, 28, 32, 36, 40],
        25: [41, 44, 48, 52, 56, 60],
        26: [61, 64, 68, 72, 76, 80],
        27: [81, 84, 88, 92, 96, 100],
        28: [101, 104, 107, 110, 113, 117],
    }

    # Build list of specific data files to load
    data_files = []
    for part_num, chunks in CHUNKS_PER_PART.items():
        for chunk_num in chunks:
            file_path = f"part_{part_num}/aug_chunk_{chunk_num}.parquet"
            data_files.append(file_path)

    print(f"Files to load ({len(data_files)} total):")
    for part_num, chunks in CHUNKS_PER_PART.items():
        print(f"  part_{part_num}: chunks {chunks}")
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
                print(f"   Processed: {total_processed:,} | Collected: {len(collected):,} English tweets in August")

            # Language filter
            tweet_lang = row.get('lang', '')
            if tweet_lang != LANGUAGE:
                skipped_language += 1
                continue

            # Date filter - August 2024
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
                print(f"Target reached: {len(collected):,} English tweets in August")
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
    print(f"English tweets in August: {len(collected):,}")
    print(f"Skipped (language): {skipped_language:,}")
    print(f"Skipped (date): {skipped_date:,}")
    print("=" * 70)
    print()

    return collected

# ============================================================================
# BALANCED SAMPLING
# ============================================================================

def classify_engagement(like_count):
    """
    Classify tweet into low, medium, or high engagement tier.
    
    Args:
        like_count (int): Number of likes
        
    Returns:
        str: 'low', 'medium', or 'high'
    """
    if like_count <= 50:
        return 'low'
    elif like_count <= 500:
        return 'medium'
    else:
        return 'high'


def balanced_sample(df, n):
    """
    Perform stratified sampling by keyword category AND engagement tier.
    
    Keyword categories: republican, democrat, neutral (40/40/20)
    Engagement tiers:   low (0-50), medium (51-500), high (500+) (50/30/20)
    
    Args:
        df (pd.DataFrame): DataFrame containing tweets
        n (int): Total number of tweets to sample
        
    Returns:
        pd.DataFrame: Balanced stratified sample
    """
    print("KEYWORD-BALANCED STRATIFIED SAMPLING")
    print("=" * 70)
    print(f"Target sample size: {n}")
    print()
    
    # Classify tweets
    df['engagement_level'] = df['like_count'].apply(classify_engagement)
    df['keyword_category'] = df.apply(classify_keyword_category, axis=1)
    df['has_hashtags'] = df['hashtags'].apply(has_hashtags_check)
    
    # Show pool composition
    print("Pool composition:")
    print(f"  Total tweets: {len(df):,}")
    print()
    print("By keyword category (subject matter, not affiliation):")
    for cat in ['republican', 'democrat', 'neutral']:
        count = (df['keyword_category'] == cat).sum()
        pct = count / len(df) * 100
        print(f"  {cat.capitalize():12} {count:,} tweets ({pct:.1f}%)")
    print()
    print("By engagement tier:")
    for tier in ['low', 'medium', 'high']:
        lo, hi = ENGAGEMENT_TIERS[tier]
        count = (df['engagement_level'] == tier).sum()
        pct = count / len(df) * 100
        hi_str = str(hi) if hi else '+'
        print(f"  {tier.capitalize():8} ({lo}-{hi_str} likes): {count:,} ({pct:.1f}%)")
    print("=" * 70)
    print()
    
    samples = []
    
    for cat, pol_prop in POLITICAL_BALANCE.items():
        n_cat = int(n * pol_prop)
        subset = df[df['keyword_category'] == cat]
        
        print(f"{cat.capitalize()} target: {n_cat} tweets")
        
        cat_samples = []
        for tier, eng_prop in ENGAGEMENT_PROPORTIONS.items():
            n_tier = int(n_cat * eng_prop)
            tier_subset = subset[subset['engagement_level'] == tier]
            
            print(f"  {tier.capitalize():8} engagement target: {n_tier:3d} | available: {len(tier_subset):,}")
            
            if len(tier_subset) >= n_tier:
                sampled_tier = tier_subset.sample(n=n_tier, random_state=RANDOM_SEED)
            else:
                sampled_tier = tier_subset
                print(f"  WARNING: Only {len(tier_subset)} {tier} engagement tweets (target: {n_tier})")
            
            cat_samples.append(sampled_tier)
        
        combined = pd.concat(cat_samples, ignore_index=True)
        samples.append(combined)
        print(f"  Sampled: {len(combined)} tweets")
        print()
    
    # Combine all groups
    result = pd.concat(samples, ignore_index=True)
    
    # Report final composition
    print("=" * 70)
    print("FINAL SAMPLE COMPOSITION")
    print("=" * 70)
    for cat in ['republican', 'democrat', 'neutral']:
        count = (result['keyword_category'] == cat).sum()
        pct = count / len(result) * 100
        print(f"  {cat.capitalize():12} {count} tweets ({pct:.1f}%)")
    print()
    for tier in ['low', 'medium', 'high']:
        count = (result['engagement_level'] == tier).sum()
        pct = count / len(result) * 100
        print(f"  {tier.capitalize():8} engagement: {count} ({pct:.1f}%)")
    hashtags_final = result['has_hashtags'].sum()
    print()
    print(f"  With hashtags: {hashtags_final} ({hashtags_final/len(result)*100:.1f}%)")
    print("=" * 70)
    
    # Clean up temporary columns
    result = result.drop(columns=['engagement_level', 'keyword_category', 'has_hashtags'])
    # result = result.drop(columns=['engagement_level', 'keyword_category'])

    
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
    print(f"AUGUST 2024 TWEET POOL: {len(df):,} tweets")
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
    
    # Perform balanced sampling
    # Note: balanced_sample adds keyword_category column temporarily but drops it
    # We need to preserve it, so we'll add it back
    df_with_leaning = df.copy()
    df_with_leaning['keyword_category'] = df_with_leaning.apply(classify_keyword_category, axis=1)
    
    sampled = balanced_sample(df_with_leaning, SAMPLE_SIZE)
    
    # Re-add keyword_category column to sampled data (it was dropped in balanced_sample)
    sampled['keyword_category'] = sampled.apply(classify_keyword_category, axis=1)
    
    # Create sequential row IDs
    sampled = sampled.reset_index(drop=True)
    sampled['row_id'] = range(1, len(sampled) + 1)
    
    # Add metadata
    sampled['dataset_split'] = None
    sampled['sampling_strategy'] = 'topic_diversified'
    sampled['sampled_at'] = datetime.now()
    
    print()
    print("=" * 70)
    print("FINAL SAMPLE")
    print("=" * 70)
    print(f"Total: {len(sampled)} tweets")
    print(f"Row IDs: 1 to {len(sampled)}")
    print(f"Date range: {sampled['date'].min()} to {sampled['date'].max()}")
    print()
    print("Post type distribution:")
    print(sampled['post_type'].value_counts())
    print()
    print("Political leaning distribution:")
    print(sampled['keyword_category'].value_counts())
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
        'keyword_category',
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
    print()
    print("=" * 70)
    print("AUGUST 2024 POLITICAL TWEET SAMPLER")
    print("=" * 70)
    print(f"Target sample: {SAMPLE_SIZE} tweets")
    print(f"Source parts: {AUGUST_PARTS}")
    print(f"Date range: {DATE_START} to {DATE_END} (August 2024)")
    print(f"Collection pool: Up to {MAX_COLLECT:,} English tweets")
    print(f"Language: {LANGUAGE}")
    print(f"Stratification: Keyword balance (40/40/20) + Engagement low/med/high (50/30/20)")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 70)
    print()
    
    # Load August data
    tweets = load_august_tweets()
    
    if len(tweets) < SAMPLE_SIZE:
        print(f"ERROR: Only {len(tweets)} tweets collected (need {SAMPLE_SIZE})")
        return
    
    # Process and sample
    sampled_df = process_sample(tweets)
    
    if sampled_df is None:
        return
    
    # Save
    save_parquet(sampled_df)


if __name__ == "__main__":
    main()