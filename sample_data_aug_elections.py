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
    - Balanced sampling across political ideologies (40% Rep, 40% Dem, 20% Neutral)
    - Within each political group: engagement stratification (30% high, 70% low)
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
OUTPUT_PATH = "data/tweets_aug.parquet"

# Dataset info
DATASET_REPO = "deadbirds/usc-x-24-us-election-parquet"

# August 2024 parts to load (pre-election period)
AUGUST_PARTS = [23, 24, 25, 26, 27, 28]

# Load all chunks (stream all august_chunk_*.parquet files)
# No specific chunk filtering - will load all available

# Number of tweets to collect before sampling
MAX_COLLECT = 100000

LANGUAGE = "en"

# Date range filter - Full August 2024
DATE_START = "2024-08-01"
DATE_END = "2024-08-31"

# Stratification settings
ENGAGEMENT_PROPORTIONS = {
    'high': 0.30,
    'low': 0.70
}

ENGAGEMENT_THRESHOLD = 100

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

# Political balance target
POLITICAL_BALANCE = {
    'republican': 0.40,
    'democrat': 0.40,
    'neutral': 0.20  # Tweets that don't match either set of keywords
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


def classify_political_leaning(row):
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
    
    Loads all parquet files from parts 23-28 (August 2024).
    Each part contains different chunk numbers, so we load by part.
    Filters to August 2024 date range.
    
    Returns:
        list: List of tweet dictionaries
    """
    print("LOADING AUGUST 2024 DATA")
    print("=" * 70)
    print(f"Target parts: {AUGUST_PARTS}")
    print(f"Loading all parquet files from August parts")
    print(f"Date filter: {DATE_START} to {DATE_END} (August 2024)")
    print("=" * 70)
    print()
    
    # Try loading by part directories
    # HuggingFace datasets should find all parquet files in these parts
    print("Loading dataset in streaming mode...")
    print("Attempting to load from multiple parts...")
    print()
    
    collected = []
    total_processed = 0
    skipped_language = 0
    skipped_date = 0
    
    # Load each part separately
    for part_num in AUGUST_PARTS:
        print(f"Loading part {part_num}...")
        
        try:
            # Try loading all files from this part
            # The library should find all august_chunk_*.parquet files
            dataset = load_dataset(
                DATASET_REPO,
                data_dir=f"part_{part_num}",
                split="train",
                streaming=True
            )
        except Exception as e:
            print(f"  ERROR loading part {part_num}: {e}")
            continue
        
        print(f"  Part {part_num} loaded, processing tweets...")
        
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
            break
        except Exception as e:
            print()
            print(f"  WARNING: Stream error in part {part_num}: {e}")
            continue
        
        if len(collected) >= MAX_COLLECT:
            break
    
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

def balanced_sample(df, n):
    """
    Perform stratified sampling by both engagement level AND political leaning.
    Ensures balanced representation across political ideologies.
    
    Args:
        df (pd.DataFrame): DataFrame containing tweets
        n (int): Total number of tweets to sample
        
    Returns:
        pd.DataFrame: Balanced stratified sample
    """
    print("POLITICALLY BALANCED STRATIFIED SAMPLING")
    print("=" * 70)
    print(f"Target sample size: {n}")
    print()
    
    # Classify tweets
    df['engagement_level'] = df['like_count'].apply(
        lambda x: 'high' if x >= ENGAGEMENT_THRESHOLD else 'low'
    )
    df['political_leaning'] = df.apply(classify_political_leaning, axis=1)
    df['has_hashtags'] = df['hashtags'].apply(has_hashtags_check)
    
    # Show pool composition
    print("Pool composition:")
    print(f"  Total tweets: {len(df):,}")
    print()
    print("By political leaning:")
    for leaning in ['republican', 'democrat', 'neutral']:
        count = (df['political_leaning'] == leaning).sum()
        pct = count / len(df) * 100
        print(f"  {leaning.capitalize():12} {count:,} tweets ({pct:.1f}%)")
    print()
    print("By engagement:")
    high_count = (df['engagement_level'] == 'high').sum()
    low_count = (df['engagement_level'] == 'low').sum()
    print(f"  High (>={ENGAGEMENT_THRESHOLD} likes): {high_count:,} ({high_count/len(df)*100:.1f}%)")
    print(f"  Low  (<{ENGAGEMENT_THRESHOLD} likes):  {low_count:,} ({low_count/len(df)*100:.1f}%)")
    print("=" * 70)
    print()
    
    # Calculate targets for each combination
    samples = []
    
    for leaning, pol_prop in POLITICAL_BALANCE.items():
        n_political = int(n * pol_prop)
        
        # Further split by engagement
        n_high = int(n_political * ENGAGEMENT_PROPORTIONS['high'])
        n_low = n_political - n_high
        
        print(f"{leaning.capitalize()} target: {n_political} tweets")
        print(f"  High engagement: {n_high}")
        print(f"  Low engagement:  {n_low}")
        
        # Get subsets
        subset = df[df['political_leaning'] == leaning]
        high_eng = subset[subset['engagement_level'] == 'high']
        low_eng = subset[subset['engagement_level'] == 'low']
        
        print(f"  Available: {len(high_eng):,} high, {len(low_eng):,} low")
        
        # Sample from high engagement
        if len(high_eng) >= n_high:
            high_sample = high_eng.sample(n=n_high, random_state=RANDOM_SEED)
        else:
            high_sample = high_eng
            print(f"  WARNING: Only {len(high_eng)} high engagement (target: {n_high})")
        
        # Sample from low engagement
        if len(low_eng) >= n_low:
            low_sample = low_eng.sample(n=n_low, random_state=RANDOM_SEED)
        else:
            low_sample = low_eng
            print(f"  WARNING: Only {len(low_eng)} low engagement (target: {n_low})")
        
        # Combine this political group
        combined = pd.concat([high_sample, low_sample], ignore_index=True)
        samples.append(combined)
        print(f"  Sampled: {len(combined)} tweets")
        print()
    
    # Combine all groups
    result = pd.concat(samples, ignore_index=True)
    
    # Report final composition
    print("=" * 70)
    print("FINAL SAMPLE COMPOSITION")
    print("=" * 70)
    for leaning in ['republican', 'democrat', 'neutral']:
        count = (result['political_leaning'] == leaning).sum()
        pct = count / len(result) * 100
        print(f"  {leaning.capitalize():12} {count} tweets ({pct:.1f}%)")
    
    high_final = (result['engagement_level'] == 'high').sum()
    low_final = (result['engagement_level'] == 'low').sum()
    hashtags_final = result['has_hashtags'].sum()
    
    print()
    print(f"  High engagement: {high_final} ({high_final/len(result)*100:.1f}%)")
    print(f"  Low engagement:  {low_final} ({low_final/len(result)*100:.1f}%)")
    print(f"  With hashtags:   {hashtags_final} ({hashtags_final/len(result)*100:.1f}%)")
    print("=" * 70)
    
    # Clean up temporary columns
    result = result.drop(columns=['engagement_level', 'political_leaning', 'has_hashtags'])
    
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
    sampled = balanced_sample(df, SAMPLE_SIZE)
    
    # Create sequential row IDs
    sampled = sampled.reset_index(drop=True)
    sampled['row_id'] = range(1, len(sampled) + 1)
    
    # Add metadata
    sampled['dataset_split'] = None
    sampled['sampling_strategy'] = 'balanced_political_engagement'
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
        1. Load August 2024 parts (23-28), all chunks
        2. Stream and collect English tweets from August 2024
        3. Apply politically balanced sampling (40% Rep, 40% Dem, 20% Neutral)
           with engagement stratification (30% high, 70% low within each group)
        4. Save 500-tweet sample to parquet
    """
    print()
    print("=" * 70)
    print("AUGUST 2024 POLITICAL TWEET SAMPLER")
    print("=" * 70)
    print(f"Target sample: {SAMPLE_SIZE} tweets")
    print(f"Source parts: {AUGUST_PARTS}")
    print(f"Date range: {DATE_START} to {DATE_END} (August 2024)")
    print(f"Collection pool: Up to {MAX_COLLECT:,} English tweets")
    print(f"Language: {LANGUAGE}")
    print(f"Stratification: Political balance (40/40/20) + Engagement (30/70)")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 70)
    print()
    
    # Load August data
    tweets = load_august_tweets()
    
    if len(tweets) < SAMPLE_SIZE:
        print(f"ERROR: Only {len(tweets)} tweets collected (need {SAMPLE_SIZE})")
        print()
        print("Possible solutions:")
        print(f"1. Increase MAX_COLLECT (currently {MAX_COLLECT:,})")
        print("2. Reduce SAMPLE_SIZE")
        print("3. Expand DATE_START/DATE_END range")
        print("4. Add more parts to AUGUST_PARTS")
        print("5. Check internet connection")
        return
    
    # Process and sample
    sampled_df = process_sample(tweets)
    
    if sampled_df is None:
        return
    
    # Save
    save_parquet(sampled_df)


if __name__ == "__main__":
    main()