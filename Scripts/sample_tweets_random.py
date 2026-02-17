"""
Collect first 50,000 English tweets, then randomly sample 300.
"""

import pandas as pd
from datasets import load_dataset
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_SIZE =500
RANDOM_SEED = 42
OUTPUT_PATH = "data/tweets.parquet"

# Collection settings
MAX_COLLECT = 1000000  # Collect first 50k English tweets
LANGUAGE = "en"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

"""Classify tweet type."""
def derive_post_type(row):
    if row.get('retweetedTweet') is not None:
        return 'retweet'
    elif row.get('quotedTweet') is not None:
        return 'quote'
    elif row.get('in_reply_to_screen_name') is not None:
        return 'reply'
    return 'original'

"""Clean list fields."""
def clean_list_field(field_value):
    if field_value is None:
        return []
    if isinstance(field_value, list):
        return field_value
    return []

# ============================================================================
# STREAMING COLLECTION
# ============================================================================

# Stream dataset and collect first 50,000 English tweets.

def collect_first_n_tweets():
   
    print("STREAMING DATASET")
    print("="*70)
    print(f"Collecting first {MAX_COLLECT:,} English tweets from dataset")
    print("="*70 + "\n")
    
    print("Opening stream to hugging face dataset ")
    dataset = load_dataset(
        "deadbirds/usc-x-24-us-election-parquet",
        split="train",
        streaming=True
    )
    
    collected = []
    total_processed = 0
    skipped_language = 0
    
    print(f"📥 Collecting English tweets...\n")
    
    try:
        for row in dataset:
            total_processed += 1
            
            # Progress
            if total_processed % 5000 == 0:
                print(f"   Processed: {total_processed:,} | Collected English: {len(collected):,}")
            
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
                }
                
                collected.append(tweet)
                
            except Exception as e:
                continue
            
            # Stop when we have enough
            if len(collected) >= MAX_COLLECT:
                print(f"\n✓ Collected {len(collected):,} English tweets")
                break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n⚠️  Stream error: {e}")
    
    print(f"\n{'='*70}")
    print("COLLECTION COMPLETE")
    print("="*70)
    print(f"Total processed: {total_processed:,}")
    print(f"English tweets collected: {len(collected):,}")
    print(f"Non-English skipped: {skipped_language:,}")
    print("="*70 + "\n")
    
    return collected

# ============================================================================
# SAMPLING AND PROCESSING
# ============================================================================


# Randomly sample 300 tweets from the collected pool.
def sample_and_process(tweets_list):
    if len(tweets_list) == 0:
        print("No tweets collected!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(tweets_list)
    
    # Add post_type
    df['post_type'] = df.apply(derive_post_type, axis=1)
    
    print("="*70)
    print(f"COLLECTED POOL: {len(df):,} tweets")
    print("="*70)
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nPost types:")
    print(df['post_type'].value_counts())
    print(f"\nEngagement (likes):")
    print(df['like_count'].describe())
    print("="*70 + "\n")
    
    # Random sample
    print(f"Random Sampling")
    print(f"   Sampling {SAMPLE_SIZE} tweets from {len(df):,} collected tweets")
    print(f"   Random seed: {RANDOM_SEED}\n")
    
    sampled = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED)
    
    # Create row_id
    sampled = sampled.reset_index(drop=True)
    sampled['row_id'] = range(1, len(sampled) + 1)
    
    # Add metadata
    sampled['dataset_split'] = None
    sampled['sampling_strategy'] = 'random'
    sampled['sampled_at'] = datetime.now()
    
    print("="*70)
    print("FINAL SAMPLE")
    print("="*70)
    print(f"Total: {len(sampled)} tweets")
    print(f"Row IDs: 1 to {len(sampled)}")
    print(f"Date range: {sampled['date'].min()} to {sampled['date'].max()}")
    print(f"\nPost type distribution:")
    print(sampled['post_type'].value_counts())
    print(f"\nEngagement distribution:")
    print(sampled['like_count'].describe())
    
    # Hashtag stats
    has_hashtags = sampled['hashtags'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
    print(f"\nHashtag usage:")
    print(f"  Tweets with hashtags: {has_hashtags} ({has_hashtags/len(sampled)*100:.1f}%)")
    print("="*70)
    
    return sampled

def save_parquet(df):
    """Save with annotation-friendly column order."""
    # Data types
    df['row_id'] = df['row_id'].astype('int32')
    df['tweet_id'] = df['tweet_id'].astype('int64')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['like_count'] = df['like_count'].fillna(0).astype('int32')
    df['retweet_count'] = df['retweet_count'].fillna(0).astype('int32')
    df['reply_count'] = df['reply_count'].fillna(0).astype('int32')
    
    # Column order - annotation-friendly
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
    
    print("\n" + "="*70)
    print("SAVED TO PARQUET")
    print("="*70)
    print(f"Location: {OUTPUT_PATH}")
    print(f"Size: {file_size:.1f} KB")
    print(f"Rows: {len(df)}")
    print(f"Row IDs: 1 to {len(df)}")
    print(f"Period: {df['date'].min()} to {df['date'].max()}")
    print(f"Language: {(df['language'] == LANGUAGE).all()}")
    print("="*70)

def main():
    print("\n" + "="*70)
    print("SAMPLER")
    print("="*70)
    print(f"Language: {LANGUAGE}")
    print(f"Seed: {RANDOM_SEED}")
    print("="*70 + "\n")
    
    
    # Collect first 50k English tweets
    tweets = collect_first_n_tweets()
    
    if len(tweets) < SAMPLE_SIZE:
        print(f"\nOnly collected {len(tweets)} tweets (need {SAMPLE_SIZE})")
        return
    
    # Sample 300 randomly
    sampled_df = sample_and_process(tweets)
    
    if sampled_df is None:
        return
    
    # Save
    save_parquet(sampled_df)
    
if __name__ == "__main__":
    main()