import pandas as pd

reddit = pd.read_csv('raw_reddit.csv')
youtube = pd.read_csv('raw_youtube.csv')

combined = pd.concat([reddit, youtube], ignore_index=True)
combined.drop_duplicates(subset='text', inplace=True)
combined = combined[['text', 'source', 'label']]
combined.to_csv('raw_dataset.csv', index=False)

print(f"Reddit: {len(reddit)}")
print(f"YouTube: {len(youtube)}")
print(f"Total after merging: {len(combined)}")
print("Saved to raw_dataset.csv")