import requests
import pandas as pd
import time

API_KEY = "AIzaSyDO1NSM2gGL2W3AkPRCVcDsMRKnq75is4I"

# Search queries to find relevant videos
search_queries = [
    "cyberbullying stories",
    "school bullying",
    "online harassment",
    "toxic comments",
    "hate speech examples"
]

def search_videos(query, max_results=5):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'maxResults': max_results,
        'key': API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    video_ids = []
    if 'items' in data:
        for item in data['items']:
            video_ids.append(item['id']['videoId'])
    return video_ids

def get_comments(video_id, max_comments=50):
    comments = []
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'maxResults': 100,
        'key': API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'items' not in data:
        return comments
    
    for item in data['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        if len(comment) > 10:
            comments.append({'text': comment, 'source': 'youtube', 'label': ''})
    
    return comments[:max_comments]

all_comments = []

for query in search_queries:
    print(f"\nSearching videos for: '{query}'")
    video_ids = search_videos(query, max_results=5)
    print(f"Found {len(video_ids)} videos")
    
    for vid_id in video_ids:
        comments = get_comments(vid_id)
        all_comments.extend(comments)
        print(f"  Video {vid_id}: got {len(comments)} comments (total so far: {len(all_comments)})")
        time.sleep(1)

df = pd.DataFrame(all_comments)
df.drop_duplicates(subset='text', inplace=True)
df.to_csv('raw_youtube.csv', index=False)
print(f"\nDone! Total comments: {len(df)}")
print("Saved to raw_youtube.csv")