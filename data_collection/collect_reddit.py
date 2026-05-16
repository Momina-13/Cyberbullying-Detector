import requests
import pandas as pd
import time

headers = {'User-Agent': 'cyberbullying_collector/0.1'}

subreddits = ['teenagers', 'roastme', 'AmItheAsshole', 'confession', 'unpopularopinion']

def scrape_subreddit(subreddit, limit=200):
    comments = []
    url = f"https://www.reddit.com/r/{subreddit}/comments.json?limit=100"
    
    after = None
    while len(comments) < limit:
        if after:
            response = requests.get(url + f"&after={after}", headers=headers)
        else:
            response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error on r/{subreddit}: {response.status_code}")
            break
        
        data = response.json()
        children = data['data']['children']
        
        if not children:
            break
        
        for child in children:
            body = child['data'].get('body', '')
            if body and body != '[deleted]' and body != '[removed]' and len(body) > 10:
                comments.append({'text': body, 'source': subreddit, 'label': ''})
        
        after = data['data'].get('after')
        if not after:
            break
        
        time.sleep(2)
        print(f"r/{subreddit}: collected {len(comments)} so far...")
    
    return comments[:limit]

all_data = []
for sub in subreddits:
    print(f"\nScraping r/{sub}...")
    result = scrape_subreddit(sub, limit=200)
    all_data.extend(result)
    print(f"Got {len(result)} comments from r/{sub}")
    time.sleep(3)

df = pd.DataFrame(all_data)
df.to_csv('raw_reddit.csv', index=False)
print(f"\nDone! Total comments collected: {len(df)}")
print("Saved to raw_reddit.csv")