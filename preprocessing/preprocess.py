import pandas as pd
import re
import csv

# ── 1. Load ──────────────────────────────────────────────
df = pd.read_csv('final_dataset.csv')
print(f"Loaded: {len(df)} rows")

# ── 2. Lowercase ─────────────────────────────────────────
df['text'] = df['text'].apply(lambda x: str(x).lower())

# ── 3. Remove URLs ───────────────────────────────────────
df['text'] = df['text'].apply(lambda x: re.sub(r'http\S+|www\S+', '', x))

# ── 4. Remove mentions and hashtags ──────────────────────
df['text'] = df['text'].apply(lambda x: re.sub(r'@\w+|#\w+', '', x))

# ── 5. Remove special characters (keep letters, numbers, spaces) ──
df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))

# ── 6. Handle emojis (already removed by step 5, but clean extra spaces) ──
df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

# ── 7. Custom tokenizer (NO NLTK — pure Python) ──────────
def tokenize(text):
    return text.split()

# ── 8. Stopword removal (manual list — NO NLTK) ──────────
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your',
    'yours','yourself','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs',
    'what','which','who','whom','this','that','these','those','am',
    'is','are','was','were','be','been','being','have','has','had',
    'having','do','does','did','doing','a','an','the','and','but',
    'if','or','because','as','until','while','of','at','by','for',
    'with','about','against','between','into','through','during',
    'before','after','above','below','to','from','up','down','in',
    'out','on','off','over','under','again','further','then','once',
    'here','there','when','where','why','how','all','both','each',
    'few','more','most','other','some','such','no','nor','not',
    'only','own','same','so','than','too','very','s','t','can',
    'will','just','don','should','now','d','ll','m','o','re','ve',
    'y','ain','aren','couldn','didn','doesn','hadn','hasn','haven',
    'isn','ma','mightn','mustn','needn','shan','shouldn','wasn',
    'weren','won','wouldn'
}

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

# ── 9. Apply tokenization and stopword removal ────────────
df['tokens'] = df['text'].apply(tokenize)
df['tokens'] = df['tokens'].apply(remove_stopwords)

# ── 10. Remove empty rows after processing ────────────────
df = df[df['tokens'].apply(len) > 0]

# ── 11. Rejoin tokens back to cleaned text ────────────────
df['text'] = df['tokens'].apply(lambda x: ' '.join(x))
df = df[['text', 'label']]
df = df[df['text'].str.strip() != '']
df = df.dropna()

print(f"After preprocessing: {len(df)} rows")

# ── 12. Train / Val / Test split (70 / 15 / 15) ──────────
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)

train = df[:train_end]
val   = df[train_end:val_end]
test  = df[val_end:]

train.to_csv('train.csv', index=False)
val.to_csv('val.csv',   index=False)
test.to_csv('test.csv', index=False)

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print("Saved: train.csv, val.csv, test.csv")