import pandas as pd
import re

df = pd.read_excel('demo/tweets.xlsx')
df.columns = ['tweet']

def extract_tweet(raw):
    lines = str(raw).strip().split('\n')
    # remove empty lines
    lines = [l.strip() for l in lines if l.strip()]
    # drop lines that are username, timestamp, replying to, display name
    cleaned = []
    for line in lines:
        if line.startswith('@'):
            continue
        if line == '·':
            continue
        if line.lower().startswith('replying to'):
            continue
        if re.match(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d+$', line.lower()):
            continue
        if re.match(r'^\d+m$', line.lower()):
            continue
        cleaned.append(line)
    # the actual tweet is everything after the metadata
    # skip the first line which is always the display name
    return ' '.join(cleaned[1:]) if len(cleaned) > 1 else ' '.join(cleaned)

df['tweet'] = df['tweet'].apply(extract_tweet)
print(df.head(5))
df.to_excel('demo/tweets.xlsx', index=False)