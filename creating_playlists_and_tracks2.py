import numpy as np
import pandas as pd
import re
import json

tracks_features_processed = pd.read_pickle('tracks_features_processed.pkl')
track_identifier = pd.read_pickle('track_identifier.pkl')
start = 4000
end = 7999
print(f'---- we are extracting data starting from playlist {start} to {end} ----')

curr = start
df = pd.DataFrame({})
while (curr < end):
    if curr == start:
        with open(f'spotify_million_playlist_dataset/data/mpd.slice.{curr}-{curr+999}.json','r') as f:
            data = json.loads(f.read())
        df = pd.json_normalize(data, record_path =['playlists'])
        df = df[['pid','name','tracks','num_tracks']]
    else:
        with open(f'spotify_million_playlist_dataset/data/mpd.slice.{curr}-{curr+999}.json','r') as f:
            data1 = json.loads(f.read())
        df1 = pd.json_normalize(data1, record_path =['playlists'])
        df1 = df1[['pid','name','tracks','num_tracks']]
        df = pd.concat([df, df1])
    curr += 1000

def convert_to_tracks(tracks_dict):
    tracks = []
    for track in tracks_dict:
        curr_artist = track['artist_name']
        curr_artist = re.sub(r'[^A-Za-z0-9 ]+', '', curr_artist)
        curr_artist = re.sub(r'\s+', '', curr_artist)
        curr_album = track['album_name']
        curr_album = re.sub(r'[^A-Za-z0-9 ]+', '', curr_album)
        curr_album = re.sub(r'\s+', '', curr_album)
        curr_duration = track['duration_ms']
        curr_artist = curr_artist.lower()
        curr_album = curr_album.lower()
        tmp_df = track_identifier[track_identifier['album'].str.contains(curr_album)].copy()
        tmp_df = tmp_df[tmp_df['duration_ms'] < curr_duration + 100].copy()
        tmp_df = tmp_df[tmp_df['duration_ms'] > curr_duration - 100].copy()
        if len(list(tmp_df['id'])) > 0:
            tracks.append(list(tmp_df['id'])[0])
    return tracks

print('Converting tracks now!')
length = df.shape[0]
for i in range(length):
    if i % 100 == 0:
        df[:i].to_pickle(f'crafted_arrays/playlists_and_trackid_start{start}.pkl')
        df['num_tracks_id'] = df['tracks'].apply(lambda row: len(row))
        print(f'--- saved from playlist {start} to {i+start} so far! ---')
    if i % 25 == 0:
        print(f"iter: {i}")
    df['tracks'].iloc[i] = convert_to_tracks(df['tracks'].iloc[i].copy())

df['num_tracks_id'] = df['tracks'].apply(lambda row: len(row))
df[:length].to_pickle(f'crafted_arrays/playlists_and_trackid_start{start}.pkl')
print('------ DONE -------')