# playlist-recommender

Training various models to recommend playlists, with a focus on danceability where applicable. Used spotify millions dataset 
joined with another dataset to associate song features. Notably, **we use the last id'ed song as the ground truth song to recommend**.

## Dataset

I consider the following song features:

'id', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 
'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'

## Models
1. Linear Regression
2. Lasso Regression
3. CNN
4. Random Forest

## Metrics

