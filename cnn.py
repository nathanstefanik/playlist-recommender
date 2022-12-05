model = load_model('models/cnn/best_cnn.hdf5')
data = pd.read_pickle('crafted_arrays/playlists_and_trackid.pkl')
data = data[data['num_tracks_id'] >= 10]
max_length = max(data['num_tracks_id'])
yhat = model.predict(X_test)
print('Best model MSE on test data = ', mse(y_test, yhat).numpy())