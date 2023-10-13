import numpy as np
import scipy as sp
import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

entropy_songs_data_path =  'data/entropy/songs_post_encoding_songs.pickle'
entropy_shuffle_data_path =  'data/entropy/shuffle_post_encoding.pickle'
entropy_random_data_path =  'data/entropy/random_post_encoding.pickle'

with open(entropy_songs_data_path, 'rb') as handle:
    entropy_songs_data = pickle.load(handle)
with open(entropy_shuffle_data_path, 'rb') as handle:
    entropy_shuffle_data = pickle.load(handle)
with open(entropy_random_data_path, 'rb') as handle:
    entropy_random_data = pickle.load(handle)

min_songs = np.min(entropy_songs_data, axis=1)
min_shuffle = np.min(entropy_shuffle_data, axis=1)
min_random = np.min(entropy_random_data, axis=1)

songs_min_adjusted = entropy_songs_data - min_songs[:,np.newaxis]
shuffle_min_adjusted = entropy_shuffle_data - min_shuffle[:,np.newaxis]
random_min_adjusted = entropy_random_data - min_random[:,np.newaxis]

sum_songs = np.sum(songs_min_adjusted, axis=1)
sum_shuffle = np.sum(shuffle_min_adjusted, axis=1)
sum_random = np.sum(random_min_adjusted, axis=1)

songs_distributions = songs_min_adjusted/sum_songs[:,np.newaxis]
shuffle_distributions = shuffle_min_adjusted/sum_songs[:,np.newaxis]
random_distributions = random_min_adjusted/sum_songs[:,np.newaxis]

entropy_songs = sp.stats.entropy(songs_distributions, axis=1)
entropy_shuffle = sp.stats.entropy(shuffle_distributions, axis=1)
entropy_random = sp.stats.entropy(random_distributions, axis=1)

mean_songs = np.mean(entropy_songs)
mean_shuffle = np.mean(entropy_shuffle)
mean_random = np.mean(entropy_random)
std_songs = np.std(entropy_songs)
std_shuffle = np.std(entropy_shuffle)
std_random = np.std(entropy_random)
median_songs = np.median(entropy_songs)
median_shuffle = np.median(entropy_shuffle)
median_random = np.median(entropy_random)

min_songs = np.min(entropy_songs)
min_shuffle = np.min(entropy_shuffle)
min_random = np.min(entropy_random)
max_songs = np.max(entropy_songs)
max_shuffle = np.max(entropy_shuffle)
max_random = np.max(entropy_random)

print('songs:')
print(mean_songs, std_songs, median_songs)
print(min_songs, max_songs)

print('shuffle:')
print(mean_shuffle, std_shuffle, median_shuffle)
print(min_shuffle, max_shuffle)

print('random:')
print(mean_random, std_random, median_random)
print(min_random, max_random)