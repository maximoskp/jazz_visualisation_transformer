import os

import keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras_nlp
import pandas as pd
from copy import deepcopy

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm


history_path = 'data/model_mask_history_log.csv'
initial_epoch = 0
history = None
if os.path.isfile( history_path ):
    print('history exists')
    history = pd.read_csv(history_path)
    initial_epoch = history['epoch'].iloc[-1]
else:
    print('starting fresh')

checkpoint_path = 'models/mask_1/model'
songs_excel_path = '1058_songs_without_melodies.xlsx'
augmented_excel_path = 'data/augmented_excel.csv'
music_vocab_list_path = 'data/music_vocab_list.pickle'
tokens_path = 'data/tokens.pickle'

print('reading augmented excel, tokens and vocabulary')
augmented_excel = pd.read_csv(augmented_excel_path)
useful_excel = augmented_excel[ augmented_excel['is_original_tonality'] ]
tokens = None
with open(tokens_path, 'rb') as handle:
    tokens = pickle.load(handle)

max_size_tokens = -1
for t in tokens:
    if max_size_tokens < len(t):
        max_size_tokens = len(t)

music_vocab_list = None
with open(music_vocab_list_path, 'rb') as handle:
    music_vocab_list = pickle.load(handle)

useful_tokens = []
for i,b in enumerate(augmented_excel['is_original_tonality']):
    if b:
        useful_tokens.append( tokens[i] )
useful_excel['tokens'] = useful_tokens

# Preprocessing params.
PRETRAINING_BATCH_SIZE = 16
FINETUNING_BATCH_SIZE = 16
SEQ_LENGTH = max_size_tokens #
MASK_RATE = 0.25
PREDICTIONS_PER_SEQ = 32

# Model params.
NUM_LAYERS = 8
MODEL_DIM = 256
INTERMEDIATE_DIM = 256
NUM_HEADS = 8
DROPOUT = 0.3
EMBED_DIM = 256
NORM_EPSILON = 1e-5

MUSIC_VOCAB_SIZE = len( music_vocab_list )
MAX_SEQUENCE_LENGTH = max_size_tokens

# Training params.
PRETRAINING_LEARNING_RATE = 5e-5 # was 5e-4
PRETRAINING_EPOCHS = 5000
FINETUNING_LEARNING_RATE = 5e-5
FINETUNING_EPOCHS = 3

BATCH_SIZE = 16

print('padding to sequence length')
music_texts_SEQ_LENGTH = []
for s in useful_tokens:
    residual_length = SEQ_LENGTH - len( s )
    music_texts_SEQ_LENGTH.append( s + residual_length*['[PAD]'] )

print('making vocabulary')
music_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=music_vocab_list,
    sequence_length=SEQ_LENGTH,
    lowercase=False,
    strip_accents=False,
    split=False
)

def get_dictionary_and_idxs_from_pd_column( c ):
    c_list = c.to_list()
    c_dict = list(np.unique(c_list))
    c_idx = [ c_dict.index(s) for s in c_list ]
    return c_dict, c_idx

def get_idxs_from_pd_column_and_dictionary( c , d ):
    c_list = c.to_list()
    c_idx = [ d.index(s) for s in c_list ]
    return c_idx

print('making target vectors')
# normalized year data
year = useful_excel['composition_date'].to_numpy()
year_min = np.min(year)
year_max = np.max(year)
year_norm = list( (year - year_min)/(year_max - year_min) )
# style categories
harmonic_style_dict, _ = get_dictionary_and_idxs_from_pd_column( augmented_excel['harmonic_style'] )
harmonic_style_idx = get_idxs_from_pd_column_and_dictionary( useful_excel['harmonic_style'] , harmonic_style_dict )
# style categories
form_dict, _ = get_dictionary_and_idxs_from_pd_column( augmented_excel['form'] )
form_idx = get_idxs_from_pd_column_and_dictionary( useful_excel['form'] , form_dict )
# tonality categories
tonality_dict, _ = get_dictionary_and_idxs_from_pd_column( augmented_excel['tonality'] )
tonality_idx = get_idxs_from_pd_column_and_dictionary( useful_excel['tonality'] , tonality_dict )
# composer categories
composer_dict, _ = get_dictionary_and_idxs_from_pd_column( augmented_excel['composer'] )
composer_idx = get_idxs_from_pd_column_and_dictionary( useful_excel['composer'] , composer_dict )
# genre categories
genre_dict, _ = get_dictionary_and_idxs_from_pd_column( augmented_excel['genre_style'] )
genre_idx = get_idxs_from_pd_column_and_dictionary( useful_excel['genre_style'] , genre_dict )

print('making dataset')
music_train_ds = tf.data.Dataset.from_tensor_slices( (music_texts_SEQ_LENGTH, year_norm, harmonic_style_idx, form_idx, tonality_idx, composer_idx, genre_idx) )
music_train_ds = music_train_ds.batch( BATCH_SIZE )

print('making masker and data preprocessing')
masker = keras_nlp.layers.MaskedLMMaskGenerator(
    vocabulary_size=music_tokenizer.vocabulary_size(),
    mask_selection_rate=MASK_RATE,
    mask_selection_length=PREDICTIONS_PER_SEQ,
    mask_token_id=music_tokenizer.token_to_id("[MASK]"),
)

# Create the preprosessing function that performs masking on the fly
def preprocess(inputs, year, style, form, tonality, composer, genre):
    inputs = music_tokenizer( inputs )
    outputs = masker(inputs)
    # Split the masking layer outputs into a (features, labels, and weights)
    # tuple that we can use with keras.Model.fit().
    features = {
        "token_ids": outputs["token_ids"],
        "mask_positions": outputs["mask_positions"],
    }
    labels = outputs["mask_ids"]
    # weights = outputs["mask_weights"]
    # return features, (year, style, form, labels), weights
    return features, (year, style, form, tonality, composer, genre, labels)

processed_ds = music_train_ds.map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Create transformer encoder model
inputs = keras.Input(shape=(SEQ_LENGTH,), dtype=tf.int32)

# Embed our tokens with a positional embedding.
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=music_tokenizer.vocabulary_size(),
    sequence_length=SEQ_LENGTH,
    embedding_dim=MODEL_DIM,
)
outputs = embedding_layer(inputs)

print('creating transformer')
# Apply layer normalization and dropout to the embedding.
outputs = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)(outputs)
outputs = keras.layers.Dropout(rate=DROPOUT)(outputs)

# Add a number of encoder blocks
for i in range(NUM_LAYERS):
    outputs = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=INTERMEDIATE_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        layer_norm_epsilon=NORM_EPSILON,
    )(outputs)

encoder_model = keras.Model(inputs, outputs)
# encoder_model.summary()

tf.keras.utils.plot_model(encoder_model, show_shapes=True, expand_nested=True, to_file='figs/transformer_base.png')

print('creating model')
# Create the pretraining model by attaching a masked language model head.
inputs = {
    "token_ids": keras.Input(shape=(SEQ_LENGTH,), dtype=tf.int32),
    "mask_positions": keras.Input(shape=(PREDICTIONS_PER_SEQ,), dtype=tf.int32),
}

# Encode the tokens.
encoded_tokens = encoder_model(inputs["token_ids"])

post_encoding_reduction = keras.layers.GlobalAveragePooling1D()(encoded_tokens)

# post_encoding_reduction = keras.layers.Reshape([SEQ_LENGTH*EMBED_DIM])(encoded_tokens)
# post_encoding_reduction = keras.layers.Dense(256, activation='relu')(post_encoding_reduction)

year_output1 = keras.layers.Dense(256, activation='relu')(post_encoding_reduction)
year_output1 = keras.layers.Dense(512, activation='relu')(year_output1)
year_output1 = keras.layers.Dense(256, activation='relu')(year_output1)
year_output1 = keras.layers.Dense(64, activation='relu', name='year_descriptor')(year_output1)
year_output = keras.layers.Dense(1, activation='sigmoid', name='year_predictor')(year_output1)

style_output1 = keras.layers.Dense(256, activation='relu')(post_encoding_reduction)
style_output1 = keras.layers.Dense(256, activation='relu')(style_output1)
style_output1 = keras.layers.Dense(64, activation='relu', name='style_descriptor')(style_output1)
style_output = keras.layers.Dense(len(harmonic_style_dict), activation='softmax', name='style_predictor')(style_output1)

form_output1 = keras.layers.Dense(256, activation='relu')(post_encoding_reduction)
form_output1 = keras.layers.Dense(256, activation='relu')(form_output1)
form_output1 = keras.layers.Dense(64, activation='relu', name='form_descriptor')(form_output1)
form_output = keras.layers.Dense(len(form_dict), activation='softmax', name='form_predictor')(form_output1)

tonality_output1 = keras.layers.Dense(256, activation='relu')(post_encoding_reduction)
tonality_output1 = keras.layers.Dense(256, activation='relu')(tonality_output1)
tonality_output1 = keras.layers.Dense(64, activation='relu', name='tonality_descriptor')(tonality_output1)
tonality_output = keras.layers.Dense(len(tonality_dict), activation='softmax', name='tonality_predictor')(tonality_output1)

composer_output1 = keras.layers.Dense(512, activation='relu')(post_encoding_reduction)
composer_output1 = keras.layers.Dense(512, activation='relu')(composer_output1)
composer_output1 = keras.layers.Dense(256, activation='relu')(composer_output1)
composer_output1 = keras.layers.Dense(64, activation='relu', name='composer_descriptor')(composer_output1)
composer_output = keras.layers.Dense(len(composer_dict), activation='softmax', name='composer_predictor')(composer_output1)

genre_output1 = keras.layers.Dense(256, activation='relu')(post_encoding_reduction)
genre_output1 = keras.layers.Dense(256, activation='relu')(genre_output1)
genre_output1 = keras.layers.Dense(64, activation='relu', name='genre_descriptor')(genre_output1)
genre_output = keras.layers.Dense(len(genre_dict), activation='softmax', name='genre_predictor')(genre_output1)

def print_summary(s):
    with open('data/modelsummary.txt','a') as f:
        print(s, file=f)

# mask output
mask_output = keras_nlp.layers.MaskedLMHead(
    embedding_weights=embedding_layer.token_embedding.embeddings,
    activation="softmax",
    name='mask_predictor'
)(encoded_tokens, mask_positions=inputs["mask_positions"])

model = keras.Model(inputs, outputs=[year_output, style_output, form_output, tonality_output, composer_output, genre_output, mask_output])

model_year = keras.Model(inputs, outputs=year_output1)
model_style = keras.Model(inputs, outputs=style_output1)
model_form = keras.Model(inputs, outputs=form_output1)
model_tonality = keras.Model(inputs, outputs=tonality_output1)
model_composer = keras.Model(inputs, outputs=composer_output1)
model_genre = keras.Model(inputs, outputs=genre_output1)

model.compile(
    loss=["mean_squared_error", "sparse_categorical_crossentropy", "sparse_categorical_crossentropy", "sparse_categorical_crossentropy", "sparse_categorical_crossentropy", "sparse_categorical_crossentropy", "sparse_categorical_crossentropy"],
    optimizer=keras.optimizers.experimental.AdamW(PRETRAINING_LEARNING_RATE),
    weighted_metrics=["sparse_categorical_accuracy"],
    jit_compile=True,
)
model.summary(print_fn=print_summary)

tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, to_file='figs/model_expanded.png')
tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=False, to_file='figs/model_compressed.png')

print('initializing checkpoint and logger')
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

print('loading weights')
model.load_weights(latest)

print('predicting ds')
y = model.predict( processed_ds )

# sort years and keep indexes
sort_year_norm_idx = np.argsort( year_norm )

print('plotting year - stats')
plt.clf()
plt.plot(np.array(year_norm)[sort_year_norm_idx], 'b')
plt.plot(y[0][sort_year_norm_idx], 'rx')
plt.savefig('figs/year_sort_plot.png', dpi=300)

print('plotting harmonic style - stats')
with open('figs/stats_mask.txt', 'w') as f:
    print( 'harmonic style accuracy: ' + str(np.sum( np.array(harmonic_style_idx) == np.argmax(y[1], axis=1) ) / len(harmonic_style_idx)), file=f )
plt.clf()
plt.plot(harmonic_style_idx, 'bo')
plt.plot(np.argmax(y[1], axis=1), 'rx', alpha=0.5)
plt.savefig('figs/harmonic_style_accuracy.png', dpi=300)

print('plotting form - stats')
with open('figs/stats_mask.txt', 'a') as f:
    print( '\n' + 'form accuracy: ' + str(np.sum( np.array(form_idx) == np.argmax(y[2], axis=1) ) / len(form_idx)), file=f )
plt.clf()
plt.plot(form_idx, 'bo')
plt.plot(np.argmax(y[2], axis=1), 'rx')
plt.savefig('figs/harmonic_style_accuracy.png', dpi=300)

print('plotting tonality - stats')
with open('figs/stats_mask.txt', 'a') as f:
    print( '\n' + 'tonality accuracy: ' + str(np.sum( np.array(tonality_idx) == np.argmax(y[3], axis=1) ) / len(tonality_idx)), file=f )
plt.clf()
plt.plot(tonality_idx, 'bo')
plt.plot(np.argmax(y[3], axis=1), 'rx')
plt.savefig('figs/tonality_idx_accuracy.png', dpi=300)

# sort composers and keep indexes
sort_composers_norm_idx = np.argsort( composer_idx )
print('plotting composer - stats')
with open('figs/stats_mask.txt', 'a') as f:
    print( '\n' + 'composer accuracy: ' + str(np.sum( np.array(composer_idx) == np.argmax(y[4], axis=1) ) / len(composer_idx)), file=f )
plt.clf()
plt.plot(np.array(composer_idx)[sort_composers_norm_idx], 'bo')
plt.plot(np.argmax(y[4][sort_composers_norm_idx], axis=1), 'rx')
plt.savefig('figs/composer_idx_accuracy.png', dpi=300)

print('plotting genre - stats')
with open('figs/stats_mask.txt', 'a') as f:
    print( '\n' + 'genre accuracy: ' + str(np.sum( np.array(genre_idx) == np.argmax(y[5], axis=1) ) / len(genre_idx)), file=f )
plt.clf()
plt.plot(genre_idx, 'bo')
plt.plot(np.argmax(y[5], axis=1), 'rx')
plt.savefig('figs/genre_idx_accuracy.png', dpi=300)

print('year predict - TSNE plot')
y_year = model_year.predict( processed_ds )
year_embedded = TSNE(n_components=2).fit_transform(y_year)
plt.clf()
year_cols = cm.rainbow(year_norm)
for i in range(year_embedded.shape[0]):
    plt.plot(year_embedded[i,0], year_embedded[i,1],'x', c=year_cols[i])
plt.savefig('figs/year_TSNE.png', dpi=300)

print('harmonic style predict - TSNE plot')
y_harmonic_style = model_style.predict( processed_ds )
harmonic_style_embedded = TSNE(n_components=2).fit_transform(y_harmonic_style)
plt.clf()
harmonic_style_cols = cm.rainbow(harmonic_style_idx/np.max(harmonic_style_idx))
for i in range(harmonic_style_embedded.shape[0]):
    plt.plot(harmonic_style_embedded[i,0], harmonic_style_embedded[i,1],'x', c=harmonic_style_cols[i])
plt.savefig('figs/harmonic_style_TSNE.png', dpi=300)

print('form predict - TSNE plot')
y_form = model_form.predict( processed_ds )
form_embedded = TSNE(n_components=2).fit_transform(y_form)
plt.clf()
form_cols = cm.rainbow(form_idx/np.max(form_idx))
for i in range(form_embedded.shape[0]):
    plt.plot(form_embedded[i,0], form_embedded[i,1],'x', c=form_cols[i])
plt.savefig('figs/form_TSNE.png', dpi=300)

print('tonality predict - TSNE plot')
y_tonality = model_tonality.predict( processed_ds )
tonality_embedded = TSNE(n_components=2).fit_transform(y_tonality)
plt.clf()
tonality_cols = cm.rainbow(tonality_idx/np.max(tonality_idx))
for i in range(tonality_embedded.shape[0]):
    plt.plot(tonality_embedded[i,0], tonality_embedded[i,1],'x', c=tonality_cols[i])
plt.savefig('figs/tonality_TSNE.png', dpi=300)

print('composer predict - TSNE plot')
y_composer = model_composer.predict( processed_ds )
composer_embedded = TSNE(n_components=2).fit_transform(y_composer)
plt.clf()
composer_cols = cm.rainbow(composer_idx/np.max(composer_idx))
for i in range(composer_embedded.shape[0]):
    plt.plot(composer_embedded[i,0], composer_embedded[i,1],'x', c=composer_cols[i])
plt.savefig('figs/composer_TSNE.png', dpi=300)

print('genre predict - TSNE plot')
y_genre = model_genre.predict( processed_ds )
genre_embedded = TSNE(n_components=2).fit_transform(y_genre)
plt.clf()
genre_cols = cm.rainbow(genre_idx/np.max(genre_idx))
for i in range(genre_embedded.shape[0]):
    plt.plot(genre_embedded[i,0], genre_embedded[i,1],'x', c=genre_cols[i])
plt.savefig('figs/genre_TSNE.png', dpi=300)

print('plotting history')
import copy
history_normalized = copy.deepcopy(history)
for l in ['year_predictor_loss','form_predictor_loss','style_predictor_loss', 'tonality_predictor_loss', 'composer_predictor_loss', 'genre_predictor_loss', 'mask_predictor_loss']:
    history_normalized[l] = history_normalized[l]/np.max(history_normalized[l])
ax = history_normalized.plot(x='epoch', y=['year_predictor_loss', 'form_predictor_loss', 'style_predictor_loss', 'tonality_predictor_loss', 'composer_predictor_loss', 'genre_predictor_loss', 'mask_predictor_loss'])
ax.figure.savefig('figs/history_losses.png', dpi=300)

ax = history.plot(x='epoch', y=['form_predictor_sparse_categorical_accuracy', 'style_predictor_sparse_categorical_accuracy', 'tonality_predictor_sparse_categorical_accuracy', 'composer_predictor_sparse_categorical_accuracy', 'genre_predictor_sparse_categorical_accuracy', 'mask_predictor_sparse_categorical_accuracy'])
ax.figure.savefig('figs/history_accuracies.png', dpi=300)

print('exporting data')

mask_visualization_data = {
    'year': {
        'coordinates': year_embedded,
        'colors': year_cols
    },
    'style': {
        'coordinates': harmonic_style_embedded,
        'colors': harmonic_style_cols
    },
    'form': {
        'coordinates': form_embedded,
        'colors': form_cols
    },
    'tonality': {
        'coordinates': tonality_embedded,
        'colors': tonality_cols
    },
    'composer': {
        'coordinates': composer_embedded,
        'colors': composer_cols
    },
    'genre': {
        'coordinates': genre_embedded,
        'colors': genre_cols
    },
    'titles': list(useful_excel['Title'])
}

visualization_data_path =  'data/mask_visualization_data.pickle'
with open(visualization_data_path, 'wb') as handle:
    pickle.dump(mask_visualization_data, handle, protocol=pickle.HIGHEST_PROTOCOL)