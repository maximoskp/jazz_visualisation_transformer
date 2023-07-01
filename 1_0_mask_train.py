import os
import keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras_nlp
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import os

history_path = 'data/model_mask_history_log.csv'
initial_epoch = 0
if os.path.isfile( history_path ):
    print('history exists')
    history = pd.read_csv(history_path)
    initial_epoch = history['epoch'].iloc[-1] + 1
else:
    print('starting fresh')

checkpoint_path = 'models/mask_1/model'
songs_excel_path = 'data/1058_songs_without_melodies.xlsx'
augmented_excel_path = 'data/augmented_excel.csv'
music_vocab_list_path = 'data/music_vocab_list.pickle'
tokens_path = 'data/tokens.pickle'

print('reading augmented excel, tokens and vocabulary')
augmented_excel = pd.read_csv(augmented_excel_path)
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

augmented_excel['tokens'] = tokens

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
PRETRAINING_EPOCHS = 700
FINETUNING_LEARNING_RATE = 5e-5
FINETUNING_EPOCHS = 3

BATCH_SIZE = 16

print('padding to sequence length')
music_texts_SEQ_LENGTH = []
for s in tokens:
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

print('making target vectors')
# normalized year data
year = augmented_excel['composition_date'].to_numpy()
year_min = np.min(year)
year_max = np.max(year)
year_norm = list( (year - year_min)/(year_max - year_min) )
# style categories
harmonic_style_dict, harmonic_style_idx = get_dictionary_and_idxs_from_pd_column( augmented_excel['harmonic_style'] )
# style categories
form_dict, form_idx = get_dictionary_and_idxs_from_pd_column( augmented_excel['form'] )
# tonality categories
tonality_dict, tonality_idx = get_dictionary_and_idxs_from_pd_column( augmented_excel['tonality'] )
# composer categories
composer_dict, composer_idx = get_dictionary_and_idxs_from_pd_column( augmented_excel['composer'] )
# genre categories
genre_dict, genre_idx = get_dictionary_and_idxs_from_pd_column( augmented_excel['genre_style'] )

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

year_output = keras.layers.Dense(256, activation='relu')(post_encoding_reduction)
year_output = keras.layers.Dense(512, activation='relu')(year_output)
year_output = keras.layers.Dense(256, activation='relu')(year_output)
year_output = keras.layers.Dense(64, activation='relu', name='year_descriptor')(year_output)
year_output = keras.layers.Dense(1, activation='sigmoid', name='year_predictor')(year_output)

style_output = keras.layers.Dense(256, activation='relu')(post_encoding_reduction)
style_output = keras.layers.Dense(256, activation='relu')(style_output)
style_output = keras.layers.Dense(64, activation='relu', name='style_descriptor')(style_output)
style_output = keras.layers.Dense(len(harmonic_style_dict), activation='softmax', name='style_predictor')(style_output)

form_output = keras.layers.Dense(256, activation='relu')(post_encoding_reduction)
form_output = keras.layers.Dense(256, activation='relu')(form_output)
form_output = keras.layers.Dense(64, activation='relu', name='form_descriptor')(form_output)
form_output = keras.layers.Dense(len(form_dict), activation='softmax', name='form_predictor')(form_output)

tonality_output = keras.layers.Dense(256, activation='relu')(post_encoding_reduction)
tonality_output = keras.layers.Dense(256, activation='relu')(tonality_output)
tonality_output = keras.layers.Dense(64, activation='relu', name='tonality_descriptor')(tonality_output)
tonality_output = keras.layers.Dense(len(tonality_dict), activation='softmax', name='tonality_predictor')(tonality_output)

composer_output = keras.layers.Dense(512, activation='relu')(post_encoding_reduction)
composer_output = keras.layers.Dense(512, activation='relu')(composer_output)
composer_output = keras.layers.Dense(256, activation='relu')(composer_output)
composer_output = keras.layers.Dense(64, activation='relu', name='composer_descriptor')(composer_output)
composer_output = keras.layers.Dense(len(composer_dict), activation='softmax', name='composer_predictor')(composer_output)

genre_output = keras.layers.Dense(256, activation='relu')(post_encoding_reduction)
genre_output = keras.layers.Dense(256, activation='relu')(genre_output)
genre_output = keras.layers.Dense(64, activation='relu', name='genre_descriptor')(genre_output)
genre_output = keras.layers.Dense(len(genre_dict), activation='softmax', name='genre_predictor')(genre_output)

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

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor='loss',
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)

csv_path = history_path
csv_dir = os.path.dirname(csv_path)
csv_logger = tf.keras.callbacks.CSVLogger(csv_path, append=True)

if initial_epoch > 0:
    print('loading previous weights')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

print('starting training')
model.fit(
    processed_ds,
    epochs=PRETRAINING_EPOCHS+initial_epoch,
    callbacks=[cp_callback, csv_logger],
    initial_epoch=initial_epoch
)
