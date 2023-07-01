import os
import keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from copy import deepcopy

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import os

songs_excel_path = 'data/1058_songs_without_melodies.xlsx'
augmented_excel_path = 'data/augmented_excel.csv'
music_vocab_list_path = 'data/music_vocab_list.pickle'
tokens_path = 'data/tokens.pickle'
data_info_txt_path = 'data/data_info.txt'

print('loading initial excel')
loaded_excel = pd.read_excel(songs_excel_path)
loaded_excel = loaded_excel.set_index('Title')

print('making useful excel - discarding nans')
useful_columns_excel = loaded_excel.drop(['original_string','original_key','appearing_name', 'is_favourite','performed_by'], axis=1)
useful_excel = useful_columns_excel.dropna()

root_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
# rename to all sharps
def rename_to_sharps(s):
    # change tonality
    tonality_split = s.split('tonality~')
    if tonality_split[1][1] == 'b' or tonality_split[1][1] == '-':
        root_idx = root_names.index( tonality_split[1][0] )
        rest_of_the_piece = tonality_split[1][2:]
        # check for Fb and Cb
        if tonality_split[1][0] == 'F' or tonality_split[1][0] == 'C':
            new_root = root_names[ np.mod( root_idx-1, len(root_names) ) ]
        else:
            new_root = root_names[ np.mod( root_idx-1, len(root_names) ) ] + '#'
        tonality_split[1] = new_root + rest_of_the_piece
    elif tonality_split[1][1] == '#':
        # check for B# and E#
        if tonality_split[1][0] == 'B' or tonality_split[1][0] == 'E':
            root_idx = root_names.index( tonality_split[1][0] )
            new_root = root_names[ np.mod( root_idx+1, len(root_names) ) ]
            rest_of_the_piece = tonality_split[1][2:]
            tonality_split[1] = new_root + rest_of_the_piece
    s = 'tonality~'.join(tonality_split)
    # change chords
    chord_split = s.split('chord~')
    new_chords = [chord_split[0]]
    for chord in chord_split[1:]:
        if chord[1] == 'b' or chord[1] == '-':
            root_idx = root_names.index( chord[0] )
            new_type = chord[2:] if len(chord) > 2 else ''
            # check for Fb and Cb
            if chord[0] == 'F' or chord[0] == 'C':
                new_root = root_names[ np.mod( root_idx-1, len(root_names) ) ]
            else:
                new_root = root_names[ np.mod( root_idx-1, len(root_names) ) ] + '#'
            chord = new_root + new_type
        elif chord[1] == '#':
            # check for B# and E#
            if chord[0] == 'B' or chord[0] == 'E':
                root_idx = root_names.index( chord[0] )
                new_root = root_names[ np.mod( root_idx+1, len(root_names) ) ]
                new_type = chord[2:] if len(chord) > 2 else ''
                chord = new_root + new_type
        new_chords.append(chord)
    return 'chord~'.join(new_chords)
# end rename_to_sharps

print('making sharps excel')
sharps_excel = deepcopy( useful_excel )
for i, row in useful_excel.iterrows():
    renamed = rename_to_sharps( row['string'] )
    # get tonality
    tonality = renamed.split('tonality~')[1].split(',')[0]
    sharps_excel.at[i, 'string'] = renamed
    sharps_excel.at[i, 'tonality'] = tonality
# end for

# array of root pitch names
root_pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# transpose
def transpose_semitones(piece, semitones):
    # change tonality
    tonality_split = piece.split('tonality~')
    root_split_idx = 1
    if tonality_split[1][1] == '#':
        root_split_idx = 2
    rest_of_the_piece = tonality_split[1][root_split_idx:]
    root_idx = root_pitches.index( tonality_split[1][:root_split_idx] )
    new_root = root_pitches[ np.mod( root_idx + semitones, len(root_pitches) ) ]
    tonality_split[1] = new_root + rest_of_the_piece
    piece = 'tonality~'.join(tonality_split)
    # change chords
    chord_split = piece.split('chord~')
    new_chords = [chord_split[0]]
    for chord in chord_split[1:]:
        root_split_idx = 1
        if chord[1] == '#':
            root_split_idx = 2
        new_type = chord[root_split_idx:] if len(chord) > root_split_idx else ''
        root_idx = root_pitches.index( chord[:root_split_idx] )
        new_root = root_pitches[ np.mod( root_idx + semitones, len(root_pitches) ) ]
        new_chords.append( new_root + new_type )
    return 'chord~'.join( new_chords )
# end transpose_semitones

sharps_excel = sharps_excel.reset_index()
augmented_excel = pd.DataFrame(columns=sharps_excel.columns)

tmp_idx = 0

is_original_tonality = []

print('appending new tonalities')
for i, row in sharps_excel.iterrows():
    for t in range(12):
        tmp_row = deepcopy( row )
        tmp_row['string'] = transpose_semitones( row['string'], t )
        tmp_row['tonality'] = tmp_row['string'].split('tonality~')[1].split(',')[0]
        if t == 0:
            is_original_tonality.append(True)
        else:
            is_original_tonality.append(False)
        augmented_excel.loc[tmp_idx] = tmp_row
        tmp_idx += 1
    # end for
# end for
augmented_excel['is_original_tonality'] = is_original_tonality

print('saving augmented excel')
with open(augmented_excel_path, 'w', encoding = 'utf-8-sig') as f:
    augmented_excel.to_csv(f)

print('creating tokens')
import re
tokens = []
max_size_tokens = -1
for s in augmented_excel['string']:
    splt = re.split( ',|@' , s)
    if len( splt ) > max_size_tokens:
        max_size_tokens = len(splt)
    tokens.append( splt )

augmented_excel['tokens'] = tokens
music_vocab_list = []
for s in tokens:
    for t in s:
        if t not in music_vocab_list:
            music_vocab_list.append( t )

if '[UNK]' not in music_vocab_list:
    music_vocab_list.append('[UNK]')
if '[PAD]' not in music_vocab_list:
    music_vocab_list.append('[PAD]')
if '[MASK]' not in music_vocab_list:
    music_vocab_list.append('[MASK]')

with open(data_info_txt_path, 'w') as f:
    print('total number of pieces including transpositions: ' + str(len(tokens)), file=f)
    print('maximum sequence length: ' + str(max_size_tokens), file=f)
    print('vocabulary size: ' + str(len(music_vocab_list)), file=f)

with open(tokens_path, 'wb') as handle:
    pickle.dump(tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(music_vocab_list_path, 'wb') as handle:
    pickle.dump(music_vocab_list, handle, protocol=pickle.HIGHEST_PROTOCOL)