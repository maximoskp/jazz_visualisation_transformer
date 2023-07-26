import numpy as np
from copy import deepcopy

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t

# A = np.array([[ 23, 178],
#               [ 66, 173],
#               [ 88, 187],
#               [119, 202],
#               [122, 229],
#               [170, 232],
#               [179, 199]])
# B = np.array([[232, 38],
#               [208, 32],
#               [181, 31],
#               [155, 45],
#               [142, 33],
#               [121, 59],
#               [139, 69]])

# R, c, t = kabsch_umeyama(A, B)
# B = np.array([t + c * R @ b for b in B])

with open('data/transposition_visualization_data.pickle', 'rb') as handle:
    mask_visualization_data = pickle.load(handle)

with open('data/no_mask_visualization_data.pickle', 'rb') as handle:
    no_mask_visualization_data = pickle.load(handle)

# align all with year
kz = ['style', 'form', 'tonality', 'composer', 'genre']

mask_rotated_visualization_data = deepcopy(mask_visualization_data)
no_mask_rotated_visualization_data = deepcopy(no_mask_visualization_data)

# first rotate mask data
a = mask_visualization_data['year']['coordinates']
for k in kz:
    b = mask_visualization_data[k]['coordinates']
    R, c, t = kabsch_umeyama(a, b)
    # b = np.array([t + c * R @ b_line for b_line in b]) # translation, scaling and rotation
    b = np.array([R @ b_line for b_line in b]) # rotation only
    mask_rotated_visualization_data[k]['coordinates'] = b

visualization_data_path =  'data/mask_rotated_visualization_data.pickle'
with open(visualization_data_path, 'wb') as handle:
    pickle.dump(mask_rotated_visualization_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# then rotate no mask data
a = no_mask_visualization_data['year']['coordinates']
for k in kz:
    b = no_mask_visualization_data[k]['coordinates']
    R, c, t = kabsch_umeyama(a, b)
    # b = np.array([t + c * R @ b_line for b_line in b]) # translation, scaling and rotation
    b = np.array([R @ b_line for b_line in b]) # rotation only
    no_mask_rotated_visualization_data[k]['coordinates'] = b

visualization_data_path =  'data/no_mask_rotated_visualization_data.pickle'
with open(visualization_data_path, 'wb') as handle:
    pickle.dump(no_mask_rotated_visualization_data, handle, protocol=pickle.HIGHEST_PROTOCOL)