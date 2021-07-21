import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.sparse import dia_matrix


my_data = np.genfromtxt("VV_MASS2.mtx", delimiter=',')

my_data = my_data[np.logical_not(np.logical_or(my_data[:, 1] == 6, my_data[:, 3] == 6)), :]

x_values = my_data[:, 0].astype(np.int32)
x_deg_values = my_data[:, 1].astype(np.int32)
y_values = my_data[:, 2].astype(np.int32)
y_deg_values = my_data[:, 3].astype(np.int32)
node_values = my_data[:, 4]

# negative node have degrees 1,2
# positive node have degrees 1,2,6
node_ids = list(set(x_values))
node_ids.sort()
# x index is matrix column or rows from node identifier (take account of previous node dimension columns)
node_index = {}
negative_nodes = len([x for x in node_ids if x < 0])
positive_nodes = len([x for x in node_ids if x > 0])


for x_id, j in zip(node_ids, range(len(node_ids))):
    if x_id < 0:
        j = j * 2
    else:
        j = negative_nodes * 2 + (j - negative_nodes) * 2
    node_index[x_id] = j

final_matrix_length = negative_nodes * 2 + positive_nodes * 2

dimension_index = {1: 0, 2: 1}

labels = []
for node_id in [x for x in node_ids if x < 0]:
    for node_dim in [1, 2]:
        labels.append("%d(%d)" % (node_id, node_dim))
for node_id in [x for x in node_ids if x > 0]:
    for node_dim in [1, 2]:
        labels.append("%d(%d)" % (node_id, node_dim))

fig, ax = plt.subplots()

mat = np.zeros((final_matrix_length, final_matrix_length))

for x, deg_x, y, deg_y, node_value in zip(x_values, x_deg_values, y_values, y_deg_values, node_values):
    j = node_index[x] + dimension_index[deg_x]
    i = node_index[y] + dimension_index[deg_y]
    mat[j, i] = node_value
    mat[i, j] = node_value
    #ax.text(i, j, "%.2g" % node_value, va='center', ha='center')
    #ax.text(j, i, "%.2g" % node_value, va='center', ha='center')



cax = ax.matshow(mat, cmap=plt.cm.Blues, vmin=0, vmax=node_values.max())
fig.colorbar(cax)

#ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_xticklabels([""] + labels)
ax.set_yticklabels([""] + labels)


plt.show()

#ret = #dia_matrix(tuple(a*31 for a in mat.shape), dtype=np.float32)
ret = np.zeros(tuple(a*31 for a in mat.shape), dtype=np.float32)

for loc in range(31):
    from_col = loc*mat.shape[0]
    to_col = loc*mat.shape[0] + mat.shape[0]
    from_row = loc*mat.shape[1]
    to_row = loc*mat.shape[1] + mat.shape[1]
    ret[from_col:to_col, from_row:to_row] = mat

np.set_printoptions(precision=3)
print(ret.shape)