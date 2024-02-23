import pyepo
import numpy as np

num_data = 1 # number of data
num_feat = 5 # size of feature
num_item = 4 # number of items
dim = 10 # dimension of knapsack
w, x, c = pyepo.data.knapsack.genData(num_data, num_feat, num_item, dim, deg=4, noise_width=0, seed=135)

print(w) # dim x num_items
print(x) # num_data X num_features
print(c) # num_Data x num_items

zero_matrix = np.zeros_like(w)
new_w = zero_matrix
new_w[1] = w[1]
print(new_w)
capacities = list(map(lambda x: np.sum(x) * 0.66, new_w))
print(capacities)