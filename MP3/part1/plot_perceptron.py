
# import numpy as np
# test=np.load('./data/x_train.npy',encoding = "latin1")  #加载文件
# doc = open('1.txt', 'a')  #打开一个存储文件，并依次写入
# print(test, file=doc)


import numpy as np
from math import log
import matplotlib.pyplot as plt
temp = np.zeros((10))
train_set=np.load('./data/x_train.npy')
train_label=np.load('./data/y_train.npy')
test_set=np.load('./data/x_test.npy')
test_label=np.load('./data/y_test.npy')
likelihood=np.zeros((784,256,10))
old_w = np.zeros((785,10))
w = np.transpose(old_w)
new_train_set=train_set.reshape((50000,784))
new_test_set=test_set.reshape((10000,784))
accuracy=0.0
score = 0
image_type = 0
new_train_set = train_set.reshape((50000,784))
ARRAY = np.ones(50000)
new_train_set = np.insert(new_train_set,784,values=ARRAY, axis=1)
for i in range(new_train_set.shape[0]):
    #print(i)
    score = 0
    for j in range(w.shape[0]):
        # print(1)
        new_score = 0
        new_score_product = np.multiply(w[j],new_train_set[i])
        new_score = np.sum(new_score_product)
        if new_score > score:
                # print(3)
                score = new_score
                image_type = j
    if image_type!=train_label[i]:
            # print(4)
            for p in range(w.shape[1]):
                # print(5)
                w[image_type][p] -= new_train_set[i][p]
                w[train_label[i]][p] += new_train_set[i][p]

accuracy = 0
pred_label = np.zeros((len(test_set)))
new_test_set = test_set.reshape((10000,784))
bias = np.ones(10000)
max_list = []
min_list = []
new_test_set = np.insert(new_test_set,784,values=bias,axis=1)
# for i in range(new_test_set.shape[0]):
#     score = 0
#     for j in range(w.shape[0]):
#         new_score = 0
#         new_score_product = np.multiply(w[j],new_test_set[i])
#         new_score = np.sum(new_score_product)
#         if new_score>score:
#             image_type = j
#             score = new_score
#     pred_label[i] = image_type
# for i in range(pred_label.size):
#     if pred_label[i] == test_label[i]:
#         accuracy += 1
classes = np.array(["T-shirt/top","Trouser","Pullover","Dress",
           "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])
for j in range(w.shape[0]):
    max_score = float("-inf")
    min_score = float("inf")

    for i in range(new_test_set.shape[0]):
        if test_label[i] == j:
            new_score_product = np.multiply(w[j],new_test_set[i])
            new_score = np.sum(new_score_product)
        if max_score<new_score:
            max_score = new_score
            max_index = i
        if min_score>new_score:
            min_score = new_score
            min_index = i
    max_list.append(max_index)
    min_list.append(min_index)
fig, ax = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
        ax[i%2, i//2].imshow(new_test_set[min_list[i],:784].reshape((28, 28)), cmap="Greys")
        ax[i%2, i//2].set_xticks([])
        ax[i%2, i//2].set_yticks([])
        ax[i%2, i//2].set_title(classes[i])
plt.show()
accuracy = accuracy/10000
print(accuracy)
