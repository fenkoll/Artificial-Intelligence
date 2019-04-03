
# import numpy as np
# test=np.load('./data/x_train.npy',encoding = "latin1")  #加载文件
# doc = open('1.txt', 'a')  #打开一个存储文件，并依次写入
# print(test, file=doc)


import numpy as np
import math
import matplotlib.pyplot as plt
temp = np.zeros((10))
train_set=np.load('./data/x_train.npy')
train_label=np.load('./data/y_train.npy')
test_set=np.load('./data/x_test.npy')
test_label=np.load('./data/y_test.npy')
likelihood=np.zeros((784,256,10))
map_table = np.zeros((10,10000))
new_train_set=train_set.reshape((50000,784))
new_test_set=test_set.reshape((10000,784))
max_list = []
min_list = []
accuracy=0.0
for i in range(new_train_set.shape[0]):
	for j in range(new_train_set.shape[1]):
		likelihood[j][new_train_set[i][j]][train_label[i]]+=1

for i in range(likelihood.shape[0]):
	for j in range(likelihood.shape[1]):
		for k in range(likelihood.shape[2]):
                    likelihood[i][j][k] = (likelihood[i][j][k]+0.1)/(5000+0.1*256)
            
#print(likelihood[396][155][0])

pred_label = np.zeros((len(test_set)))
class_map=np.zeros((10))
# for i in range(new_test_set.shape[0]):
#     print(i)
#     for k in range(10):
#         map = math.log(0.1,10)
#         for j in range(new_test_set.shape[1]):
#             map = map+math.log(likelihood[j][new_test_set[i][j]][k],10)
#             class_map[k] = map
#
#         else:
#             map_table[k][i] = float("nan")
#         pred_label[i] = np.argmax(class_map)

classes = np.array(["T-shirt/top","Trouser","Pullover","Dress",
           "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])
for j in range(10):
    max_index = 0
    min_index = 0
    max_map = float("-inf")
    min_map = float("inf")
    for i in range(10000):
        #print(i)
        new_map = math.log(0.1,10)
        if test_label[i] == j:
            for k in range(new_test_set.shape[1]):
                new_map = new_map+math.log(likelihood[k][new_test_set[i][k]][j],10)
            if max_map<new_map:
                #print(new_map)
                max_map = new_map
                max_index = i
            if min_map>new_map:
                min_map = new_map
                min_index = i
    max_list.append(max_index)
    min_list.append(min_index)

# for i in range(pred_label.size):
#     if pred_label[i]==test_label[i]:
#         accuracy+=1
# accuracy=accuracy/10000
print (accuracy)
print(max_list)
print(min_list)
fig, ax = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
        ax[i%2, i//2].imshow(new_test_set[max_list[i],:].reshape((28, 28)), cmap="Greys")
        ax[i%2, i//2].set_xticks([])
        ax[i%2, i//2].set_yticks([])
        ax[i%2, i//2].set_title(classes[i])
plt.show()
# print (likelihood)
#print (a.shape[1])
# for each in test_label:
# 	print(each)
