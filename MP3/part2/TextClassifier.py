# TextClassifier.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019
from math import exp, log
import numpy as np
"""
You should only modify code within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
def bi_list_to_string(a, b):
    return a + "&" + b

# def string_to_bi_list(c):
#     t = list(c)
#     a = str(c[0: t.index['\n']])
#     b = str(c[t.index['\n'] + 1, len(c)])
#     return a,b

class TextClassifier(object):
    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 0.0

    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
        """

        # TODO: Write your code here
        self.v = set()
        self.prior = [0] * 14
        self.dic = {l:dict() for l in train_label}
        self.dic_bi = {l:dict() for l in train_label}
        # self.probability = {l:dict() for l in train_label}
        smoother = 1
        for i in range(len(train_set)):
            self.prior[train_label[i] - 1] += 1
            for w in train_set[i]:
                self.v.add(w)
                self.dic[train_label[i]][w] = self.dic[train_label[i]].get(w, 0) + 1
            for j in range(len(train_set[i]) - 1):
                self.dic_bi[train_label[i]][bi_list_to_string(train_set[i][j], train_set[i][j + 1])] = self.dic_bi[train_label[i]].get(bi_list_to_string(train_set[i][j], train_set[i][j + 1]), 0) + 1
        self.num_words = len(self.v)
        # for i in self.dic:
        #     n = 0
        #     for j in self.dic[i]:
        #         n += self.dic[i][j]
            # for j in self.dic[i]:
            #     self.probability[i][j] = self.probability[i].get(j, 0) + log((self.dic[i][j] + smoother)/(n + smoother * num_words))

    def predict(self, dev_set, dev_label,lambda_mix=0.0):
        """
        :param dev_set: List of list of words corresponding with each text in dev set that we are testing on
              It follows the same format as train_set
        :param dev_label : List of class labels corresponding to each text
        :param lambda_mix : Will be supplied the value you hard code for self.lambda_mixture if you attempt extra credit

        :return:
                accuracy(float): average accuracy value for dev dataset
                result (list) : predicted class for each text
        """
        # * (self.dic[i + 1].get(dev_set[s][0], 0) + 10) / (sum(self.dic[i + 1].values()) + self.num_words)
        accuracy = 0.0
        result = []
        smoother = 0.01
        smoother_bi = 2.5
        # cmt = np.zeros((14,14))
        # ct = np.zeros(14)
        for s in range(len(dev_set)):
            uni_res = []
            bi_res = []
            # print(dev_set[s])
            for i in range(14):
                uni_res.append(exp(sum(log((self.dic[i + 1].get(w, 0) + smoother)/(sum(self.dic[i + 1].values()) + self.num_words * smoother)) for w in dev_set[s])))
                bi_res.append(self.prior[i] / sum(self.prior) * exp(sum(log((self.dic_bi[i + 1].get(bi_list_to_string(dev_set[s][b], dev_set[s][b + 1]), 0) + smoother_bi)/(sum(self.dic[i + 1].values()) + smoother_bi * self.num_words)) for b in range(len(dev_set[s]) - 1))))
            mix_res = [lambda_mix * bi_res[i] + (1 - lambda_mix) * uni_res[i] for i in range(len(bi_res))]
            result.append(mix_res.index(max(mix_res)) + 1)
            # cmt[dev_label[s] - 1][res[s] - 1] += 1
            if result[s] == dev_label[s]:
                accuracy += 1 / len(dev_set)
            # ct[dev_label[s] - 1] += 1
            # print(uni_res)
        # for i in range(14):
        #     cmt[i] = cmt[i]/ct[i]
        # cm = [[str(cmt[i][j])[0:4] for j in range(14)] for i in range(14)]
        # # print(res)
        # f = open("res.txt", "w")
        # for i in range(14):
        #     for j in range(14):
        #         f.write(cm[i][j] + ("0" * (4 - len(cm[i][j])))+ ", ")
        #     f.write("\n")
        # # f.write(cm)
        # f.close()


        return accuracy, result
