from __future__ import absolute_import
import random
from random import shuffle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import xlrd
import os


class Dataset(object):

    def __init__(self, seed=22, dataset_nm='np'):
        self.seed = seed
        self.cur = 0
        self.classes = 4
        self.tables = 6
        self.num_ins = 4
        self.dataset_nm = dataset_nm
        self.data_dir = './data/data190724_2'
        self.datasets = self.load_dataset()


    def load_dataset(self):
        # load data from file
        labels, names = [], []
        data_full = [[] for _ in range(self.num_ins +1)]
        classes = ['1', '2', '3', '4']
        for num in os.listdir(self.data_dir + '/train'):
            lab = classes.index(num.split('.xlsx')[0])
            data = xlrd.open_workbook(self.data_dir + '/train/' + str(num))
            tables = [[] for _ in range(self.tables)]
            for idx in range(self.tables):
                tables[idx] = data.sheets()[idx]
            # start is the first row
            start = 2
            # end is the final row
            end = 6002
            for row in range(start, end):
                data_row = [[] for _ in range(self.tables)]
                for i in range(self.tables):
                    data_row[i].extend(tables[i].row_values(row))

                # get every line feature and full feature
                bag = np.hstack((data_row[0][0:72], data_row[1][0:72], data_row[2][0:72],
                                 data_row[3][0:60], data_row[4][0:60], data_row[5][0:60]))
                ins1 = np.hstack((data_row[0][0:12], data_row[1][0:12], data_row[2][0:12],
                                  data_row[3][0:9], data_row[4][0:9], data_row[5][0:9]))
                ins2 = np.hstack((data_row[0][12:33], data_row[1][12:33], data_row[2][12:33],
                                  data_row[3][9:27], data_row[4][9:27], data_row[5][9:27]))
                ins3 = np.hstack((data_row[0][33:60], data_row[1][33:60], data_row[2][33:60],
                                  data_row[3][27:51], data_row[4][27:51], data_row[5][27:51]))
                ins4 = np.hstack((data_row[0][60:72], data_row[1][60:72], data_row[2][60:72],
                                  data_row[3][51:60], data_row[4][51:60], data_row[5][51:60]))

                for idx, fea in enumerate([bag, ins1, ins2, ins3, ins4]):
                    data_full[idx].append(fea)

                label = [0] * self.classes
                label[lab] = 1
                labels.append(label)
                names.append(num)

        for i in range(len(data_full)):
            mean_fea = np.mean(data_full[i], axis=0)
            # print i
            std_fea = np.std(data_full[i], axis=0, keepdims=True) + 1e-6
            data_full[i] = np.divide(data_full[i] - mean_fea, std_fea)

        # write data to dictionary
        dataset = {}
        for idx, lab in enumerate(labels):
            dataset[idx] = {'fea' + str(_): data_full[_][idx] for _ in range(self.num_ins + 1)}
            dataset[idx]['label'] = lab
            dataset[idx]['name'] = names[idx]
        shuffle(dataset)

        kf = KFold(n_splits=10, shuffle=True, random_state=self.seed)
        datasets = []
        for train_idx, test_idx in kf.split(dataset):
            dataset_ = {}
            dataset_['train'] = [dataset[i] for i in train_idx]
            dataset_['test'] = [dataset[i] for i in test_idx]
            datasets.append(dataset_)

        return datasets

    def get_next_batch(self, step, fold=0, is_train=True, batch_size=1, test_type=0):
        if is_train is True:
            dataset = self.datasets[fold]['train']
            fea_ins = [[] for _ in range(self.num_ins)]
            ins = [[] for _ in range(self.num_ins)]
            labels = []
            fea_all = []
            name = []
            for i in range(batch_size):
                for idx, fea in enumerate(ins):
                    ins[idx] = np.asarray(dataset[self.cur]['fea' + str(idx + 1)], dtype='float32')

                for idx, fea in enumerate(fea_ins):
                    fea.append(ins[idx])
                fea_all.append(np.asarray(dataset[self.cur]['fea0'], dtype='float32'))
                label = dataset[self.cur]['label']

                labels.append(label)
                name.append(dataset[self.cur]['name'])
                self.cur += 1
                if self.cur == len(dataset):
                    shuffle(dataset)
                    self.cur = 0

        else:
            if test_type == 1:
                dataset = self.datasets[fold]['train']
            else:
                dataset = self.datasets[fold]['test']

            fea_ins = [[] for _ in range(self.num_ins)]
            ins = [[] for _ in range(self.num_ins)]
            for idx, fea in enumerate(ins):
                ins[idx] = np.asarray(dataset[step]['fea' + str(idx + 1)], dtype='float32')

            for idx, fea in enumerate(fea_ins):
                fea.append(ins[idx])

            labels = []
            fea_all = []
            name = []
            fea_all.append(np.asarray(dataset[step]['fea0'], dtype='float32'))
            label = dataset[step]['label']

            labels.append(label)
            name.append(dataset[step]['name'])
        return fea_ins, fea_all, labels, name
