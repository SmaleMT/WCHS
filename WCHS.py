import math
import numpy as np
import pandas as pd
from sklearn import decomposition, preprocessing
from sklearn.neighbors import NearestNeighbors


class KNN:
    def __init__(self, k):
        self.k = k
        self.model = NearestNeighbors(n_neighbors=self.k)

    def fit(self, X):
        self.X = X
        self.model.fit(X)

    def predict(self, sample):
        distances, indices = self.model.kneighbors(np.array(sample).reshape(1, -1))
        indices = indices[0]
        samples = [self.X[i] for i in indices]
        return distances[0][1:], samples[1:], indices[:]


class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.Entropy = np.nan
        self.pred = np.nan
        self.pred_prob = np.nan


def WCHS(x_train, x_test, y_train, y_test, d, k, n, lamba):
    standard_train_x = preprocessing.MinMaxScaler()
    scaler_standard = standard_train_x.fit(x_train)
    x_train = scaler_standard.transform(x_train)
    x_test = scaler_standard.transform(x_test)

    # KPCA
    KPCA_lc_x1 = decomposition.KernelPCA(n_components=d, kernel='poly')
    kpca_standard = KPCA_lc_x1.fit(x_train)
    x_train = kpca_standard.transform(x_train)
    x_test = kpca_standard.transform(x_test)
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    print(f'train set:{x_train.shape}')
    # data balance
    x_train, y_train = wcontribution_sampling(x_train, y_train, k, n, lamba)

    return x_train, y_train, x_test, y_test


def wcontribution_sampling(x_train, y_train, k, n, lamba):
    x_train_balanced = np.array(x_train.copy())
    y_train_balanced = np.array(y_train.copy())
    train_data = Data(x_train_balanced, y_train_balanced)
    x_train_balanced, y_train_balanced = WUnderSampling(train_data, k, n=n, lamba=lamba)

    x_train_balanced, y_train_balanced = OverSampling(x_train_balanced, y_train_balanced, k, lamba=lamba)
    return x_train_balanced, y_train_balanced


# underSampling
def WUnderSampling(train_data, k, n, lamba):
    x_train = train_data.x
    y_train = train_data.y
    contribution = Contribution(train_data, k, lamba)

    saveIndex = []
    num_positive = np.sum(y_train == 1)
    num_negative = np.sum(y_train == 0)
    contribution = contribution.sort_values(by='contribution', ascending=False)

    for i in range(contribution.shape[0]):
        if int(contribution.iloc[i]['y_train']) == 1:
            saveIndex.append(int(contribution.iloc[i]['x_train']))
    j = 0
    for i in range(contribution.shape[0]):
        if int(contribution.iloc[i]['y_train']) == 0:
            saveIndex.append(int(contribution.iloc[i]['x_train']))
            j += 1
            if j == int(n * num_positive) or j == int(num_negative):
                break

    x_train_balanced = x_train[saveIndex]
    y_train_balanced = y_train[saveIndex]

    return x_train_balanced, y_train_balanced


# overSampling and filter
def OverSampling(x_train, y_train, k, lamba):
    num_positive = np.sum(y_train == 1)
    num_negative = np.sum(y_train == 0)
    num_to_generate = max(0, num_negative - num_positive)
    if num_to_generate == 0:
        return x_train, y_train

    train_data_over = Data(x_train, y_train)

    contribution = Contribution(train_data_over, k, lamba)
    positive_indices = np.where(y_train == 1)[0]
    positive_contributions = contribution.iloc[positive_indices]['contribution']
    normalized_value = (positive_contributions - np.min(positive_contributions)) / (
            np.max(positive_contributions) - np.min(positive_contributions))
    weights = normalized_value / sum(normalized_value)

    knn = KNN(k=k + 1)
    knn.fit(x_train)
    new_samples = []
    for _ in range(num_to_generate):
        chosen_index = np.random.choice(positive_indices, p=weights)
        chosen_sample = x_train[chosen_index]

        _, neighbor_samples, neighbor_indices = knn.predict(chosen_sample.reshape(1, -1))
        neighbor_indices = neighbor_indices[1:]
        neighbor_index = np.random.choice(neighbor_indices)
        neighbor_sample = x_train[neighbor_index]
        alpha = np.random.rand()
        new_sample = chosen_sample + alpha * (neighbor_sample - chosen_sample)
        new_samples.append(new_sample)

    new_samples = np.array(new_samples)
    x_syn = np.vstack([x_train, new_samples])
    y_syn = np.hstack([y_train, np.ones(len(new_samples))])
    syn_data = Data(x_syn, y_syn)
    contribution_syn = Contribution(syn_data, k, lamba)
    new_samples_contribution = contribution_syn.iloc[len(x_train):]['contribution'].values

    fin_new_samples = new_samples[new_samples_contribution >= 0]
    x_train_resampled = np.vstack([x_train, fin_new_samples])
    y_train_resampled = np.hstack([y_train, np.ones(len(fin_new_samples))])

    return x_train_resampled, y_train_resampled


def Contribution(train_data, k, lamba):
    x_train = train_data.x
    y_train = train_data.y
    contribution = pd.DataFrame()
    contribution['x_train'] = np.arange(x_train.shape[0])
    contribution['y_train'] = y_train

    same_count_UMatrix, same_count_IUMatrix, Rknn_index = U_IU_same_Matrix(x_train, y_train, k)

    contribution['WInf'] = w_Information_entropy(train_data, k)

    SNoi_x_train = []
    IU_Neighbor = np.array([len(row) for row in Rknn_index])
    for i in range(len(x_train)):
        IU_Information_entropy = Information_entropy(IU_Neighbor[i], same_count_IUMatrix[i])
        SNoi_x_train.append(IU_Information_entropy)

    min_num = 0
    max_num = 1
    entropy = np.array(SNoi_x_train)
    min_entropy = np.min(entropy)
    max_entropy = np.max(entropy)
    SNoi_x_train = min_num + (max_num - min_num) * (entropy - min_entropy) / (max_entropy - min_entropy)

    densities = calculate_density(x_train, k)
    contribution['WNoi'] = SNoi_x_train * (1 - densities)

    contribution['contribution'] = (1 - lamba) * contribution['WInf'] - lamba * contribution['WNoi']
    return contribution


def calculate_density(x_train, k):
    eps = 0.00001

    knn = KNN(k + 1)
    knn.fit(x_train)
    densities = []
    for sample in x_train:
        dis, _, _ = knn.predict(sample)
        avg_dis = np.mean(dis)
        density = 1 / (avg_dis + eps)
        densities.append(density)

    densities = np.array(densities)
    max_density = np.max(densities)
    min_density = np.min(densities)
    normalized_densities = (densities - min_density) / (max_density - min_density + eps)  # 归一化密度

    return normalized_densities


def U_IU_same_Matrix(x_train, y_train, k):
    x_train = np.array(x_train.copy())

    knn = KNN(k=k + 1)
    knn.fit(x_train)
    knn_index = []
    knn_samples = []
    for item in x_train:
        _, samples, ls_res = knn.predict(item)
        knn_index.append(ls_res[1:])
        knn_samples.append(samples[:])

    same_count_UMatrix = []
    Rknn_index = []
    for idx in range(len(x_train)):
        neighbor_labels = y_train[knn_index[idx]]
        same_count = np.sum(neighbor_labels == y_train[idx])
        same_count_UMatrix.append(same_count)

        row_Rknn = []
        for IU_index, item in enumerate(knn_index):
            if idx in item:
                row_Rknn.append(IU_index)
        Rknn_index.append(row_Rknn)

    same_count_IUMatrix = []
    for i in range(len(x_train)):
        neighbor_labels = y_train[Rknn_index[i]]
        same_count = np.sum(neighbor_labels == y_train[i])
        same_count_IUMatrix.append(same_count)

    return same_count_UMatrix, same_count_IUMatrix, Rknn_index


def Information_entropy(k, same):
    Bias = 0.001
    diff = k - same
    prob_same = (same + Bias) / (k + Bias)
    prob_diff = (diff + Bias) / (k + Bias)
    if k == 0:
        information_entropy = 0
    elif same / k < 0.5:
        information_entropy = - prob_same * math.log(prob_same) - prob_diff * math.log(prob_diff)  # 信息熵
        information_entropy = 2 * (-math.log(0.5)) - information_entropy
    else:
        information_entropy = - prob_same * math.log(prob_same) - prob_diff * math.log(prob_diff)

    return information_entropy


def w_Information_entropy(train_data, k):
    x_train = train_data.x
    y_train = train_data.y
    gamma = 1

    knn = KNN(k=k + 1)
    knn.fit(x_train)
    Entropy = []
    for item in x_train:
        entropy = 0
        dis, samples, ls_res = knn.predict(item)
        index = ls_res[0]
        ls_res = ls_res[1:]
        same = np.sum(y_train[ls_res] == y_train[index])
        weights = [np.exp(-gamma * d) for d in dis]
        total_weight = sum(weights)

        labels = y_train[ls_res]
        class_weights = {}
        for label, weight in zip(labels, weights):
            class_weights[label] = class_weights.get(label, 0) + weight
        p_k = np.array(list(class_weights.values())) / total_weight
        entropy = -sum(p * np.log(p) for p in p_k if p > 0)

        p_same = sum(w for label, w in zip(labels, weights) if label == y_train[index]) / total_weight
        if k == 0:
            entropy = 0
        elif p_same < 0.5:
            entropy = 2 * (-math.log(0.5)) - entropy
        Entropy.append(entropy)

    return Entropy
