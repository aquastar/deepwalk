import cPickle as pk
import collections
import copy
import time
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from networkx import from_scipy_sparse_matrix
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC

from deepwalk.__main__ import main
from gen_simulate import graph_forge


def evaluate_roc(y_test, pred, acc, name='deepwalk'):
    pass
    t_test = time.time()

    # plot
    # fpr, tpr, _ = roc_curve(np.where(labels == 1)[1], np.argmax(outs_val[2], axis=1)[mask])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = np.max(pred) + 1
    y_test = to_categorical(y_test)
    pred = to_categorical(pred)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Plot ROC curves for the multiclass problem
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 10))
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(
        ['aqua', 'darkorange', 'cornflowerblue', 'bisque', 'seagreen', 'magenta', 'b', 'c', 'r', 'plum', 'cyan',
         'lime'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{}, Precison:{}'.format(name, acc))
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(name)

if __name__ == '__main__':
    # gen simulated
    graph = graph_forge(opt='label-graph-feat')
    # get adj list for Deepwalk input
    adjlist = [str(k) + '\t' + '\t'.join(map(str, v.keys())) for k, v in
               from_scipy_sparse_matrix(graph[0]).adj.iteritems()]
    f = open('adjlist.dw', 'w')
    f.write('\n'.join(adjlist))

    # pk.dump(graph, open('graph.dat', 'wb'))
    # graph = pk.load(open('/home/danny/PycharmProjects/gcn/gcn/graph.dat', 'rb'))

    # deep walk
    main()

    # read label
    label = (graph[2] + graph[3] + graph[4]).argmax(1)

    # read X
    x_file = open('kerate.emb', 'r')
    x = {}
    for line in x_file.readlines()[1:]:
        val = map(float, line.split())
        x[val[0]] = val[1:]
    feat = collections.OrderedDict(sorted(x.items())).values()

    # shuffle and split 9:1
    feat_list = np.array(feat)
    raw_labels = np.array(label)
    data_len = len(label)
    reorder = np.array(list(xrange(data_len)))
    old_order = copy.deepcopy(reorder)
    np.random.shuffle(reorder)
    feat_list[old_order, :] = feat_list[reorder, :]
    raw_labels[old_order] = raw_labels[reorder]
    train_x = feat_list[:data_len * 9 / 10, :]
    train_y = raw_labels[:data_len * 9 / 10]
    test_x = feat_list[data_len * 9 / 10:, :]
    test_y = raw_labels[data_len * 9 / 10:]

    clf = SVC()
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)

    evaluate_roc(test_y, pred_y, clf.score(test_x, test_y))
