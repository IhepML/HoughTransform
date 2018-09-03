# coding: utf-8
'''
import sys
sys.path.insert(0, '.')
'''
from GetHfEachEvent import GetHfEachEvent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

LOCAL_COLUMNS = ['mdetmt0', 'me']
NEIGH_COLUMNS = ['ldetmt0', 'rdetmt0', 'le', 're']


def main():
    train1_sig = pd.read_csv('../data/train1_sig.csv')
    train2_sig = pd.read_csv('../data/train2_sig.csv')
    test_sig = pd.read_csv('../data/test_sig.csv')

    train1_bak = pd.read_csv('../data/train1_bak.csv')
    train2_bak = pd.read_csv('../data/train2_bak.csv')
    test_bak = pd.read_csv('../data/test_bak.csv')

    i = 0
    color = ['r', 'b', 'g']
    lab = ['local', 'local+neighbor', 'local+neigh+hough']
    title = ['local', 'local neighbor', 'local neigh hough']
    y_pred_test = [0] * 3
    y_pred_proba_test = [0] * 3
    acu_scr = [0] * 3
    roc_auc = [0] * 3

    # prepare samples
    y_train2_samp = np.concatenate((train2_sig['isSignal'],
                                   train2_bak['isSignal']), axis=0)
    y_test_samp = np.concatenate((test_sig['isSignal'],
                                  test_bak['isSignal']), axis=0)
    y_train1_samp = np.concatenate((train1_sig['isSignal'],
                                    train1_bak['isSignal']), axis=0)

    for columns in (LOCAL_COLUMNS, LOCAL_COLUMNS + NEIGH_COLUMNS):
        x_train2_samp = np.concatenate((train2_sig[columns],
                                       train2_bak[columns]), axis=0)

        x_test_samp = np.concatenate((test_sig[columns],
                                      test_bak[columns]), axis=0)

        x_train1_samp = np.concatenate((train1_sig[columns],
                                        train1_bak[columns]), axis=0)

        x = np.concatenate((x_train1_samp, x_train2_samp), axis=0)
        y = np.concatenate((y_train1_samp, y_train2_samp), axis=0)
        gbm0 = GradientBoostingClassifier(random_state=10, learning_rate=0.1, n_estimators=80, min_samples_leaf=20,
                                          max_features='sqrt', subsample=0.8, max_depth=13, min_samples_split=500)
        gbm0.fit(x, y)

        y_pred_test[i] = gbm0.predict(x_test_samp)
        y_pred_proba_test[i] = gbm0.predict_proba(x_test_samp)[:, 1]

        acu_scr[i] = metrics.accuracy_score(y_test_samp, y_pred_test[i])
        roc_auc[i] = metrics.roc_auc_score(y_test_samp, y_pred_proba_test[i])

        i += 1

    gbm2 = GradientBoostingClassifier(random_state=10, learning_rate=0.1, n_estimators=80, min_samples_leaf=20,
                                      max_features='sqrt', subsample=0.8, max_depth=13, min_samples_split=500)
    gbm2.fit(x_train1_samp, y_train1_samp)
    y_pred_proba_train = gbm2.predict_proba(x_train2_samp)[:, 1]
    y_pred_proba_test_tem = gbm2.predict_proba(x_test_samp)[:, 1]

    x_train_hf = GetHfEachEvent(train2_sig, train2_bak, y_pred_proba_train).hf_scr
    x_train_hf = np.transpose([x_train_hf])
    x_train2_samp = np.concatenate((x_train2_samp, x_train_hf), axis=1)

    x_test_hf = GetHfEachEvent(test_sig, test_bak, y_pred_proba_test_tem).hf_scr
    x_test_hf = np.transpose([x_test_hf])
    x_test_samp = np.concatenate((x_test_samp, x_test_hf), axis=1)

    gbm1 = GradientBoostingClassifier(random_state=10, learning_rate=0.1, n_estimators=80, min_samples_leaf=20,
                                      max_features='sqrt', subsample=0.8, max_depth=13, min_samples_split=500)
    gbm1.fit(x_train2_samp, y_train2_samp)

    y_pred_test[2] = gbm1.predict(x_test_samp)
    y_pred_proba_test[2] = gbm1.predict_proba(x_test_samp)[:, 1]
    acu_scr[2] = metrics.accuracy_score(y_test_samp, y_pred_test[2])
    roc_auc[2] = metrics.roc_auc_score(y_test_samp, y_pred_proba_test[2])

    # plot histogram
    for i in range(3):
        print("accuracy: %.4g" % acu_scr[i])
        print("Pro_Accuracy: %.4g" % roc_auc[i])
        plt.hist(y_pred_proba_test[i][:len(test_sig)] * 100, 50, density=True, alpha=0.5, label="signals")
        plt.hist(y_pred_proba_test[i][len(test_sig):] * 100, 50, density=True, alpha=0.5, label="backgrounds")
        plt.legend(prop={'size': 10})
        plt.xlabel("probability")
        plt.ylabel("rate")
        plt.title('use %s features' % title[i])
        plt.show()

    # plot fpr and tpr
    for i in range(3):
        bak_ret, sig_ret, thresholds = roc_curve(y_test_samp, y_pred_proba_test[i])
        roc_auc = auc(bak_ret, sig_ret)
        bak_rej = 1 - bak_ret
        if i == 0:
            plt.figure(figsize=(10, 6))
        plt.plot(sig_ret, bak_rej, color[i], label='%s AUC = %0.4f' % (lab[i], roc_auc))
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax.yaxis.grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)
    plt.legend(loc='lower left')
    plt.plot([0.997, 0.997], [0, 1], 'y--')
    plt.plot([0.99, 0.99], [0, 1], 'g--')
    plt.xlim([0.98, 1])
    plt.ylim([0.7, 1])
    plt.xlabel('signal retention efficiency')
    plt.ylabel('backgrounds rejection efficiency')
    plt.show()

    seq1 = np.where(abs(sig_ret - 0.99) < 10 ** (-4))
    print('sig.ret = 0.99,bak.rej = %.3f' % bak_rej[seq1[0][0]])
    seq2 = np.where(abs(sig_ret - 0.997) < 10 ** (-4))
    print('sig.ret = 0.997,bak.rej = %.3f' % bak_rej[seq2[0][0]])


if __name__ == '__main__':
    main()
