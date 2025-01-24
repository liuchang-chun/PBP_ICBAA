import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from sklearn.metrics import roc_auc_score, roc_curve
from codes.first_layer_feature import one_hot, to_one_hot
from codes.first_layer_feature import Binary, to_properties_code_1, to_C2_code, to_properties_density_code

from codes.model import  build_model_2





def read_fasta(fasta_file_name):
    seqs = []
    seqs_num = 0
    file = open(fasta_file_name)

    for line in file.readlines():
        if line.strip() == '':
            continue

        if line.startswith('>'):
            seqs_num = seqs_num + 1
            continue
        else:
            seq = line.strip()

            result1 = 'N' in seq
            result2 = 'n' in seq
            if result1 == False and result2 == False:
                seqs.append(seq)
    return seqs


def show_performance(y_true, y_pred):
    # 定义tp, fp, tn, fn初始值
    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] > 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] > 0.5:
                FP += 1
            else:

                TN += 1

    # 计算敏感性Sn
    Sn = TP / (TP + FN + 1e-06)
    # 计算特异性Sp
    Sp = TN / (FP + TN + 1e-06)
    # 计算Acc值
    Acc = (TP + TN) /( len(y_true) + 1e-06)
    # 计算MCC：马修斯相关系数是在混淆矩阵环境中建立二元分类器预测质量的最具信息性的单一分数
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)
    # 计算pre
    Pre = TP / (TP + FP + 1e-06)

    return Sn, Sp, Acc, MCC , Pre

def performance_mean(performance):
    print('Sn = %.4f ± %.4f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.4f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f ± %.4f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    print('Auc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))
    print('Pre = %.4f ± %.4f' % (np.mean(performance[:, 5]), np.std(performance[:, 5])))


if __name__ == '__main__':

    np.random.seed(0)
    tf.random.set_seed(1)  # for reproducibility

    # Read the training set
    train_pos_seqs = np.array(read_fasta('./data/C.jejuni/train_all_p.txt'))
    train_neg_seqs = np.array(read_fasta('./data/C.jejuni/train_all_n.txt'))
    # one-hot NCP-D
    train_seqs = np.concatenate((train_pos_seqs,train_neg_seqs ), axis=0)
    train_one_hot = np.array(to_one_hot(train_seqs)).astype(np.float32)
    to_properties_density = np.array(to_properties_density_code(train_seqs)).astype(np.float32)
    train_to_C2_code = np.array(to_C2_code(train_seqs)).astype(np.float32)
    train_2 = np.concatenate((  train_one_hot , to_properties_density ,train_to_C2_code), axis=1)


    #
    file_path = r'./data/C.jejuni/jiangwei.numpy.train.esm.feature.npy'
    train_3 = np.load(file_path)

    train_label = np.array([1] *3393+ [0] *3393).astype(np.float32)
    train_label = to_categorical(train_label, num_classes=2)

# C:\Users\22750\Desktop\iEnhancer-DCSV-main\data\M.smegmatis\C.jejuni
    # Read the testing set
    test_pos_seqs = np.array(read_fasta('./data/C.jejuni/test_p.txt'))
    test_neg_seqs = np.array(read_fasta('./data/C.jejuni/test_n.txt'))

    test_seqs = np.concatenate((test_pos_seqs, test_neg_seqs), axis=0)

    test_one_hot = np.array(to_one_hot(test_seqs)).astype(np.float32)
    test_properties_code = np.array(to_properties_density_code(test_seqs)).astype(np.float32)
    test_C2_code = np.array(to_C2_code(test_seqs)).astype(np.float32)
    test = np.concatenate(( test_one_hot, test_properties_code,test_C2_code ), axis=1)

    file_path = r'./data/C.jejuni/jiangwei.numpy.test.esm.feature.npy'
    test_3 = np.load(file_path)



    test_label = np.array([1] * 377  + [0] * 377).astype(np.float32)
    test_label = to_categorical(test_label, num_classes=2)

    # Cross-validation
    n = 10
    k_fold = KFold(n_splits=n, shuffle=True, random_state=42)
    sv_10_result = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    print(np.array(sv_10_result).shape)

    for k in range(1):
        print('*' * 30 + ' the ' + str(k) + ' cycle ' + '*' * 30)
        all_Sn = []
        all_Sp = []
        all_Acc = []
        all_MCC = []
        all_AUC = []
        all_Pre = []
        test_pred_all = []
        mean_fpr = np.linspace(0, 1, 100)
        for fold_count, (train_index, val_index) in enumerate(k_fold.split(train_2)):
            print('*' * 30 + ' fold ' + str(fold_count) + ' ' + '*' * 30)
            trains, val = train_3[train_index], train_3[val_index]
            #         trains_1, val_1 = train_1[train_index], train_1[val_index]
            trains_2, val_2 = train_2[train_index], train_2[val_index]
            trains_label, val_label = train_label[train_index], train_label[val_index]

            model = build_model_2()

            BATCH_SIZE = 30
            EPOCHS = 300

            history = model.fit(x=[trains_2, trains], y=trains_label, validation_data=([val_2, val], val_label),
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE, shuffle=True,
                                callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='auto')],
                                verbose=1)  # callbacks回调，将数据传给history

            model.summary()
            with open('./files/log_history/one_log_history.txt', 'w') as f:
                f.write(str(history.history))

            train_loss = history.history["loss"]
            train_acc = history.history["accuracy"]
            val_loss = history.history["val_loss"]
            val_acc = history.history["val_accuracy"]

            loss, accuracy = model.evaluate([val_2, val], val_label, verbose=1)

            print('val loss:', loss)
            print('val accuracy:', accuracy)

            #
            model.save(
                './models/one_P_model_' + str(fold_count) + '.h5')

            del model

            model = load_model(
                './models/one_P_model_' + str(fold_count) + '.h5')

            test_pred = model.predict([test, test_3], verbose=1)
            test_pred_all.append(test_pred[:, 1])

            # Sn, Sp, Acc, MCC, AUC  ,Pre
            Sn, Sp, Acc, MCC, Pre = show_performance(test_label[:, 1], test_pred[:, 1])
            AUC = roc_auc_score(test_label[:, 1], test_pred[:, 1])
            print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f，Pre = %f' % (Sn, Sp, Acc, MCC, AUC, Pre))

            # Put each collapsed evaluation metric into a master list

            all_Sn.append(Sn)
            all_Sp.append(Sp)
            all_Acc.append(Acc)
            all_MCC.append(MCC)
            all_AUC.append(AUC)
            all_Pre.append(Pre)
            fold_count += 1
        fold_avg_Sn = np.mean(all_Sn)
        fold_avg_Sp = np.mean(all_Sp)
        fold_avg_Acc = np.mean(all_Acc)
        fold_avg_MCC = np.mean(all_MCC)
        fold_avg_AUC = np.mean(all_AUC)
        fold_avg_Pre = np.mean(all_Pre)

        # soft voting
        test_pred_all = np.array(test_pred_all).T

        ruan_voting_test_pred = test_pred_all.mean(axis=1)

        sv_Sn, sv_Sp, sv_Acc, sv_MCC, sv_Pre = show_performance(test_label[:, 1], ruan_voting_test_pred)
        sv_AUC = roc_auc_score(test_label[:, 1], ruan_voting_test_pred)
        sv_result = [sv_Sn, sv_Sp, sv_Acc, sv_MCC, sv_AUC, sv_Pre]
        sv_10_result.append(sv_result)

        '''Mapping the ROC'''
        fpr, tpr, thresholds = roc_curve(test_label[:, 1], ruan_voting_test_pred, pos_label=1)
        # 将ROC曲线的数值保存到一个列表中
        roc_data = [fpr, tpr, thresholds]

        # 指定文件路径
        file_path = './files/ROC.txt'

        # 将ROC曲线的数值写入到文件中
        with open(file_path, 'w') as f:
            # 将fpr, tpr, thresholds转换为字符串，并用换行符分隔
            f.write('FPR: ' + str(roc_data[0]) + '\n')
            f.write('TPR: ' + str(roc_data[1]) + '\n')
            f.write('Thresholds: ' + str(roc_data[2]))

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, label='ROC cycle {} (AUC={:.4f})'.format(str(k), sv_AUC))

    print('---------------------------------------------soft voting 10---------------------------------------')
    print(np.array(sv_10_result))
    performance_mean(np.array(sv_10_result))

    '''Mapping the ROC'''
    plt.plot([0, 1], [0, 1], '--', color='red')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = np.mean(np.array(sv_10_result)[:, 4])

    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.4f)' % (mean_auc), lw=2, alpha=.8)

    plt.title('ROC Curve of Second Layer')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #
    plt.legend(loc='lower right')
    plt.savefig('...n/images/ROC_Curve_of_First_Layer.jpg', dpi=1200, bbox_inches='tight')
    plt.show()

