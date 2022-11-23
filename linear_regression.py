import numpy as np
from sklearn import linear_model
import os
from os.path import join
import json

# 数据文件夹是本py文件的父目录
datasets_dir_path = os.path.dirname(__file__)   #  当前py文件的绝对路径  除去本py文件名，即得到了父目录
linear_weight_save_path = join(datasets_dir_path, 'linear_weight.json')
delete_metric = ['nll', 'nce', 'ppl', 'USL-HS']  #USL-H中有很多sub_metric都打分了，但需要去除一些非norm(部分sub_metirc)的打分
saved_metirc = ['vup', 'nup', 'norm_nll', 'norm_nce', 'norm_ppl'] # 到底有哪些metric参与到了集成模型

quality_record = {}


def stdError_func(y_test, y):
    return np.sqrt(np.mean((y_test - y) ** 2))


def R2_1_func(y_test, y):
    return 1 - ((y_test - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()


def R2_2_func(y_test, y):
    y_mean = np.array(y)
    y_mean[:] = y.mean()
    return 1 - stdError_func(y_test, y) / stdError_func(y_mean, y)


def load_model_score(json_path):  # 用的是usl_hs目录，里面的每一个sub_metric都对所有数据集打好了分
    with open(json_path, 'r')as f:
        json_scores = json.load(f)  # 一定返回的是一个列表
    keys = json_scores[0].keys()
    usl_scores = {}
    for key in keys:
        if key in delete_metric:
            continue
        usl_scores[key] = []  # eg: usl_scores["vup"] = []  ,存储这个数据集所有样本的vup打分
    for score in json_scores:
        for k in keys:
            if k in delete_metric:
                continue
            usl_scores[k].append(score[k])
    return usl_scores

#  这个函数很重要，
#  其实可以在传参列表传入scores_dict = {}，这样可以将所有数据集整合起来，对quality进行分类全部放到这个字典中。利用type.txt，之后进行更多数据的训练
def load_human_scores(json_path):   #用的是dstc_eval这个目录的 all_anno.json
    with open(json_path, 'r')as f:
        all_samples = json.load(f)  # 打开文件后要变成json格式，需要load
    keys = all_samples[0].keys() # 拿到所有当前数据集的quality
    scores_dict = {}   #scores_dict[k]=[]    代表这个数据集的的quality=k时，的所有标注分列表
    for key in keys:
        scores_dict[key] = [] # 先手动创建这些列表

    for sample in all_samples:  #拿到每一个样本
        for k in keys: # 对这个样本的不同quality分数都放到不同的scores_dict[k]=[]中
            scores_dict[k].append(sum(sample[k]) / len(sample[k]))  #因为有多个人类标注，所以需要平均
    return scores_dict


def load_json(load_path):
    with open(load_path, 'r')as f:
        json_lines = json.load(f)
    return json_lines


def linear_regr(model_scores_dict, human_score, quality):
    score_lst = []
    for k in model_scores_dict.keys():
        score_lst.append(model_scores_dict[k])  # 构建二维数组，用于输入。  k_sum*sample_sum
    x_scores = np.array(score_lst).T    #  画个图就知道了 ，需要矩阵的转置才行。
    y_score = np.array(human_score) # sample_num*1
    cft = linear_model.LinearRegression()     # 输入是 sample_sum * k_sum  *(模型权重：k_sum*1) = 输出是 sample_sum*1
    cft.fit(x_scores, y_score)  #

    # print("model coefficients", cft.coef_)
    # print("model intercept", cft.intercept_)

    predict_y = cft.predict(x_scores)
    strError = stdError_func(predict_y, y_score)
    R2_1 = R2_1_func(predict_y, y_score)
    R2_2 = R2_2_func(predict_y, y_score)
    # sklearn中自带的模型评估，与R2_1逻辑相同
    score = cft.score(x_scores, y_score)

    # print('strError={:.2f}, R2_1={:.2f},  R2_2={:.2f}, clf.score={:.2f}'.format(
    #     strError, R2_1, R2_2, score))
    # 记录每个metric的权重系数
    if quality not in quality_record.keys():
        quality_record[quality] = []
    # 这里其实就是所有数据集的，只要是对于所有针对quality=k的线性模型都保存线性模型，进而之后可以拟合所有线性模型
    #···· 如dailydialog-gupta 的overall 和 dailydialog-zhao 都有quality==overall,那么应该全部收集在一起，之后拟合成一条直线
    quality_record[quality].append(cft.coef_.tolist())

    return cft.coef_.tolist()  #返回线性回归的权重列表


# 将来自多个数据集中同一quality的权重系数求均值，作为该quality单一的系数
def get_single_quality_weight(record, metric_names):
    for k in record.keys(): #拿到某个quality
        record[k] = np.mean(np.array(record[k]), axis=0).tolist()   # 二维numpy数组求均值，并转化为1维
        record[k] = [round(i, 2) for i in record[k]]
        metric_dict = {}
        for cnt, name in enumerate(metric_names):
            metric_dict[name] = max(record[k][cnt], 0)
        record[k] = metric_dict

    return record


def save_json(lst, save_path):
    with open(save_path, 'w')as f:
        json.dump(lst, f)


def load_dialog_model_score(score_json_path, eval_json_path, data_name):
    with open(score_json_path, 'r')as f:
        json_scores = json.load(f)
    turn_scores = {}   #把dialog-level的数据集的所有dialog样本拆成一个个turn-level后的打分情况
    sub_metric_names = [i for i in list(json_scores[0].keys()) if i not in delete_metric]
    for sub_metric in sub_metric_names:
        turn_scores[sub_metric] = []

    for score in json_scores:
        for sub_metric in sub_metric_names:
            turn_scores[sub_metric].append(score[sub_metric])

    dialog_scores = {}  #dialog-level的数据集的所有dialog样本【一个个turn-level分数的整合情况】后的打分情况
    for sub_metric in sub_metric_names:
        dialog_scores[sub_metric] = []

    with open(eval_json_path, 'r')as f:
        json_data = json.load(f)

    for k in sub_metric_names:
        left_index = 0
        for line in json_data:
            dialog = line['dialog']   # 拿到当前样本的dialog
            cur_turn_cnt = int(len(dialog) / 2)
            if data_name == 'persona-see' and dialog[0]['speaker'] == 'model':
                continue
            cur_turn_score = turn_scores[k][left_index:left_index + cur_turn_cnt]
            left_index += cur_turn_cnt  # 更新下一个dialog-level样本的起始游标
            mean_score = sum(cur_turn_score) / len(cur_turn_score)  #  对turn-level的分数取平均等于 当前dialog的score
            dialog_scores[k].append(mean_score)
    return dialog_scores


def load_dialog_human_scores(json_path):
    with open(json_path, 'r')as f:
        anno_json = json.load(f)
    keys = anno_json[0].keys()
    scores_dict = {}
    for key in keys:
        scores_dict[key] = []  #scores_dict[k]=[]    代表这个数据集的quality=k时，的所有样本的标注分列表
    for anno in anno_json:
        for k in keys:
            if 'Error recovery' in k:
                if len(anno[k]) == 0:
                    scores_dict[k].append((False, 0))
                else:
                    scores_dict[k].append((True, sum(anno[k]) / len(anno[k])))
            else:
                scores_dict[k].append((True, sum(anno[k]) / len(anno[k])))
            # scores_dict[k].append(sum(anno[k]) / len(anno[k]))
    return scores_dict


if __name__ == '__main__':
    turn_names = ['dailydialog-gupta', 'dailydialog-zhao', 'persona-usr',
                  'persona-zhao', 'topical-usr', 'fed-turn',
                  'convai2-grade', 'empathetic-grade', 'dailydialog-grade',
                  'dstc6', 'dstc7', 'humod'
                  ]
    dialog_names = ['persona-see', 'fed-dial']

    model_dir_path = join(datasets_dir_path, 'usl_s')  #

    ############################### turn-level级别的训练
    for cnt, name in enumerate(turn_names):
        model_score_path = join(model_dir_path, "%s_score.json" % name) #usl-h在某个数据集name的的所有sub_metric的打分
        if not os.path.exists(model_score_path):
            continue
        human_anno_path = join(datasets_dir_path, 'dstc_eval', "%s_all_anno.json" % name)  # usl-h在某个数据集的的所有人类标注
        model_scores = load_model_score(model_score_path) #返回的是一个字典，key是sub_metic，value是[],代表sub_metirc的每一个样本metric打分
        human_scores = load_human_scores(human_anno_path) #返回的是一个字典，key是quality,value是[],代表quality的每一个样本标注分
        spear_lst = []
        print('dataset%d:' % (cnt + 1), name)
        for k, v in human_scores.items():  #k,v分别代表当前数据集的quality，，，和 人类标注分
            if name == 'humod': v = v[:-1]
            weights = linear_regr(model_scores, v, k)
            print(k, weights)

    ############################### dialog-level级别的训练

    for cnt, name in enumerate(dialog_names):
        model_score_path = join(model_dir_path, "%s_score.json" % name) # USL-H的sub_metric(即模型)的打分，Metric模型打分
        if not os.path.exists(model_score_path):
            continue
        human_anno_path = join(datasets_dir_path, 'dstc_eval', "%s_all_anno.json" % name) # 人类标注打分

        data_evaL_path = join(datasets_dir_path, 'human_evaluation_data', '%s_eval.json' % name) # 原始数据（dialog + human annotation）

        human_scores = load_dialog_human_scores(human_anno_path)
        #human_scores['overall'] = [1，2，3，4，3，4，5],所有dialog样本在quality==overall的人类打分
        model_scores = load_dialog_model_score(model_score_path, data_evaL_path, name)
        #model_scores['vup'] = [1，2，3，4，3，4，5]，所有dialog样本通过sub_metric=='VUP'的Metric模型打分

        print('dataset%d:' % (cnt + len(turn_names) + 1), name)
        for k, v in human_scores.items():
            new_model_scores = {}
            for ki in model_scores:
                new_model_scores[ki] = []
            new_v = []
            for idx, x in enumerate(v):
                if x[0]:
                    new_v.append(x[1])
                    # if k in quality_metric_dict.keys():
                    for mk in model_scores.keys():
                        new_model_scores[mk].append(model_scores[mk][idx])
                    # else:
                    #     s2.append(model_scores['USL-HS'][idx])
            # mix_model_score = get_mix_model_score(new_model_scores, new_v, k)
            weights = linear_regr(new_model_scores, new_v, k)
            print(k, weights)
    ############################################################
    avg_quality_weight = get_single_quality_weight(quality_record, saved_metirc)
    save_json(avg_quality_weight, linear_weight_save_path)
