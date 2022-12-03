import numpy as np
from sklearn import linear_model
import os
from os.path import join
import json


datasets_dir_path = os.path.dirname(__file__)  
linear_weight_save_path = join(datasets_dir_path, 'linear_weight.json')

quality_record = {}


def stdError_func(y_test, y):
    return np.sqrt(np.mean((y_test - y) ** 2))


def R2_1_func(y_test, y):
    return 1 - ((y_test - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()


def R2_2_func(y_test, y):
    y_mean = np.array(y)
    y_mean[:] = y.mean()
    return 1 - stdError_func(y_test, y) / stdError_func(y_mean, y)


def load_model_score(json_path): 
    with open(json_path, 'r')as f:
        json_scores = json.load(f) 
    keys = json_scores[0].keys()
    usl_scores = {}
    for key in keys:
        if key in delete_metric:
            continue
        usl_scores[key] = []  # eg: usl_scores["vup"] = [] 
    for score in json_scores:
        for k in keys:
            if k in delete_metric:
                continue
            usl_scores[k].append(score[k])
    return usl_scores


def load_human_scores(json_path):  
    with open(json_path, 'r')as f:
        all_samples = json.load(f)  
    keys = all_samples[0].keys() 
    scores_dict = {}  
    for key in keys:
        scores_dict[key] = [] 

    for sample in all_samples:  
        for k in keys: 
            scores_dict[k].append(sum(sample[k]) / len(sample[k]))  
    return scores_dict


def load_json(load_path):
    with open(load_path, 'r')as f:
        json_lines = json.load(f)
    return json_lines


def linear_regr(model_scores_dict, human_score, quality):
    score_lst = []
    for k in model_scores_dict.keys():
        score_lst.append(model_scores_dict[k])  
    x_scores = np.array(score_lst).T   
    y_score = np.array(human_score) 
    cft = linear_model.LinearRegression()    
    cft.fit(x_scores, y_score)  #

    predict_y = cft.predict(x_scores)
    strError = stdError_func(predict_y, y_score)
    R2_1 = R2_1_func(predict_y, y_score)
    R2_2 = R2_2_func(predict_y, y_score)
    score = cft.score(x_scores, y_score)

    if quality not in quality_record.keys():
        quality_record[quality] = []
 
    quality_record[quality].append(cft.coef_.tolist())

    return cft.coef_.tolist()  



def get_single_quality_weight(record, metric_names):
    for k in record.keys():
        record[k] = np.mean(np.array(record[k]), axis=0).tolist() 
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
    turn_scores = {}  
    sub_metric_names = [i for i in list(json_scores[0].keys()) if i not in delete_metric]
    for sub_metric in sub_metric_names:
        turn_scores[sub_metric] = []

    for score in json_scores:
        for sub_metric in sub_metric_names:
            turn_scores[sub_metric].append(score[sub_metric])

    dialog_scores = {}  
    for sub_metric in sub_metric_names:
        dialog_scores[sub_metric] = []

    with open(eval_json_path, 'r')as f:
        json_data = json.load(f)

    for k in sub_metric_names:
        left_index = 0
        for line in json_data:
            dialog = line['dialog']   
            cur_turn_cnt = int(len(dialog) / 2)
            if data_name == 'persona-see' and dialog[0]['speaker'] == 'model':
                continue
            cur_turn_score = turn_scores[k][left_index:left_index + cur_turn_cnt]
            left_index += cur_turn_cnt  
            mean_score = sum(cur_turn_score) / len(cur_turn_score)  
            dialog_scores[k].append(mean_score)
    return dialog_scores


def load_dialog_human_scores(json_path):
    with open(json_path, 'r')as f:
        anno_json = json.load(f)
    keys = anno_json[0].keys()
    scores_dict = {}
    for key in keys:
        scores_dict[key] = []  
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

    model_dir_path = join(datasets_dir_path, 'usl_s')  

    # turn-level
    for cnt, name in enumerate(turn_names):
        model_score_path = join(model_dir_path, "%s_score.json" % name) 
        if not os.path.exists(model_score_path):
            continue
        human_anno_path = join(datasets_dir_path, 'dstc_eval', "%s_all_anno.json" % name) 
        model_scores = load_model_score(model_score_path) 
        human_scores = load_human_scores(human_anno_path) 
        spear_lst = []
        print('dataset%d:' % (cnt + 1), name)
        for k, v in human_scores.items(): 
            if name == 'humod': v = v[:-1]
            weights = linear_regr(model_scores, v, k)
            print(k, weights)

    #dialog-level

    for cnt, name in enumerate(dialog_names):
        model_score_path = join(model_dir_path, "%s_score.json" % name) 
        if not os.path.exists(model_score_path):
            continue
        human_anno_path = join(datasets_dir_path, 'dstc_eval', "%s_all_anno.json" % name) 

        data_evaL_path = join(datasets_dir_path, 'human_evaluation_data', '%s_eval.json' % name) 

        human_scores = load_dialog_human_scores(human_anno_path)
        model_scores = load_dialog_model_score(model_score_path, data_evaL_path, name)
        print('dataset%d:' % (cnt + len(turn_names) + 1), name)
        for k, v in human_scores.items():
            new_model_scores = {}
            for ki in model_scores:
                new_model_scores[ki] = []
            new_v = []
            for idx, x in enumerate(v):
                if x[0]:
                    new_v.append(x[1])              
                    for mk in model_scores.keys():
                        new_model_scores[mk].append(model_scores[mk][idx])
                
            weights = linear_regr(new_model_scores, new_v, k)
            print(k, weights)

