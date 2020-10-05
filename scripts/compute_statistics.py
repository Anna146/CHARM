import numpy as np
from collections import defaultdict

def compute_multilabel_precision_k(filepath, k=5, offset=0):
    with open(filepath, "r") as f_in:
        correct = 0
        total = 0
        for line in f_in:
            total += 1
            fields = line.split("\t")
            answ = eval(fields[0 + offset])
            curr_correct = 0
            for i in enumerate(fields[1+offset:]):
                if i[0] >= k:
                    break
                if int(i[1].split(",")[0][1:]) in answ:
                    curr_correct += 1
                    break
            correct += (curr_correct * 1.0 / k)
    return correct * 1.0 / total

def compute_multilabel_hit_k(filepath, k=5, offset=0, with_dict = False):
    user_dict = dict()
    with open(filepath, "r") as f_in:
        correct = 0
        total = 0
        for line in f_in:
            fields = line.split("\t")
            answ = eval(fields[0 + offset])
            curr_correct = 0
            for i in enumerate(fields[1+offset:]):
                if i[0] >= k:
                    break
                if int(i[1].split(",")[0][1:]) in answ:
                    curr_correct += 1
                    break
            correct += curr_correct
            user_dict[total] = curr_correct
            total += 1
    if with_dict:
        return correct * 1.0 / total, user_dict
    return correct * 1.0 / total

def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def compute_multilabel_ndcg_macro(filepath, offset = 0, at = 10):
    all_scores = 0
    ndcgs_dict = defaultdict(float)
    count_dict = defaultdict(int)
    with open(filepath, "r") as f_in:
        for line in f_in:
            ranks = []
            fields = line.split("\t")
            answ = eval(fields[0 + offset])
            for i in fields[1 + offset:]:
                pred_num = int(i.split(",")[0][1:])
                if pred_num in answ:
                    ranks.append(1)
                else:
                    ranks.append(0)
            for an in answ:
                ndcgs_dict[an] += ndcg_at_k(ranks, at)
                count_dict[an] += 1
    for k, v in ndcgs_dict.items():
        all_scores += v * 1.0 / count_dict[k]
    return all_scores / len(ndcgs_dict)

def compute_multilabel_ndcg_micro(filepath, offset = 0, at = 10, with_dict = False):
    all_scores = 0
    count = 0
    user_scores = dict()
    with open(filepath, "r") as f_in:
        for line in f_in:
            ranks = []
            fields = line.split("\t")
            answ = eval(fields[0 + offset])
            for i in fields[1 + offset:]:
                pred_num = int(i.split(",")[0][1:])
                if pred_num in answ:
                    ranks.append(1)
                else:
                    ranks.append(0)
            ndcg = ndcg_at_k(ranks, at)
            all_scores += ndcg
            user_scores[count] = ndcg
            count += 1
    if with_dict:
        return all_scores / count, user_scores
    return all_scores / count

def compute_multilabel_MRR_macro(filepath, offset = 0, with_dict = False):
    prof_dict = defaultdict(lambda: [0.0,0])
    big_count = 0
    big_MRR = 0
    user_scores = []
    with open(filepath, "r") as f_in:
        for line in f_in:
            fields = line.split("\t")
            answ = eval(fields[0 + offset])
            user_scores.append(0)
            for an in answ:
                for i in enumerate(fields[1 + offset:]):
                    pred_num = int(i[1].split(",")[0][1:])
                    if pred_num == an:
                        score = 1.0 / (i[0] + 1)
                        prof_dict[an][0] += score
                        prof_dict[an][1] += 1
                        user_scores[-1] += score
                        break
            user_scores[-1] /= len(answ)
        all_mrrs = dict()
        for prof, stats in prof_dict.items():
            all_mrrs[prof] = stats[0] / stats[1]
            big_count += 1
            big_MRR += float(stats[0] / stats[1])
    if not with_dict:
        return big_MRR / big_count
    return big_MRR / big_count, all_mrrs#, user_scores


def compute_whatever_stats(output_file, with_dict = False):
    stats = dict()
    if not with_dict:
        stats["ndcg_mic"] = round(compute_multilabel_ndcg_micro(output_file, 1, 1000), 2)
        stats["hit@3"] = round(compute_multilabel_hit_k(output_file, 3, 1), 2)
        stats["hit@5"] = round(compute_multilabel_hit_k(output_file, 5, 1), 2)
        stats["mrr_macro"] = round(compute_multilabel_MRR_macro(output_file, 1, with_dict=with_dict), 2)
    if with_dict:
        dicts_dict = dict()
        stats["ndcg_mic"], dicts_dict["ndcg_mic"]  = compute_multilabel_ndcg_micro(output_file, 1, 1000, with_dict=with_dict)
        stats["hit@3"], dicts_dict["hit@3"] = compute_multilabel_hit_k(output_file, 3, 1, with_dict=with_dict)
        stats["hit@5"], dicts_dict["hit@5"] = compute_multilabel_hit_k(output_file, 5, 1, with_dict=with_dict)
        stats["mrr_macro"], dicts_dict["mrr_macro"] = compute_multilabel_MRR_macro(output_file, 1, with_dict=with_dict)
        return stats, dicts_dict
    return stats