#coding=utf-8

def f1(scores, labels, threshold=0.5):
    tp, tn, fn, fp = 0,0,0,0
    pos_num, neg_num = 0,0
    for i in range(len(labels)):
        if scores[i] > threshold:
            if labels[i] > 0.5:
                pos_num += 1
                tp += 1
            else:
                neg_num += 1
                fp += 1
        else:
            if labels[i] > 0.5:
                pos_num += 1
                fn += 1
            else:
                neg_num += 1
                tn += 1
    accurate = float(tp + tn) / (pos_num + neg_num)
    precision = float(tp) / (tp + fp) if tp + fp != 0 else 0
    recall = float(tp) / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return [accurate, precision, recall, f1]

def auc(scores, labels, threshold=0.5):
    label_scores = zip(labels, scores)
    label_scores.sort(key=lambda x:x[1])

    negative_sums = [0] * len(label_scores)
    for i in range(len(label_scores)):
        if label_scores[i][0] < 0.5:
            negative_sums[i] = negative_sums[i-1] + 1
        else:
            negative_sums[i] = negative_sums[i-1]

    sum_, sum_neg, sum_pos = 0, 0, 0
    for i in range(len(label_scores)):
        if int(label_scores[i][0]) == 1:
            sum_ += negative_sums[i]
            sum_pos += 1
        else:
            sum_neg += 1

    return sum_ * 1.0 / (sum_neg * sum_pos)

def test():
    scores = [-1, 2, 3, 4]
    labels = [0, 0, 0 ,1]
    return auc(scores, labels)
    
if __name__ == '__main__':
    print test()
    

