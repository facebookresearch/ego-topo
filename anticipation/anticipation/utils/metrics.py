# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np

def topk_accuracy(scores, labels, ks, selected_class=None):
    """Computes TOP-K accuracies for different values of k
    Args:
        rankings: numpy ndarray, shape = (instance_count, label_count)
        labels: numpy ndarray, shape = (instance_count,)
        ks: tuple of integers
    Returns:
        list of float: TOP-K accuracy for each k in ks
    """
    if selected_class is not None:
        idx = labels == selected_class
        scores = scores[idx]
        labels = labels[idx]
    rankings = scores.argsort()[:, ::-1]
    # trim to max k to avoid extra computation
    maxk = np.max(ks)

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    return [tp[:, :k].max(1).mean() for k in ks]

def topk_recall(scores, labels, k=5, classes=None):
    unique = np.unique(labels)
    if classes is None:
        classes = unique
    else:
        classes = np.intersect1d(classes, unique)
    recalls = 0
    #np.zeros((scores.shape[0], scores.shape[1]))
    for c in classes:
        recalls += topk_accuracy(scores, labels, ks=(k,), selected_class=c)[0]
    return recalls/len(classes)


# def topk_recall_multiple_timesteps(preds, labels, k=5, classes=None):
#     accs = np.array([topk_recall(preds[:, t, :], labels, k, classes)
#                      for t in range(preds.shape[1])])
#     return accs.reshape(1, -1)


def tta(scores, labels):
    """Implementation of time to action curve"""
    rankings = scores.argsort()[..., ::-1]
    comparisons = rankings == labels.reshape(rankings.shape[0], 1, 1)
    cum_comparisons = np.cumsum(comparisons, 2)
    cum_comparisons = np.concatenate([cum_comparisons, np.ones(
        (cum_comparisons.shape[0], 1, cum_comparisons.shape[2]))], 1)
    time_stamps = np.array([2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0])
    return np.nanmean(time_stamps[np.argmax(cum_comparisons, 1)], 0)[4]