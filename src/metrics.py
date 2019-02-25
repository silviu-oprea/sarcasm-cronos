import numpy as np
from collections import namedtuple

import tensorflow as tf

def simple_scalar_summary(tag, simple_value):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=simple_value)
    return summary

ThreshRocMetrics = namedtuple('RocMetrics', ['precision', 'recall', 'fscore'])
ThreshRocMetric = namedtuple('ThreshRocMetric', ['thresh', 'value'])

class RocAccumulator:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.thresholds = np.array([0.05,
                                    0.1, 0.15,
                                    0.2, 0.25,
                                    0.3, 0.35,
                                    0.4, 0.45,
                                    0.5, 0.55,
                                    0.6, 0.65,
                                    0.7, 0.75,
                                    0.8, 0.85])

    def add_batch(self, targets, outputs):
        for target, dist in zip(targets, outputs):
            target_thresh = np.full(len(self.thresholds), target)
            pred_thresh = (dist[1] >= self.thresholds).astype(np.int32)

            self.tp += np.logical_and(target_thresh == 1, pred_thresh == 1)
            self.fp += np.logical_and(target_thresh == 0, pred_thresh == 1)
            self.fn += np.logical_and(target_thresh == 1, pred_thresh == 0)

    def _add_thresholds_to_metric(self, metric):
        metric_with_thresholds = dict(zip(self.thresholds, metric))
        return metric_with_thresholds

    def get_prf(self):
        precision = self.tp / (self.tp + self.fp + 1e-08)
        recall = self.tp / (self.tp + self.fn + 1e-08)
        fscore = 2 * precision * recall / (precision + recall + 1e-08)

        return ThreshRocMetrics(dict(zip(self.thresholds, precision)),
                                dict(zip(self.thresholds, recall)),
                                dict(zip(self.thresholds, fscore)))
