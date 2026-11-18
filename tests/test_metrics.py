import numpy as np

from exoplanet_vetting.metrics import mean_std, roc_auc, scores


def test_confusion_and_auc():
    labels = np.array([1, 1, 0, 0])
    probability = np.array([0.9, 0.2, 0.4, 0.1])
    result = scores(labels, probability)
    assert (result.tp, result.fn, result.fp, result.tn) == (1, 1, 0, 2)
    assert result.accuracy == 0.75
    assert result.sensitivity == 0.5
    assert result.specificity == 1.0
    assert roc_auc(labels, probability) == 0.75


def test_auc_is_undefined_for_one_class():
    assert np.isnan(roc_auc(np.array([1, 1, 1]), np.array([0.2, 0.4, 0.6])))


def test_single_member_deviation_is_zero():
    member = scores(np.array([1, 0]), np.array([0.8, 0.1]))
    summary = mean_std([member])
    assert summary["accuracy"][1] == 0.0
