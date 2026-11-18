from exoplanet_vetting.config import Stage
from exoplanet_vetting.console import Status
from exoplanet_vetting.data import model_inputs
from exoplanet_vetting.synthetic import synthetic_cohort
from exoplanet_vetting.training import train_ensemble


def test_one_synthetic_baseline_epoch():
    cohort = synthetic_cohort("kepler", n=32, seed=0)
    x_train, y_train = model_inputs(cohort, cohort.train_idx, "baseline")
    x_val, y_val = model_inputs(cohort, cohort.val_idx, "baseline")
    x_test, y_test = model_inputs(cohort, cohort.test_idx, "baseline")
    lengths = {key: value for key, value in cohort.lengths().items() if value is not None}
    result = train_ensemble(
        "kepler",
        "baseline",
        lengths,
        (Stage(1, 1e-3, 16),),
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        ensemble=1,
        seed=0,
        status=Status(),
    )
    assert result.parameters == 14
    assert len(result.member_scores) == 1
    assert 0.0 <= result.ensemble.accuracy <= 1.0
