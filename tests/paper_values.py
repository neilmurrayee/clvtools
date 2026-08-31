"""Numbers printed in Meierer, Bachmann, Naef, Schilter & Algesheimer,
"Estimating Individual Customer Lifetime Values with R: The CLVTools Package"
(Journal of Statistical Software, submission 5634), in one place.

Kept out of ``conftest.py`` so that test modules can import them normally
rather than relying on pytest's rootdir insertion.

Section references are to the LaTeX source in ``arXiv-2602.09845v1/``.
"""

from __future__ import annotations

# -- The case study setup, S6.1 -----------------------------------------------

#: apparelTrans is a single acquisition cohort: 600 customers whose first
#: purchase was on 2005-01-02.
N_CUSTOMERS = 600
N_TRANSACTIONS = 3187
COHORT_START = "2005-01-02"
DATA_END = "2010-12-20"

#: ``estimation.split = 104`` -- 104 weeks from the first purchase record.
ESTIMATION_WEEKS = 104
TIME_UNIT = "week"
ESTIMATION_END = "2006-12-31"

# -- summary(clv.apparel), S6.1.2 ---------------------------------------------
#
# The descriptive table the paper prints, to the three decimals it prints it
# to. "-" is a statistic that does not apply to that sample.

DESCRIPTIVES = {
    "Period Start": ("2005-01-02", "2007-01-01", "2005-01-02"),
    "Period End": ("2006-12-31", "2010-12-20", "2010-12-20"),
    "Number of customers": (None, None, 600),
    "First Transaction in period": ("2005-01-02", "2007-01-01", "2005-01-02"),
    "Last Transaction in period": ("2006-12-31", "2010-12-20", "2010-12-20"),
    "Total # Transactions": (1866, 1317, 3183),
    "Mean # Transactions per cust": (3.110, 5.557, 5.305),
    "(SD) # Transactions": (2.714, 5.123, 6.119),
    "Mean Spending per Transaction": (40.545, 36.977, 39.069),
    "(SD) Spending": (73.362, 55.356, 66.519),
    "Total Spending": (75657.730, 48699.170, 124356.900),
    "Total # zero repeaters": (213, None, None),
    "Percentage of zero repeaters": (35.500, None, None),
    "Mean Interpurchase time": (24.823, 30.604, 37.817),
    "(SD) Interpurchase time": (19.417, 24.756, 42.339),
}

#: S6.1.2: "the dataset includes a total of 600 customers who made 3'183
#: purchases", four fewer than the 3,187 records, because same-day purchases by
#: one customer are one transaction.
N_TRANSACTIONS_AGGREGATED = 3183

#: S6.1.2's frequency plot, and the zero-repeater count above.
ZERO_REPEATERS = 213

# -- Pareto/NBD without covariates, S6.2.1 ------------------------------------

PNBD_MLE = {"r": 1.4490, "alpha": 48.6361, "s": 0.5613, "beta": 46.8844}

#: "an average purchase rate of r/alpha = 0.030 transactions and an average
#: attrition rate of s/beta = 0.012".
PNBD_MEAN_PURCHASE_RATE = 0.030
PNBD_MEAN_ATTRITION_RATE = 0.012

# -- Gamma-Gamma spending model, S6.2.3 ---------------------------------------

GG_MLE = {"p": 3.099, "q": 5.654, "gamma": 56.504}

# -- Holdout evaluation of the combined prediction, S6.3.1 --------------------

HOLDOUT_ERRORS = {
    "mae.cet": 2.039532,
    "rmse.cet": 3.329395,
    "mae.total.spending": 87.64222,
    "rmse.total.spending": 182.38,
}

# -- Final prediction on the full data, S6.3.2 --------------------------------
#
# est.pnbd.full / est.gg.full fitted with estimation.split = NULL, then
# predict(prediction.end = 95, continuous.discount.factor = log(1.075)/52).

PREDICTION_WEEKS = 95
DISCOUNT_RATE_ANNUAL = 0.075
PREDICTION_PERIOD_FIRST = "2010-12-21"
PREDICTION_PERIOD_LAST = "2012-10-15"

#: ``head(dt.pred.full, 3)`` -- keyed by Id.
PREDICT_FULL_HEAD = {
    "1": {
        "PAlive": 0.007191623,
        "CET": 0.01300226,
        "DERT": 0.06200625,
        "predicted.mean.spending": 77.79363,
        "predicted.period.spending": 1.011493,
        "predicted.CLV": 4.823691,
    },
    "10": {
        "PAlive": 0.836860928,
        "CET": 0.89770449,
        "DERT": 4.28104733,
        "predicted.mean.spending": 36.04491,
        "predicted.period.spending": 32.357674,
        "predicted.CLV": 154.309950,
    },
    "100": {
        "PAlive": 0.922281780,
        "CET": 2.34558536,
        "DERT": 11.18582123,
        "predicted.mean.spending": 37.23417,
        "predicted.period.spending": 87.335919,
        "predicted.CLV": 416.494747,
    },
}

# -- Prospective ("new") customers, S6.3.4 ------------------------------------

NEWCUSTOMER_PERIODS = 52
NEWCUSTOMER_TRANSACTIONS = 2.218635
NEWCUSTOMER_SPENDING = 39.1372
NEWCUSTOMER_TOTAL = 86.83115

# -- Pareto/NBD with time-varying covariates, S6.4.2 --------------------------
#
# head(dt.pred.mixed.future, 3), from a fit on the full data with the covariate
# series extended by apparelDynCovFuture.
#
# These are NOT reproduced. CLVTools 0.12.1 no longer reaches the fit the paper
# printed: its own predict() gives 0.0107292 where the paper prints 0.0139206
# for customer 1. This package matches CLVTools 0.12.1 to 1e-12 at that fit --
# see tests/test_pnbd_dyncov_predict.py, which pins both the agreement and the
# gap.

DYNCOV_FUTURE_PAPER = {
    "1": {"PAlive": 0.0139206, "CET": 0.0253848, "DECT": 0.02379146},
    "10": {"PAlive": 0.8108995, "CET": 1.5938786, "DECT": 1.49351918},
    "100": {"PAlive": 0.9103230, "CET": 4.0419238, "DECT": 3.78742184},
}

# -- Pareto/NBD with time-invariant covariates, S6.4.1 ------------------------
#
# latentAttrition(~ Gender + Channel | Gender + Channel, family = pnbd,
#                 data = clv.static)

PNBD_STATIC_MLE = {
    "r": 1.8378,
    "alpha": 92.9123,
    "s": 0.5920,
    "beta": 49.6227,
    "life.Gender": -0.6430,
    "life.Channel": 0.7907,
    "trans.Gender": 0.2859,
    "trans.Channel": 0.6241,
}

PNBD_STATIC_SE = {
    "r": 0.3455,
    "alpha": 16.9670,
    "s": 0.2609,
    "beta": 36.2509,
    "life.Gender": 0.2955,
    "life.Channel": 0.3059,
    "trans.Gender": 0.1041,
    "trans.Channel": 0.1050,
}

PNBD_STATIC_Z = {
    "life.Gender": -2.176,
    "life.Channel": 2.585,
    "trans.Gender": 2.745,
    "trans.Channel": 5.946,
}

PNBD_STATIC_LL = -5821.0627
PNBD_STATIC_AIC = 11658.1254
PNBD_STATIC_BIC = 11693.3009
