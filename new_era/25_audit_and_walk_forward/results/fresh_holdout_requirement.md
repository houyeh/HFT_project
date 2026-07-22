# Fresh Holdout Requirement

The 20260127 test set is burned and is not used in these walk-forward folds.
Each fold trains a fold-local causal risk ranker on earlier sessions, selects rolling calibration on the next validation session, and evaluates on the next unseen test session.
Because only three pre-burned fresh folds are available, any positive result should be treated as confirmation evidence, not final proof. A future truly fresh holdout remains required before advisor-facing profitability claims.
