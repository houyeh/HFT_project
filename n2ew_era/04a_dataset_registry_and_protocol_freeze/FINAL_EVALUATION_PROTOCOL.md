# Final Evaluation Protocol

1. Model and strategy selection are allowed only in the Development pool: `NEW1` + `NEW2`.
2. `NEW3` + `NEW4` are used only as robust diagnostic OOS checks after the selected configuration is frozen.
3. `NEW5` is a one-shot final holdout. It remains locked until final development is complete.
4. No strategy may be tuned on `NEW3`, `NEW4`, or `NEW5`.
5. No PnL, policy scoring, threshold sweep, or model comparison may be run on `NEW5` during development.
6. Any NEW5 evaluation must be explicitly labeled as final one-shot holdout evaluation and logged with exact code/config.
