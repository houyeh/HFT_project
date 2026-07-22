# From Directional Prediction to Toxic-flow Cancel Filter

- TCN is not used as a directional trading signal.
- TCN is used as a toxic-flow risk score for defensive maker quoting.
- Strategy mapping:
  - `P(Down)` high -> cancel bid
  - `P(Up)` high -> cancel ask
- Status: research candidate, not production-ready.
