# Hardware-path run: ionq_simulator-aria-1_shots1000_steps2

- circuits: 20 (8 calibration + 12 test), 10 qubits, depth 32, 90 RXX gates each, 1000 shots
- Trotter circuit vs exact evolution: max |dZ| = 0.0281
- measured feature shrinkage s = 0.794
- test months: mean |hardware Z - s*exact Z| = 0.0183 (pure shot noise would be ~0.0252)

MSE over 2008-07..2009-06 (12 months, Lehman/GFC):

| model | MSE |
|---|---|
| QR1 K=1 on ionq_simulator-aria-1_shots1000_steps2 | 0.2242 |
| QR1 K=1 exact simulation | 0.1802 |
| HAR | 0.2195 |
| Linear, same inputs, lags 1-3 | 0.1829 |

12 months is a small sample: this run tests that the hardware path works and matches the simulator's prediction, not which model is better.
