# Hardware-path run: local_m15_s1_pack3_shots400

- 4 circuits x 30 qubits, 45 RXX each, 400 shots, debiasing off
- measured feature shrinkage s = 1.002; corr(hardware <Z>, simulated <Z>) = 0.997
- mean |hardware Z - s * simulated Z| = 0.0270 (shot noise alone ~0.0399)

MSE over 2008-07..2009-06 (12 months, Lehman/GFC):

| model | MSE |
|---|---|
| QR1 on local_m15_s1_pack3_shots400 | 0.2114 |
| same circuit, exact simulation | 0.2996 |
| HAR | 0.2195 |
| Linear, same inputs, lags 1-3 | 0.1829 |

12 months is a small sample: this run tests that the hardware path works and matches the simulator's prediction, not which model is better.
