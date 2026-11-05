# QPU shot budget: simulator-trained readout, test months measured on hardware

245 test months (1997-08..2017-12). Hardware model: features shrunk by s=0.79 (IonQ aria-1 cloud level) plus binomial shot noise. Benchmarks: HAR 0.1156, linear on QR2 inputs 0.1016. Mean of 5 shot draws.


**QR1, naive** — test MSE by memory depth K (rows) and shots per circuit (columns)

| K | 100 | 300 | 1,000 | 3,000 | 10,000 | exact |
|---|---|---|---|---|---|---|
| 1 | 0.8358 | 0.4813 | 0.3351 | 0.3037 | 0.2938 | 0.2856 |
| 2 | 3.2899 | 1.3738 | 0.5790 | 0.3886 | 0.3131 | 0.2791 |
| 3 | 2.3352 | 1.0254 | 0.5152 | 0.3620 | 0.3018 | 0.2866 |

**QR1, noise-aware** — test MSE by memory depth K (rows) and shots per circuit (columns)

| K | 100 | 300 | 1,000 | 3,000 | 10,000 | exact |
|---|---|---|---|---|---|---|
| 1 | 0.1281 | 0.1202 | 0.1117 | 0.1125 | 0.1102 | 0.1088 |
| 2 | 0.1282 | 0.1151 | 0.1131 | 0.1118 | 0.1113 | 0.1054 |
| 3 | 0.1325 | 0.1207 | 0.1160 | 0.1147 | 0.1134 | 0.1051 |

**QR1, noise-aware + ridge** — test MSE by memory depth K (rows) and shots per circuit (columns)

| K | 100 | 300 | 1,000 | 3,000 | 10,000 | exact |
|---|---|---|---|---|---|---|
| 1 | 0.1221 | 0.1152 | 0.1133 | 0.1121 | 0.1113 | 0.1077 |
| 2 | 0.1213 | 0.1126 | 0.1117 | 0.1111 | 0.1107 | 0.1060 |
| 3 | 0.1230 | 0.1150 | 0.1124 | 0.1113 | 0.1111 | 0.1052 |

**QR1, noise-aware + ridge, s mismatch** — test MSE by memory depth K (rows) and shots per circuit (columns)

| K | 100 | 300 | 1,000 | 3,000 | 10,000 | exact |
|---|---|---|---|---|---|---|
| 1 | 0.1235 | 0.1159 | 0.1155 | 0.1165 | 0.1130 | 0.1627 |
| 2 | 0.1239 | 0.1168 | 0.1162 | 0.1165 | 0.1190 | 0.1081 |
| 3 | 0.1262 | 0.1189 | 0.1150 | 0.1145 | 0.1193 | 0.1049 |

**QR2, naive** — test MSE by memory depth K (rows) and shots per circuit (columns)

| K | 100 | 300 | 1,000 | 3,000 | 10,000 | exact |
|---|---|---|---|---|---|---|
| 1 | 220.1025 | 69.5232 | 20.1916 | 7.4088 | 2.3544 | 0.2733 |
| 2 | 13.6685 | 4.7334 | 1.5643 | 0.7060 | 0.3979 | 0.2676 |
| 3 | 38.7423 | 12.9396 | 3.8019 | 1.4918 | 0.6438 | 0.2601 |

**QR2, noise-aware** — test MSE by memory depth K (rows) and shots per circuit (columns)

| K | 100 | 300 | 1,000 | 3,000 | 10,000 | exact |
|---|---|---|---|---|---|---|
| 1 | 0.1239 | 0.1191 | 0.1162 | 0.1159 | 0.1143 | 0.1141 |
| 2 | 0.1236 | 0.1187 | 0.1164 | 0.1169 | 0.1156 | 0.1118 |
| 3 | 0.1238 | 0.1200 | 0.1181 | 0.1179 | 0.1143 | 0.1038 |

**QR2, noise-aware + ridge** — test MSE by memory depth K (rows) and shots per circuit (columns)

| K | 100 | 300 | 1,000 | 3,000 | 10,000 | exact |
|---|---|---|---|---|---|---|
| 1 | 0.1194 | 0.1212 | 0.1163 | 0.1146 | 0.1146 | 0.1131 |
| 2 | 0.1235 | 0.1190 | 0.1157 | 0.1148 | 0.1132 | 0.1090 |
| 3 | 0.1209 | 0.1178 | 0.1150 | 0.1133 | 0.1110 | 0.1051 |

**QR2, noise-aware + ridge, s mismatch** — test MSE by memory depth K (rows) and shots per circuit (columns)

| K | 100 | 300 | 1,000 | 3,000 | 10,000 | exact |
|---|---|---|---|---|---|---|
| 1 | 0.1301 | 0.1288 | 0.1427 | 0.1230 | 0.1500 | 1.2265 |
| 2 | 0.1271 | 0.1240 | 0.1250 | 0.1217 | 0.1485 | 0.2250 |
| 3 | 0.1255 | 0.1213 | 0.1211 | 0.1192 | 0.1212 | 0.1538 |

Fewest shots per circuit that still beat HAR on average:

- QR1, K=1 (10 qubits), noise-aware: 1,000
- QR1, K=2 (17 qubits), noise-aware: 300
- QR1, K=3 (24 qubits), noise-aware: 3,000
- QR1, K=1 (10 qubits), noise-aware + ridge: 300
- QR1, K=2 (17 qubits), noise-aware + ridge: 300
- QR1, K=3 (24 qubits), noise-aware + ridge: 300
- QR1, K=1 (10 qubits), noise-aware + ridge, s mismatch: 1,000
- QR1, K=2 (17 qubits), noise-aware + ridge, s mismatch: not within 10,000
- QR1, K=3 (24 qubits), noise-aware + ridge, s mismatch: 1,000
- QR2, K=1 (10 qubits), noise-aware: 10,000
- QR2, K=2 (17 qubits), noise-aware: 10,000
- QR2, K=3 (24 qubits), noise-aware: 10,000
- QR2, K=1 (10 qubits), noise-aware + ridge: 3,000
- QR2, K=2 (17 qubits), noise-aware + ridge: 3,000
- QR2, K=3 (24 qubits), noise-aware + ridge: 1,000
- QR2, K=1 (10 qubits), noise-aware + ridge, s mismatch: not within 10,000
- QR2, K=2 (17 qubits), noise-aware + ridge, s mismatch: not within 10,000
- QR2, K=3 (24 qubits), noise-aware + ridge, s mismatch: not within 10,000
