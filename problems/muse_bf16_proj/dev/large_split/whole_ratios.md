# Held-out layer judgment against S19

H200 `7f110a377fb843a7`,16disjoint held-out seeds,fresh weights/router,960layer TASK +48strict native-pack projection/router checks. Tload=46sliding+16full; each seed retains six-load covariance. G0 is lower95%>1 across the six-load geomean.

| Fixed arm | R [95% CI] | G0 |
|---|---|---|
| AA | 0.999629 [0.998791, 1.000469] | False |
| SS | 0.971503 [0.970251, 0.972755] | False |
| PS | 0.977755 [0.976731, 0.978780] | False |
| NATIVE_PACK | 0.999817 [0.999058, 1.000576] | False |

| Shape (batch,q_len,window) | SS/S19 time | PS/S19 time | native-pack/S19 time |
|---|---|---|---|
| (1, 1, 2048) | 0.9986 [0.9939, 1.0032] | 1.0003 [0.9954, 1.0052] | 0.9986 [0.9946, 1.0026] |
| (1, 1, 0) | 0.9998 [0.9969, 1.0028] | 1.0002 [0.9967, 1.0037] | 0.9991 [0.9960, 1.0022] |
| (1, 8, 2048) | 1.0017 [0.9999, 1.0034] | 1.0017 [1.0001, 1.0032] | 1.0001 [0.9974, 1.0028] |
| (1, 8, 0) | 1.0010 [0.9984, 1.0036] | 0.9995 [0.9975, 1.0015] | 1.0017 [0.9996, 1.0037] |
| (16, 1, 2048) | 1.0031 [0.9998, 1.0064] | 1.0009 [0.9987, 1.0030] | 1.0000 [0.9971, 1.0030] |
| (16, 1, 0) | 0.9989 [0.9962, 1.0016] | 1.0009 [0.9990, 1.0027] | 1.0005 [0.9985, 1.0024] |
| (32, 1, 2048) | 1.0019 [0.9998, 1.0041] | 1.0036 [0.9998, 1.0074] | 1.0041 [1.0012, 1.0071] |
| (32, 1, 0) | 0.9983 [0.9963, 1.0003] | 1.0000 [0.9978, 1.0023] | 0.9992 [0.9971, 1.0014] |
| (16, 8, 2048) | 1.0639 [1.0565, 1.0713] | 1.0485 [1.0435, 1.0536] | 0.9934 [0.9905, 0.9963] |
| (16, 8, 0) | 1.0728 [1.0694, 1.0762] | 1.0521 [1.0493, 1.0549] | 1.0010 [0.9993, 1.0026] |
| (32, 8, 2048) | 1.1291 [1.1235, 1.1346] | 1.0987 [1.0932, 1.1042] | 1.0038 [1.0025, 1.0052] |
| (32, 8, 0) | 1.0830 [1.0803, 1.0857] | 1.0627 [1.0608, 1.0647] | 1.0036 [1.0023, 1.0049] |

| Load | native-pack R [95% CI] |
|---|---|
| decode1 | 1.0013 [0.9987, 1.0039] |
| decode16 | 0.9998 [0.9979, 1.0016] |
| decode32 | 0.9980 [0.9959, 1.0001] |
| verify1 | 0.9994 [0.9974, 1.0014] |
| verify16 | 1.0041 [1.0021, 1.0062] |
| verify32 | 0.9963 [0.9955, 0.9971] |

Combined experimental prepare144.599343s; measured additions{'ss': 2.8133091321215034, 'ps': 2.8203702110331506, 'np': 0.3155732380691916, 'np_pack': {'qkv_gate': 7.385807111859322e-05, 'pool_router': 0.00011109607294201851}}. The optional minimal NP patch drops SS/PS banks; its isolated prepare and its S20 composition remain unmeasured. The NATIVE_PACK arm has no changed T256 code; movement there receives no mechanism attribution.

No new candidate passes G0. The T128 native-layout patch is an unpromoted option for an already-planned P4 batch, with no separate confirmation requested.
