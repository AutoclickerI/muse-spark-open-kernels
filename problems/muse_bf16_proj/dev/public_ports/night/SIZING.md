# T16/T32 carriers of the small-M stream scenario

Native e70024b1 and independent S19 timeline9c16c850. Values are weighted us/layer,46S+16F; not physical decomposition or promotion.

| T | Site | Surplus us [sampling95%CI] |
|---:|---|---|
| 16 | QKV | 0.301681 [0.157609,0.445754] |
| 16 | O | 0.477709 [0.360163,0.595255] |
| 16 | pool_router | 11.580011 [11.388716,11.771306] |
| 16 | unpooler | 0.918463 [0.776768,1.060158] |
| 16 | gate | 0.000000 [0.000000,0.000000] |
| 32 | QKV | 2.999935 [2.822404,3.177465] |
| 32 | O | 0.008758 [-0.011951,0.029467] |
| 32 | pool_router | 2.848397 [2.595282,3.101513] |
| 32 | unpooler | 2.536113 [2.058510,3.013715] |
| 32 | gate | 0.000000 [0.000000,0.000000] |

T32 pool/router is **dense BF16, six splits**,75,497,472bytes; T16 pool/router is NF12,S1,56,936,448bytes. QKV16 is raw BF16; QKV32 and both unpoolers are NF12. Gate has no positive exposed surplus in this model. O32 is only.008758us and remains the parent port point rather than a new build.

Timeline9c16c850 weighted critical/roof: d16 pool27.06/15.84us, QKV9.85/8.15, O6.14/2.80, unpool14.89/10.68; d32 pool21.50/16.58,QKV11.99/6.38,O6.85/3.62,unpool15.75/10.87. Those byte/tensor roofs exclude the fitted intercept, so their excess differs.
