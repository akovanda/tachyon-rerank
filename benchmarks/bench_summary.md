# Benchmark Summary (Best p50)

case | batch | requested_backend | actual_backend | p50_ms
---|---:|---|---|---:
n1024_d128 | 1 | qnn | QNN-Graph | 0.850
n1024_d128 | 2 | qnn | QNN-Graph | 1.280
n1024_d128 | 4 | qnn | QNN-Graph | 2.880
n1024_d128 | 8 | qnn | QNN-Graph | 5.430
n1024_d128 | 16 | qnn | QNN-Graph | 13.270
n1024_d128 | 32 | qnn | QNN-Graph | 27.470
n8192_d128 | 1 | cpu | CPU | 12.020
n8192_d128 | 2 | qnn | QNN-Graph | 22.710
n8192_d128 | 4 | cpu | CPU | 40.260
n8192_d128 | 8 | qnn | QNN-Graph | 55.970
n8192_d128 | 16 | qnn | QNN-Graph | 122.590
n8192_d128 | 32 | qnn | QNN-Graph | 156.650
n8192_d768 | 1 | cpu | CPU | 53.600
n8192_d768 | 2 | qnn | QNN-Graph | 78.310
n8192_d768 | 4 | qnn | QNN-Graph | 125.460
n8192_d768 | 8 | qnn | QNN-Graph | 193.760
n8192_d768 | 16 | qnn | QNN-Graph | 300.380
n8192_d768 | 32 | qnn | QNN-Graph | 530.010
