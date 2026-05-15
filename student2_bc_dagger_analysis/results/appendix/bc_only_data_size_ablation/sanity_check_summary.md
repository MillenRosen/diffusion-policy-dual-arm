# Experiment B Stratified Nested Sanity Check

1. Clean/noisy counts correct: Yes
2. Train/val split type is stratified and source counts reconcile: Yes
3. Nested subsets hold for N=20 subset N=40 subset N=80: Yes
4. No DAgger was run: Yes
5. Main experiment results were not overwritten: Yes, all outputs are under the stratified_nested experiment-B directory.
6. BC artifacts and 50ep eval JSON exist for all 9 combos: Yes
7. Final CSV rows: 9
8. Final plot generated: Yes

Results:
- noise0 N=20: clean=20, noisy=0, train=(16 clean, 0 noisy), val=(4 clean, 0 noisy), SR=0.20, success=10/50
- noise0 N=40: clean=40, noisy=0, train=(32 clean, 0 noisy), val=(8 clean, 0 noisy), SR=0.40, success=20/50
- noise0 N=80: clean=80, noisy=0, train=(64 clean, 0 noisy), val=(16 clean, 0 noisy), SR=0.74, success=37/50
- noise20 N=20: clean=16, noisy=4, train=(12 clean, 3 noisy), val=(4 clean, 1 noisy), SR=0.52, success=26/50
- noise20 N=40: clean=32, noisy=8, train=(25 clean, 6 noisy), val=(7 clean, 2 noisy), SR=0.76, success=38/50
- noise20 N=80: clean=64, noisy=16, train=(51 clean, 12 noisy), val=(13 clean, 4 noisy), SR=0.26, success=13/50
- noise30 N=20: clean=14, noisy=6, train=(11 clean, 4 noisy), val=(3 clean, 2 noisy), SR=0.18, success=9/50
- noise30 N=40: clean=28, noisy=12, train=(22 clean, 9 noisy), val=(6 clean, 3 noisy), SR=0.00, success=0/50
- noise30 N=80: clean=56, noisy=24, train=(44 clean, 19 noisy), val=(12 clean, 5 noisy), SR=0.60, success=30/50

CSV: D:\A-6019\project\project_new\experiments\final_dual_arm\bc_only varying number of demonstrations_stratified_nested\bc_demo_count_ablation_50ep.csv
Figure: D:\A-6019\project\project_new\experiments\final_dual_arm\bc_only varying number of demonstrations_stratified_nested\bc_demo_count_success_rate.png