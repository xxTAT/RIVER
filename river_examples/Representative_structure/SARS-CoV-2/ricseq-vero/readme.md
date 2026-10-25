# Usage

``` bash
python step0_make_step1_step2_command.py

python step1_partition.py NC_045512_2_md.fasta NC_045512.2 5000 3000 > step1.log

python step2_structure_feature_detect.py SARSCoV2_VERO_RIC_merge.hic HS1 29903 15 10 5 > step2.log

python step3_optimal_structure_predict.py partition_file_id.txt SARSCoV2_VERO_RIC_merge_KR_15_15_remake10_merge_KR_10_10_OEmask_fill_diagonal_filter_gaussian.bedpe SARSCoV2_VERO_RIC_merge_KR_10.matrix SARSCoV2_VERO_RIC_merge_KR_5.matrix SARSCoV2_VERO_RIC_merge_KR_1.matrix 29903 HS1 NC_045512_2_md.fasta NC_045512.2 > step3.log
```

