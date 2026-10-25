# Usage

``` bash
python step0_make_step1_step2_command.py

python step1_partition.py KT336560_md.fasta KT336560.1 5000 3000 > step1.log

python step2_structure_feature_detect.py PDCoV_merge_cell_ric.hic HS1 25420 15 10 5 > step2.log

python step3_optimal_structure_predict.py partition_file_id.txt PDCoV_merge_cell_ric_KR_15_15_remake10_merge_KR_10_10_OEmask_fill_diagonal_filter_gaussian.bedpe PDCoV_merge_cell_ric_KR_10.matrix PDCoV_merge_cell_ric_KR_5.matrix PDCoV_merge_cell_ric_NONE_1.matrix 25420 HS1 KT336560_md.fasta KT336560.1 > step3.log
```

