# Usage

``` bash
python step0_make_step1_step2_command.py

python step1_partition.py MK584552_md.fasta MK584552.1 5000 3000 > step1.log

python step2_structure_feature_detect.py PEDV_merge_cell_ric.hic HS1 28044 15 10 5 > step2.log

python step3_optimal_structure_predict.py partition_file_id.txt PEDV_merge_cell_ric_KR_15_15_remake10_merge_KR_10_10_OEmask_fill_diagonal_filter_gaussian.bedpe PEDV_merge_cell_ric_KR_10.matrix PEDV_merge_cell_ric_KR_5.matrix PEDV_merge_cell_ric_NONE_1.matrix 28044 HS1 MK584552_md.fasta MK584552.1 > step3.log
```

