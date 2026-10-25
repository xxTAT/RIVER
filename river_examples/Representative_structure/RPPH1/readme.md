# Usage

``` bash
partition-smp NR_002312.1.fasta NR_002312.1.pfs

ProbabilityPlot -t NR_002312.1.pfs NR_002312.1.txt

python global_structure_constraint_generate_RPPH1_v1.py partition_file_name.txt RPPH1_10_KR_10_remake10_merge_RPPH1_5_VCRT_5_OEmask_fill_diagonal_filter_gaussian.bedpe RPPH1_10_KR.matrix RPPH1_5_VCRT.matrix RPPH1_1_VCRT.matrix

Fold-smp NR_002312.1.fasta RPPH1_96_v1.ct -c structure1_constrain.txt
```

