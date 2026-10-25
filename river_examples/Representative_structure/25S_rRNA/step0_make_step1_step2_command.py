files_dict = {}

with open('./step0_prepare/files.info','r')as f:
    for line in f:
        line = line.strip().split()
        files_dict[line[0]] = line[1]

command1 = 'python step1_partition.py %s %s'%(files_dict['fasta_file'],files_dict['seq_id'])

print('step1:',command1)

command2 = 'python step2_structure_feature_detect.py %s %s %s %s %s %s'%(files_dict['hic_file'],files_dict['chrom'],files_dict['length'],files_dict['low_res'],files_dict['mid_res'],files_dict['high_res'])

print('step2:',command2)

with open("step1_step2_command.txt",'w')as f:
    f.write(command1+"\n")
    f.write(command2)