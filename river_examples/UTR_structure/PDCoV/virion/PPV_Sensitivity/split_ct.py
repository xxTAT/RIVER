#######################################################
#Data 2024-1-24
#Author Max Huang
#E-mail 1849732267@qq.com
#该脚本用于从全基因组ct文件中提取出目的区域
#######################################################
import pandas as pd
import numpy as np
import os,argparse
import sys
import subprocess

os.chdir(r"D:/tmp_workplace/dynamic_structure/260403/UTR/PDCoV/virion/PPV_Sensitivity")
file_name = 'GSE254992_PDCoV'
ct_file = file_name + '.ct'
strc_ric = np.loadtxt(ct_file, usecols=[0,1,2,3,4,5], dtype=[('1','<i4'),('2','<O'),('3','<i4'),('4','<i4'),('5','<i4'), ('6','<i4')], skiprows=1)
def select_strc(ct_f, s, e,start_site,types = 'rna_strc'):
    strc_selet = ct_f[s-1:e].copy()
    mask = (strc_selet['5'] < s) | (strc_selet['5'] >= e)
    strc_selet['5'][mask] = 0
    if types == 'rna_strc':
        strc_selet['1'] = strc_selet['1'] - start_site
        strc_selet['3'] = strc_selet['3'] - start_site
        strc_selet['4'] = strc_selet['4'] - start_site
        strc_selet['5'][strc_selet['5']>0] = strc_selet['5'][strc_selet['5']>0] - start_site
        strc_selet['6'] = strc_selet['6'] - start_site
    return strc_selet

s=24970
e=25420
####如果输出文件位置用原位置start——site=0，如果输出文件位置用1开始start——site=s - 1
start_site = 0
#start_site = s - 1
ric_1 = select_strc(strc_ric,s,e,start_site)
np.savetxt(file_name+"_"+str(s)+'-'+str(e)+'.ct', ric_1, fmt = '%s')

# def splitct(inputfile,startpos,endpos,title):
#     # s = int(''.join(filter(str.isdigit, startpos)))
#     # e = int(''.join(filter(str.isdigit, endpos)))
#     startpos=int(startpos)
#     endpos=int(endpos)
#     strc_ric = pd.read_csv(inputfile,names=['1', '2', '3', '4', '5', '6'], sep='\t', skiprows=1)
#     ric_1 = select_strc(strc_ric, startpos, endpos)
#     ric_1.to_csv(title + str(startpos) + '-' + str(endpos) + '.ct', sep='\t', index=False, header=False)
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="RNAseq data analysis pipeline. First-writtern by Max.")
#     parser.add_argument("-i", "--inputfile", type=str, required=True, help="Please type in the ct file.")
#     parser.add_argument("-s", "--startpos", type=str, required=True, help="Please type in the start position.")
#     parser.add_argument("-e", "--endpos", type=str, required=True, help="Please type in the end position.")
#     parser.add_argument("-t", "--title", type=str, required=True,help="Please type in the name of outputfile;for example:SARS2")
#     Args = parser.parse_args()
#     splitct(os.path.abspath(Args.inputfile),Args.startpos,Args.endpos,os.path.abspath(Args.title))