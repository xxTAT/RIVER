import numpy as np
import sys
from scipy import sparse
from scipy.signal import convolve2d
import itertools
import math
from pathlib import Path
import os
from datetime import datetime

def dump_matrix(hicFile,chrom):
    title = hicFile.replace(".hic","")
    res_list = [15,10,5,1]
    type_list = ['KR','NONE','VC_SQRT','VC']
    out_file_title = {}
    for res in res_list:
        n = 0
        type = type_list[n]
        out_file = title.replace('../step0_prepare/','')+"_"+type+"_"+str(res)+".matrix"
        os.system("java -jar /storx/yxdeng/software/juicer_tools.jar dump observed %s %s %s %s BP %s %s"%(type,hicFile,chrom,chrom,str(res),out_file))
        while os.path.getsize(out_file) == 0:
            os.remove(out_file)
            n += 1
            if n > 3:
                print(str(res),'is empty')
                break
            type = type_list[n]
            out_file = title.replace('../step0_prepare/','')+"_"+type+"_"+str(res)+".matrix"
            os.system("java -jar /storx/yxdeng/software/juicer_tools.jar dump observed %s %s %s %s BP %s %s"%(type,hicFile,chrom,chrom,str(res),out_file))
        if os.path.getsize(out_file) != 0:
            out_file_title[res] = out_file
    print('res',out_file_title.keys(),'available')
    return out_file_title

def get_oe(matrix):
    matrix = np.array(matrix).copy()
    lens = matrix.shape[0]
    mask = np.ones(lens)
    cut_off = lens * 0.0001
    matrix_num = matrix.copy()
    matrix_num[matrix_num > 0] = 1
    col_sum = matrix_num.sum(axis = 0)
    sites = np.arange(lens)
    for i in sites:
        if col_sum[i] <= cut_off:
            mask[i] = 0
    mask = mask == 1
    x_o = sites.copy()
    y_o = sites.copy()
    dis = np.zeros(lens)
    mask_o = mask.copy()
    exp_matrix = np.ones((lens, lens))
    x_e = sites.copy()
    y_e = sites.copy()
    for i in range(lens):
        diag_n = matrix[(x_o, y_o)]
        diag_line = diag_n[mask_o]
        if diag_line.shape[0] == 0:
            dis[i] = 0
        else :
            dis[i] = diag_line.mean()
            x_o = np.delete(x_o,-1)
            y_o = np.delete(y_o,0)
            mask_o = np.delete(mask_o,-1)
        
        if dis[i] == 0:
            x_e = np.delete(x_e,-1)
            y_e = np.delete(y_e,0)
            continue
        exp_matrix[(x_e, y_e)] = dis[i]
        exp_matrix[(y_e, x_e)] = dis[i]
        x_e = np.delete(x_e,-1)
        y_e = np.delete(y_e,0)

    oe = np.matrix(matrix/exp_matrix)
    return np.asarray(oe)

def get_matrix_npz(M1,M2,res):
    S1 = sparse.coo_matrix((M1['value'],(np.asarray(M1['S'], dtype= 'i4'),np.asarray(M1['E'], dtype= 'i4')))).toarray()
    S2 = sparse.coo_matrix((M2['value'],(np.asarray(M2['E'], dtype= 'i4'),np.asarray(M2['S'], dtype= 'i4')))).toarray()
    matrix = np.zeros((max(S1.shape),max(S1.shape)))
    matrix[np.nonzero(S1)] = S1[np.nonzero(S1)]
    matrix[np.nonzero(S2)] = S2[np.nonzero(S2)]
    if res == 15:
        W,H = matrix.shape
        for i in range(1,W-1):
            #matrix[i,i] = matrix[i-1,i-1] = matrix[i-1,i] = matrix[i-1,i+1] = matrix[i,i-1] = matrix[i,i+1] = matrix[i+1,i-1] = matrix[i+1,i] = matrix[i+1,i+1] = 0
            matrix[i,i] = matrix[i-1,i] = matrix[i,i-1] = matrix[i,i+1] = matrix[i+1,i] = 0
    return matrix

def extract_matrix(file_name,res):
    data = np.loadtxt(file_name, usecols=[0,1,2], dtype=[('S','<i4'),('E','<i4'),('value','<f8')])
    data['S'] = data['S'] / res
    data['E'] = data['E'] / res
    data['value'][data['value'] < 0] = 0
    data['value'][np.isnan(data['value'])] = 0
    mat = get_matrix_npz(data,data,res)
    oe = get_oe(mat)
    return mat,oe

def mask_diagonal(raw_matrix,oe_matirx):
    tmp = raw_matrix.copy()
    ones_bool = (oe_matirx <= 1)
    indices = np.where(ones_bool)
    indices
    for row,col in zip(indices[0],indices[1]):
        tmp[row,col] = 0
    return tmp

def gaussian(sigma,ksize):
    ksize = int(ksize)
    gaussian_k = np.zeros((ksize, ksize))
    for i in range(ksize):
        for j in range(ksize):
            gaussian_k[i, j] = np.round(math.exp((-1/(2*sigma**2))*(np.square(i-1) + np.square(j-1)))/(2*math.pi*sigma**2),4)
    sum = np.sum(gaussian_k)
    gaussian_k /= sum
    gaussian_k = np.round(gaussian_k,4)
    #print(gaussian_k)
    return gaussian_k

def gaussian_blur(Matrix,Kernel):
    convolved_matrix = convolve2d(Matrix, Kernel, mode='same', boundary='fill', fillvalue=0)
    return convolved_matrix

def NMS(M): 
    W, H = M.shape
    nms = np.copy(M)
    nms[0, :] = nms[W-1, :] = nms[:, 0] = nms[:, H-1] = 0
        
    for i in range(1, W-1):
        for j in range(1, H-1):
            if M[i, j] == 0:
                nms[i, j] = 0      
            else:
                if M[i, j] >= M[i-1, j-1] and M[i, j] >= M[i+1, j+1]:
                    nms[i, j] = M[i, j]
                else:
                    nms[i, j] = 0              
    return nms

def DT(NMS,TLF=0.01,THF=0.03):
    W, H = NMS.shape
    DT = np.zeros([W, H])
    strong_edges = np.zeros([W, H])
    TL = TLF * np.max(NMS)
    TH = THF * np.max(NMS)
    print('TL',TL)
    print('TH',TH)
    for i in range(1, W-1):
        for j in range(1, H-1):    
            if (NMS[i, j] > TH):
                strong_edges[i, j] = True
            else:
                strong_edges[i, j] = False
    last = np.sum(strong_edges)
    while(1):
        this = 0
        for i in range(1, W-1):
            for j in range(1, H-1):    
                if (strong_edges[i, j]):
                    DT[i, j] = True
                else:
                    if (NMS[i, j]>TL):
                        if ((DT[i-1, j-1:j+1].any()) or (DT[i+1, j-1:j+1].any()) or (DT[i, [j-1, j+1]].any())):
                            DT[i, j] = True
        this = np.sum(DT)
        if this - last == 0 :
            break
        last = this
    final_edges = DT.astype(int)    
    return final_edges

def matrix2_bedpe(M,output,res,chrom):
    ones_bool = (M == 1)
    #ones_bool = (M > 0)
    indices = np.where(ones_bool)
    files = open(output,'w')
    files.write('chrom1'+'\t'+'start1'+'\t'+'end1'+'\t'+'chrom2'+'\t'+'start2'+'\t'+'end2'+'\n')
    for row, col in zip(indices[0], indices[1]):
        row = int(row)
        col = int(col)
        row_start = str((row * res))
        row_end = str((row * res) + res)
        col_start = str((col * res))
        col_end = str((col * res) + res)
        strs = chrom+"\t"+row_start+"\t"+row_end+"\t"+chrom+"\t"+col_start+"\t"+col_end+"\n"
        files.write(strs)
    files.close()

def canny(hic_matrix,res,chrom,oe_mask=False,type=None,TLF=0.01,THF=0.03):
    kernel = gaussian(1.5,3)
    raw, oe = extract_matrix(hic_matrix,res)
    if oe_mask:
        ma = mask_diagonal(raw,oe)
        filer_m = gaussian_blur(ma,kernel)
        canny_out_title = hic_matrix.replace(".matrix",'') + "_" + str(res) + "_" + "OEmask" + ".bedpe"
    else:
        filer_m = gaussian_blur(raw,kernel)
        canny_out_title = hic_matrix.replace(".matrix",'') + "_" + str(res) + ".bedpe"
    if type == 'low':
        dt = DT(filer_m,TLF,THF)
    elif type == 'mid':
        nms = NMS(filer_m)
        dt = DT(nms,TLF,THF)  
    matrix2_bedpe(dt,canny_out_title,res,chrom)
    return canny_out_title

def low_res_canny_to_mid_res(low_res_canny,mid_res,chrom):
    lrc = open(low_res_canny,"r")
    out_title = low_res_canny.replace(".bedpe","") + "_remake" + str(mid_res) + ".bedpe"
    rrb = open(out_title,"w")
    rrb.write('chrom1'+'\t'+'start1'+'\t'+'end1'+'\t'+'chrom2'+'\t'+'start2'+'\t'+'end2'+'\n')
    tmp_set = set()
    for line in lrc:
        if 'chrom1' in line:
            continue
        sub = line.strip().split()
        s1 = int(sub[1])
        e1 = int(sub[2])
        s2 = int(sub[4])
        e2 = int(sub[5])
        if (s1 % 10 == 5):
            s1 = s1 - 5
        if (e1 % 10 == 5):
            e1 = e1 + 5
        if (s2 % 10 == 5):
            s2 = s2 - 5
        if (e2 % 10 == 5):
            e2 = e2 - 5
        tmp_set.add((s1,s1+mid_res,s2,s2+mid_res))
        tmp_set.add((s1,s1+mid_res,e2-mid_res,e2))
        tmp_set.add((e1-mid_res,e1,s2,s2+mid_res))
        tmp_set.add((e1-mid_res,e1,e2-mid_res,e2))
    for pixel in tmp_set:
        rrb.write(chrom+"\t"+str(pixel[0])+"\t"+str(pixel[1])+"\t"+chrom+"\t"+str(pixel[2])+"\t"+str(pixel[3])+"\n")
    lrc.close()
    rrb.close()
    return out_title

def canny_diagonal_filling(mid_res_bedpe,mid_res,length,chrom):
    matrix_len = int(length/mid_res) + 1
    tmp_matrix = np.zeros((matrix_len,matrix_len))
    hrb = open(mid_res_bedpe,"r")
    out_bedpe = mid_res_bedpe.replace(".bedpe","") + "_fill_diagonal.bedpe"
    outb = open(out_bedpe,"w")
    outb.write('chrom1'+'\t'+'start1'+'\t'+'end1'+'\t'+'chrom2'+'\t'+'start2'+'\t'+'end2'+'\n')
    for line in hrb:
        if 'chrom1' in line:
            continue
        sub = line.strip().split()
        tmp_matrix[int(int(sub[1])/mid_res),int(int(sub[4])/mid_res)] = 1
    for i in range(tmp_matrix.shape[0]):
        if (i - 1 >= 0):
            if (i + 1 <= matrix_len - 1):
                if (tmp_matrix[i-1,i+1] == 1) and (tmp_matrix[i+1,i-1] == 1):
                    tmp_matrix[i,i] = 1
            if (i + 2 <= matrix_len - 1):
                if (tmp_matrix[i-1,i+2] == 1) and (tmp_matrix[i+2,i-1] == 1):
                    tmp_matrix[i,i+1] = tmp_matrix[i+1,i] = 1
        if (i - 2 >= 0):
            if (i + 2 <= matrix_len - 1):
                if (tmp_matrix[i-2,i+2] == 1) and (tmp_matrix[i+2,i-2] == 1):
                    tmp_matrix[i-1,i+1] = tmp_matrix[i+1,i-1] = tmp_matrix[i,i] = 1
    ones_bool = (tmp_matrix == 1)
    indices = np.where(ones_bool)
    for row, col in zip(indices[0], indices[1]):
        outb.write(chrom+"\t"+str(row*mid_res)+"\t"+str(row*mid_res+mid_res)+"\t"+chrom+"\t"+str(col*mid_res)+"\t"+str(col*mid_res+mid_res)+"\n")
    hrb.close()
    outb.close()
    return out_bedpe

def merge_canny_bedpe(remake_bedpe,filling_bedpe,chrom,basename):
    out_file = remake_bedpe.replace(".bedpe","") + "_merge" + filling_bedpe.replace(basename,'')
    out = open(out_file,"w")
    out.write('chrom1'+'\t'+'start1'+'\t'+'end1'+'\t'+'chrom2'+'\t'+'start2'+'\t'+'end2'+'\n')
    pixels_set = set()
    for infile in [remake_bedpe,filling_bedpe]:
        with open(infile,"r")as f:
            for line in f:
                if 'chrom1' in line:
                    continue
                sub = line.strip().split()
                pixels_set.add((sub[1],sub[2],sub[4],sub[5]))
    for pixel in pixels_set:
        s1,e1,s2,e2 = pixel
        out.write(chrom+"\t"+str(s1)+"\t"+str(e1)+"\t"+chrom+"\t"+str(s2)+"\t"+str(e2)+"\n")
    out.close()
    return out_file

def canny_filter_gaussian(canny_result,rri_resulit,mid_res,length,chrom,CTF=0.01):
    matrix_len = int(length/mid_res) + 1
    canny_filter_count = 0
    rri_filter_count = 0
    cannyr = open(canny_result,"r")
    canny = np.zeros((matrix_len,matrix_len))
    for line in cannyr:
        if 'chrom1' in line:
            continue
        sub = line.strip().split()
        canny[int(int(sub[1])/mid_res),int(int(sub[4])/mid_res)] = 1
        canny[int(int(sub[4])/mid_res),int(int(sub[1])/mid_res)] = 1
    cannyr.close()
    rrim = open(rri_resulit,"r")
    rri = np.zeros((matrix_len,matrix_len))
    for line in rrim:
        sub = line.strip().split()
        if (sub[2] == 'NaN'):
            continue
        rri[int(int(sub[0])/mid_res),int(int(sub[1])/mid_res)] = float(sub[2])
        rri[int(int(sub[1])/mid_res),int(int(sub[0])/mid_res)] = float(sub[2])
    rrim.close()
    kernel = gaussian(1.5,3)
    rri_gaussian = gaussian_blur(rri,kernel)
    rri_gaussian_max = np.max(rri_gaussian)
    #print('rri_gaussian_max',rri_gaussian_max)
    #rri_cutoff = np.percentile(rri_gaussian, 5) 
    rri_cutoff = rri_gaussian_max * CTF
    print('rri_cutoff',rri_cutoff)
    tmp_matrix = np.zeros((matrix_len,matrix_len))
    for row in range(canny.shape[0]):
        for col in range(canny.shape[1]):
            if (canny[row,col] == 1):
                if (float(rri_gaussian[row,col]) > float(rri_cutoff)):
                    tmp_matrix[row,col] = 1
                else:
                    rri_filter_count += 1
                if (float(rri[row,col]) <= 0):
                    tmp_matrix[row,col] = 0
            else:
                canny_filter_count += 1
    outbedpe = canny_result.replace(".bedpe","") + "_filter_gaussian.bedpe"
    out = open(outbedpe,"w")
    out.write('chrom1'+'\t'+'start1'+'\t'+'end1'+'\t'+'chrom2'+'\t'+'start2'+'\t'+'end2'+'\n')
    ones_bool = (tmp_matrix > 0)
    indices = np.where(ones_bool)
    for row,col in zip(indices[0],indices[1]):
        out.write(chrom+"\t"+str(row*mid_res)+"\t"+str(row*mid_res+mid_res)+"\t"+chrom+"\t"+str(col*mid_res)+"\t"+str(col*mid_res+mid_res)+"\n")
    out.close()
    #print('rri_filter_count',rri_filter_count)
    #print('canny_filter_count',canny_filter_count)
    return outbedpe

def canny_add_pixels_id(canny_result):
    out_file = canny_result.replace(".bedpe",'_pixels.bedpe')
    canny = open(canny_result,"r")
    out = open(out_file,"w")
    #out.write("chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tpixel_id\n")
    pixel = 0
    for line in canny:
        if 'chrom1' in line:
            continue
        line = line.strip()
        sub = line.split()
        if (int(sub[1]) > int(sub[4])):
            continue
        pixel += 1
        pixel_id = 'pixel' + str(pixel)
        out.write(line+"\t"+pixel_id+"\n")
    canny.close()
    out.close()
    return out_file

def pixels_find_pair(pixels_file,mid_res):
    os.system("dos2unix %s"%(pixels_file))
    out_file = pixels_file.replace('_pixels.bedpe','_pixel2pixel.txt')
    os.system('bedtools pairtopair -a %s -b %s -is -rdn -slop %s > %s'%(pixels_file,pixels_file,str(mid_res),out_file))
    return out_file

def cluster_pixel_bedpe(bedtools_result,chrom_name):
    '''
    将bedtools找到的相互覆盖的像素进行聚类统计，将相邻的像素聚类后，输出聚类后的范围
    bedtools_result                  hs1	180	195	hs1	195	210	pixel134	hs1	180	195	hs1	180	195	pixel133                  相邻的像素位置
    out_file                         HS1	30	90	HS1	25350	25425	cluster_2                                             聚类簇
    chrom_name                       HS1                                                                                      染色体名称，好像没关系
    '''
    out_file = bedtools_result.replace("_pixel2pixel.txt","") + '_clusters.bedpe'
    pairs = {}
    pixels = {}
    result = open(bedtools_result,'r')
    for pair in result:
        sub = pair.strip().split()
        pixels[sub[6]] = [sub[0], sub[1], sub[2], sub[3], sub[4], sub[5]]
        pixels[sub[13]] = [sub[7], sub[8], sub[9], sub[10], sub[11], sub[12]]
        if (pairs.get(sub[6])):
            pairs[sub[6]].append(sub[13])
        else:
            pairs[sub[6]] = [sub[13]]
        if (pairs.get(sub[13])):
            pairs[sub[13]].append(sub[6])
        else:
            pairs[sub[13]] = [sub[6]]
    result.close()

    clustered = set()
    clusters = {}
    n = 0
    for pixel in pairs.keys():
        if (pixel in clustered):
            continue
        else:
            n += 1
            tmp_pixels = [pixel]
            tmp_cluster = []
            next_round = []
            while(1):
                if tmp_pixels:
                    for tmp_pixel in tmp_pixels:
                        if tmp_pixel in clustered:
                            continue
                        else:
                            clustered.add(tmp_pixel)
                            tmp_cluster.extend(pairs[tmp_pixel])
                            next_round.extend(pairs[tmp_pixel])
                    tmp_pixels = next_round
                    next_round = []
                else:
                    break
            cluster_id = 'cluster_' + str(n)
            tmp_cluster = list(set(tmp_cluster))
            clusters[cluster_id] = tmp_cluster

    out = open(out_file,'w')
    out.write("chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tname\n")
    cluster_loc = {}
    for cluster in clusters.keys():
        head_start = []
        head_end = []
        tail_start = []
        tail_end = []
        for pixel in clusters[cluster]:
            head_start.append(int(pixels[pixel][1]))
            head_end.append(int(pixels[pixel][2]))
            tail_start.append(int(pixels[pixel][4]))
            tail_end.append(int(pixels[pixel][5]))
        head_start_min = min(head_start)
        head_end_max = max(head_end)
        tail_start_min = min(tail_start)
        tail_end_max = max(tail_end)
        loc = [head_start_min, head_end_max, tail_start_min, tail_end_max]
        cluster_loc[cluster] = loc
        out.write(chrom_name+"\t"+str(head_start_min)+"\t"+str(head_end_max)+"\t"+chrom_name+"\t"+str(tail_start_min)+"\t"+str(tail_end_max)+"\t"+cluster+"\n")
    out.close()

    return out_file

def print_time():
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_now)

def make_step3_command(gaussian_filter_file,matrix_files):
    files_dict = {}
    with open('../step0_prepare/files.info','r')as f:
        for line in f:
            line = line.strip().split()
            files_dict[line[0]] = line[1]
    
    command = 'python step3_optimal_structure_predict.py partition_file_id.txt %s %s %s %s %s %s %s %s > step3.log'%(gaussian_filter_file,matrix_files[10],matrix_files[5],matrix_files[1],files_dict['length'],files_dict['chrom'],files_dict['fasta_file'],files_dict['seq_id'])
    print(command)

######################################################################################################################
pyname, hic_file, chrom, length, low_res, mid_res, high_res = sys.argv
print("----------------step2 start------------------")
print_time()

if not os.path.exists(path='./step2_structure_feature_detect'):
        os.mkdir(path='./step2_structure_feature_detect')
file_path = './step2_structure_feature_detect/'
os.chdir('./step2_structure_feature_detect')

base_name = hic_file.replace('.hic','')
hic_file = '../step0_prepare/' + hic_file

high_res = int(high_res)
mid_res = int(mid_res)
low_res = int(low_res)
length = int(length)

matrix_files = dump_matrix(hic_file,chrom)

canny_out_low_res_file = canny(matrix_files[low_res],low_res,chrom,oe_mask=False,type='low',TLF=0.01,THF=0.03)
canny_out_mid_res_file = canny(matrix_files[mid_res],mid_res,chrom,oe_mask=True,type='mid',TLF=0.01,THF=0.03)

remake_file = low_res_canny_to_mid_res(canny_out_low_res_file,mid_res,chrom)
filling_file = canny_diagonal_filling(canny_out_mid_res_file,mid_res,length,chrom)
merge_file = merge_canny_bedpe(remake_file,filling_file,chrom,base_name)

gaussian_filter_file = canny_filter_gaussian(merge_file,matrix_files[mid_res],mid_res,length,chrom,CTF=0.01)

canny_add_pixels_file = canny_add_pixels_id(gaussian_filter_file)

piexels_pairs_file = pixels_find_pair(canny_add_pixels_file,mid_res)

structure_feature_clusters_file = cluster_pixel_bedpe(piexels_pairs_file,chrom)

make_step3_command(gaussian_filter_file,matrix_files)

print("----------------step2 end------------------")
print_time()