import numpy as np
import sys
from scipy import sparse
from datetime import datetime
import os
from pathlib import Path
################################################################################
def expand_matrix(matrix, old_res, new_res, length):
    tmp_matrix = np.zeros((length,length))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            time = int(old_res/new_res)
            l = row * time
            r = col * time
            l_end = l + time if l + time <= length - 1 else length
            r_end = r + time if r + time <= length - 1 else length
            tmp_matrix[l:l_end,r:r_end] = matrix[row,col]
    return tmp_matrix

def load_canny_bedpe(infile,file_path,length):
    matrix_len = (int(length/25) + 2) * 5
    tmp_matrix = np.zeros((matrix_len,matrix_len))
    with open(file_path+infile,"r")as f:
        for line in f:
            if 'chrom1' in line:
                continue
            sub = line.strip().split()
            tmp_matrix[int(int(sub[1])/5),int(int(sub[4])/5)] = 1
    return tmp_matrix

class TreePoint:
    def __init__(self, num=None, father=None, son=None, structure=None, path=None):
        self.num = num
        self.father = father
        self.son = son
        self.structure = structure
        self.path = path

def pseudoknot_judgment(pre_path, units_loc, structure):
    structure_locs = units_loc[structure]
    structure_left = [int(loc.split("_")[0]) for loc in structure_locs]
    structure_right = [int(loc.split("_")[1]) for loc in structure_locs]
    structure_left_start, structure_left_end = min(structure_left), max(structure_left)
    structure_right_start, structure_right_end = min(structure_right), max(structure_right)

    for pre_structure in pre_path:
        if pre_structure == 's0':
            continue
        pre_structure_locs = units_loc[pre_structure]
        left = [int(loc.split("_")[0]) for loc in pre_structure_locs]
        right = [int(loc.split("_")[1]) for loc in pre_structure_locs]
        left_start, left_end = min(left), max(left)
        right_start, right_end = min(right), max(right)

        if (left_end < structure_left_start < right_start < structure_right_start > right_end) or \
           (structure_right_start > left_end and structure_right_end < right_start and structure_left_end < left_start):
            return True
    return False

def conflict_judgment_filter(tmp_next,units_loc,pre_path,length):
    not_conflict_structure = []
    region_start = 0
    region_end = length - 1
    region_len = region_end - region_start + 1
    dot = np.zeros(region_len)
    for structure in pre_path:
        if structure == 's0':
            continue
        for loc in units_loc[structure]:
            sub = loc.split("_")
            dot[int(sub[0]) - region_start] = 1
            dot[int(sub[1]) - region_start] = 1

    not_conflict_structure = []
    for tmp_structure in tmp_next:
        if pseudoknot_judgment(pre_path, units_loc, tmp_structure):
            continue
        if all(dot[int(loc.split("_")[0]) - region_start] == 0 and dot[int(loc.split("_")[1]) - region_start] == 0 for loc in units_loc[tmp_structure]):
            not_conflict_structure.append(tmp_structure)
    return not_conflict_structure

def build_structure_tree(structure_count, units_loc, length):
    root = TreePoint(num='point0', father=None, son=[], structure='s0', path=['s0'])
    sorted_structure_count = sorted(structure_count, key=lambda x: x[1], reverse=True)
    if not sorted_structure_count:
        return None, None

    point_num = 0
    next_round_structure = conflict_judgment_filter(
        [structure for structure, _ in sorted_structure_count if _ == sorted_structure_count[0][1]],
        units_loc, root.path, length
    )

    for structure in next_round_structure:
        point_num += 1
        root.son.append(TreePoint(num=f'point{point_num}', father=root, son=[], structure=structure, path=root.path + [structure]))

    next_round_point = root.son
    paths = []

    while next_round_point:
        tmp_next_round_point = []
        for point in next_round_point:
            tmp_path = point.path[:]
            last_structure_value = next(value for structure, value in sorted_structure_count if structure == tmp_path[-1])
            tmp_structure_count_rm_last_structure = [(structure, value) for structure, value in sorted_structure_count if value < last_structure_value]

            while tmp_structure_count_rm_last_structure:
                tmp_next_structure = [structure for structure, value in tmp_structure_count_rm_last_structure if value == tmp_structure_count_rm_last_structure[0][1]]
                next_round_structure = conflict_judgment_filter(tmp_next_structure, units_loc, tmp_path, length)
                if next_round_structure:
                    break
                tmp_structure_count_rm_last_structure = [(structure, value) for structure, value in tmp_structure_count_rm_last_structure if value != tmp_structure_count_rm_last_structure[0][1]]

            if not tmp_structure_count_rm_last_structure:
                paths.append(tmp_path)
                continue

            for structure in next_round_structure:
                point_num += 1
                new_point = TreePoint(num=f'point{point_num}', father=point, son=[], structure=structure, path=tmp_path + [structure])
                point.son.append(new_point)
                tmp_next_round_point.append(new_point)

        next_round_point = tmp_next_round_point

    paths_mvd = [list(path) for path in set(tuple(path) for path in paths)]
    return root, paths_mvd

def get_matrix_npz(M1,M2):
    '''
    use rri-matrix dump result to creat matrix 
    '''
    S1 = sparse.coo_matrix((M1['value'],(np.asarray(M1['S'], dtype= 'i4'),np.asarray(M1['E'], dtype= 'i4')))).toarray()
    S2 = sparse.coo_matrix((M2['value'],(np.asarray(M2['E'], dtype= 'i4'),np.asarray(M2['S'], dtype= 'i4')))).toarray()
    matrix = np.zeros((max(S1.shape),max(S1.shape)))
    matrix[np.nonzero(S1)] = S1[np.nonzero(S1)]
    matrix[np.nonzero(S2)] = S2[np.nonzero(S2)]
    return matrix

def extract_matrix(file_name,res,file_path):
    data = np.loadtxt(file_path+file_name, usecols=[0,1,2], dtype=[('S','<i4'),('E','<i4'),('value','<f8')])
    data['S'] = data['S'] / res
    data['E'] = data['E'] / res
    data['value'][data['value'] < 0] = 0
    data['value'][np.isnan(data['value'])] = 0
    matrix = get_matrix_npz(data,data)
    return matrix

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
    ##obs_matrix
        diag_n = matrix[(x_o, y_o)]
        diag_line = diag_n[mask_o]
        if diag_line.shape[0] == 0:
            dis[i] = 0
        else :
            dis[i] = diag_line.mean()
            x_o = np.delete(x_o,-1)
            y_o = np.delete(y_o,0)
            mask_o = np.delete(mask_o,-1)
        
        ##exp_matrix
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

def extract_oe_matrix(file_name,res,file_path):
    data = np.loadtxt(file_path+file_name, usecols=[0,1,2], dtype=[('S','<i4'),('E','<i4'),('value','<f8')])
    data['S'] = data['S'] / res
    data['E'] = data['E'] / res
    data['value'][data['value'] < 0] = 0
    data['value'][np.isnan(data['value'])] = 0
    mat = get_matrix_npz(data,data)
    oe = get_oe(mat)
    return oe

def create_all_partition_matrixs(fileprefixs_file,file_path,length):
    fileprefixs = []
    with open(file_path+fileprefixs_file,"r")as f:
        for line in f:
            fileprefixs.append(line.strip())
    tmp_matrix = np.zeros((length,length))
    for fileprefix in fileprefixs:
        fileprefix_replace = fileprefix.replace(".txt","")
        sub = fileprefix_replace.split("_")
        left_start = int(sub[0])
        right_start = int(sub[2])
        tmp = open(file_path+'tmp_pfs_txt/'+fileprefix, 'r')
        for index, line in enumerate(tmp):
            if index < 2:
                continue
            else:
                tmp_sub = line.strip().split()
                left = int(tmp_sub[0]) - 1 + left_start
                right = int(tmp_sub[1]) - 1 + right_start
                if (float(tmp_sub[2]) > 0) and (tmp_matrix[left, right] == 0):
                    tmp_matrix[left, right] = float(tmp_sub[2])
                elif (float(tmp_sub[2]) > 0) and (tmp_matrix[left, right] != 0):
                    tmp_matrix[left, right] = ((tmp_matrix[left, right] + float(tmp_sub[2]))/2)
        tmp.close()
    return tmp_matrix

def diagonal_1000_processing(all_partition_matrix,length):
    tmp = all_partition_matrix.copy()
    x,y = np.diag_indices_from(tmp)

    for time in range(10):
        tmp[x,y] = 0
        tmp[y,x] = 0
        x = np.delete(x,-1)
        y = np.delete(y,0)

    rows, cols = np.ogrid[:length, :length]
    mask = np.abs(rows - cols) >= 1000
    tmp[mask] = 0
    return tmp

def canny_partition_intersection(canny_matrix,partition_matrix,outfile,length,chrom):
    tmp_matrix = np.zeros((length,length))
    out = open(outfile,"w")
    for row in range(canny_matrix.shape[0]):
        for col in range(canny_matrix.shape[1]):
            if (canny_matrix[row,col] == 1) and (partition_matrix[row,col] > 0):
                tmp_matrix[row,col]  = partition_matrix[row,col]
                out.write(chrom+"\t"+str(row)+"\t"+str(row+1)+"\t"+chrom+"\t"+str(col)+"\t"+str(col+1)+"\n")
    out.close()
    return tmp_matrix

def cluster_pixel_bedpe(bedtools_result, out_file, chrom_name):
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
    cluster_loc = {}
    for cluster in clusters.keys():
        head_start = []
        head_end = []
        tail_start = []
        tail_end = []
        for pixel in clusters[cluster]:
            head_start.append(pixels[pixel][1])
            head_end.append(pixels[pixel][2])
            tail_start.append(pixels[pixel][4])
            tail_end.append(pixels[pixel][5])
        head_start_min = min(head_start)
        head_end_max = max(head_end)
        tail_start_min = min(tail_start)
        tail_end_max = max(tail_end)
        loc = [head_start_min, head_end_max, tail_start_min, tail_end_max]
        cluster_loc[cluster] = loc
        out.write(chrom_name+"\t"+str(head_start_min)+"\t"+str(head_end_max)+"\t"+chrom_name+"\t"+str(tail_start_min)+"\t"+str(tail_end_max)+"\t"+cluster+"\n")
    out.close()

    return pixels, clusters, cluster_loc

def number_structural_unit(candidate_structure_matrix,chrom):
    '''
    candidate_structure_matrix          matrix(28044,28044)      where pixel > 0 is candidate
    cluster_loc          [row_start,row_end,col_start,col_end]
    unit_loc      [pixelrow_pixelcol,......]           pixel_set for unit
    unit_name             [s1,........]               unit_num
    '''
    unit_loc = {}
    unit_name = []
    pixel_unit = {}
    checked_pixels = set()
    tmp_structure_num = 0
    rows, cols = candidate_structure_matrix.shape
    for row in range(rows):
        for col in range(cols):
            if (candidate_structure_matrix[row,col] > 0) and (str(row)+"_"+str(col) not in checked_pixels):
                if (row + 1 > rows - 1) or (col - 1 < 0):
                    continue
                else:
                    if (candidate_structure_matrix[row+1,col-1] > 0):
                        tmp_structure_num += 1
                        tmp_structure_id = 's' + str(tmp_structure_num)
                        pixel_list = [str(row)+"_"+str(col),str(row+1)+"_"+str(col-1)]
                        unit_loc[tmp_structure_id] = pixel_list
                        unit_name.append(tmp_structure_num)
                        pixel_unit[str(row)+"_"+str(col)] = tmp_structure_id
                        pixel_unit[str(row+1)+"_"+str(col-1)] = tmp_structure_id
                        checked_pixels.add(str(row)+"_"+str(col))
                        checked_pixels.add(str(row+1)+"_"+str(col-1))
            elif (candidate_structure_matrix[row,col] > 0) and (str(row)+"_"+str(col) in checked_pixels):
                if (row + 1 > rows - 1) or (col - 1 < 0):
                    continue
                else:
                    if (candidate_structure_matrix[row+1,col-1] > 0):
                        structure_id = pixel_unit[str(row)+"_"+str(col)]
                        unit_loc[structure_id].append(str(row+1)+"_"+str(col-1))
                        pixel_unit[str(row+1)+"_"+str(col-1)] = structure_id
                        checked_pixels.add(str(row+1)+"_"+str(col-1))
            else:
                continue
    with open('units_check.bedpe',"w")as f:
        f.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
        for unit in unit_loc:
            locs = unit_loc[unit]
            for loc in locs:
                sub = loc.split("_")
                left = int(sub[0])
                right = int(sub[1])
                f.write(chrom+"\t"+str(left)+"\t"+str(left+1)+"\t"+chrom+"\t"+str(right)+"\t"+str(right+1)+"\n")
    return unit_loc,unit_name

def calculate_unit_read_count(units_loc,rri_matrix_10,rri_matrix_5,rri_matrix_1,partition_matrix,length,average=None):
    """
    units_loc      {s1:[start_end,start_end,...],....}            unit pixel loc
    rri_matrix               matrix(28044,28044)
    units_support       [(s1,count),...]                     unit reads counts
    """
    rri_matrix_10 = rri_matrix_10[:length, :length]
    rri_matrix_5 = rri_matrix_5[:length, :length]
    rri_matrix_1 = rri_matrix_1[:length, :length]
    merge_matrix = rri_matrix_10 + rri_matrix_5 + rri_matrix_1
    merge_99 = np.percentile(merge_matrix,99)
    merge_matrix[merge_matrix >= merge_99] = merge_99
    merge_98 = np.percentile(merge_matrix[merge_matrix>0],98)
    mask = (partition_matrix <= -np.log10(0.75)) & (merge_matrix <= merge_98)
    merge_matrix[mask] = 0

    units_support = []
    for unit in units_loc.keys():
        pixels_num = 0
        count = 0
        locs = units_loc[unit]
        for loc in locs:
            pixels_num += 1
            row = int(loc.split("_")[0])
            col = int(loc.split("_")[1])
            count += merge_matrix[row,col] + (merge_matrix[row,col] / partition_matrix[row,col])
            #count += rri_matrix_5[row,col] + rri_matrix_1[row,col] + ((rri_matrix_5[row,col] + rri_matrix_1[row,col]) / partition_matrix[row,col])
        if average:
            count /= pixels_num
        units_support.append((unit,count))
    units_support_list = [count for (_,count) in units_support]
    units_support_list_96 = np.percentile(units_support_list,96)
    units_support = [(structure,count) for (structure,count) in units_support if count >= units_support_list_96]

    return units_support

def create_structure_bedpe(structure_list,structure_loc,out_file,chrom):
    pairs = []
    out = open(out_file,"w")
    for structure in structure_list:
        if structure == 's0':
            continue
        locs = structure_loc[structure]
        for loc in locs:
            sub = loc.split("_")
            out.write(chrom+"\t"+str(int(sub[0]))+"\t"+str(int(sub[0])+1)+"\t"+chrom+"\t"+str(int(sub[1]))+"\t"+str(int(sub[1])+1)+"\n")
            pairs.append((int(sub[0]),int(sub[1])))
    out.close()
    return pairs

def print_time():
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_now)

def mask_diagonal(raw_matrix,oe_matirx):
    tmp = raw_matrix.copy()
    ones_bool = (oe_matirx <= 1)
    indices = np.where(ones_bool)
    indices
    for row,col in zip(indices[0],indices[1]):
        tmp[row,col] = 0
    return tmp

def load_fasta(fasta_path, select_id=None):
    '''
    fasta_dict      {id : seq list}
    '''
    #fasta_path:in file
    #select_id:id list
    fasta_dict = {}
    fasta = open(fasta_path,'r')
    for line in fasta:
        if line.startswith(">"):
            seqid = line.strip().split()[0][1:]
            fasta_dict[seqid] = ''
        else:
            fasta_dict[seqid] += line.strip()
    fasta.close()
    if select_id:
        select_dict = {}
        for select in select_id:
            test = fasta_dict.get(select)
            if test:
                select_dict[select] = list(test)
            else:
                print("seq is not in fasta",select)
        if  (len(select_dict) != 0):
            return select_dict
        else:
            print('all select not in fasta')
    else:
        return fasta_dict

def fold_structure(prefixs):
    if not os.path.exists(path='../domain_structure'):
        os.mkdir(path='../domain_structure')
    structure_path = '../domain_structure/'
    for prefix in prefixs:
        print(prefix)
        seq = prefix + '.fasta'
        const = prefix + '.const'
        out = prefix + '.ct'
        os.system("Fold-smp %s %s -c %s"%(seq,structure_path+out,const))
        print(prefix,'-------end-----')

def split_domain(structure_bedpe,length,chrom):
    sb = open(structure_bedpe,"r")
    out_file = structure_bedpe.replace('.bedpe','') + '_domain.bedpe'
    of = open(out_file,"w")
    of.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
    pairs = {}
    pixels = {}
    pixels_list = []
    overlap_list = []
    domain_loc = []
    n = 0
    for line in sb:
        if 'chrom1' in line:
            continue
        n += 1
        pixel_id = 'pixel'+str(n)
        pixels_list.append(pixel_id)
        sub = line.strip().split()
        pixels[pixel_id] = (int(sub[1]),int(sub[4]))
        pairs[int(sub[1])] = int(sub[4])
        pairs[int(sub[4])] = int(sub[1])
    sb.close()
    for pixel in pixels_list:
        left,right = pixels[pixel]
        left_len = left - 0 + 1
        right_len = length -1 - right + 1
        if left_len <= right_len:
            for i in range(left-1,-1,-1):
                tmp_right = pairs.get(i,0)
                if int(tmp_right) > right:
                    overlap_list.append(pixel)
                    break
        else:
            for i in range(right+1,length,1):
                tmp_left = pairs.get(i,length)
                if int(tmp_left) < left:
                    overlap_list.append(pixel)
                    break
    final_pixels = [pixel for pixel in pixels_list if pixel not in overlap_list]
    for pixel in final_pixels:
        left,right = pixels[pixel]
        of.write(chrom+"\t"+str(left)+"\t"+str(right+1)+"\t"+chrom+"\t"+str(left)+"\t"+str(right+1)+"\n")
        domain_loc.append((left,right+1))
    of.close()
    return domain_loc

def find_blank(domain_locs,length):
    domain_locs_sorted = sorted(domain_locs,key= lambda x:x[0])
    blank = []
    for index,domain in enumerate(domain_locs_sorted):
        if index == 0:
            if domain[0] > 0:
                blank.append((0,domain[0]))
            continue
        if index == len(domain_locs_sorted) - 1:
            if domain[1] < length - 1:
                blank.append((domain[1],length - 1))
            continue
        last = domain_locs_sorted[index-1]
        this = domain
        if this[0] - last[1] > 0:
            blank.append((last[1],this[0]))
    return blank

def connect_domain_and_fill_blank(domain_locs,bedpe_file_name,length,chrom):
    domain_locs_sorted = sorted(domain_locs,key= lambda x:x[0])
    large_domain = []
    final_domain = []
    for domain in domain_locs_sorted:
        domain_len = domain[1] - domain[0] + 1
        if domain_len > 600:
            large_domain.append(domain)
    for index,domain in enumerate(large_domain):
        if index == 0:
            if domain[0] < 600:
                if domain[0] < 100:
                    final_domain.append((0,domain[1]))
                    continue
                final_domain.append((0,domain[0]))
                final_domain.append((domain[0],domain[1]))
            else:
                start = 0
                end = start
                tmp_domains_list = []
                tmp_index = domain_locs_sorted.index(domain)
                if tmp_index == 0:
                    final_domain.append((0,domain[0]))
                    final_domain.append((domain[0],domain[1]))
                    continue
                for i in range(tmp_index):
                    tmp_domains_list.append(domain_locs_sorted[i])
                tmp_domains_len = len(tmp_domains_list)
                n = 0
                while n < tmp_domains_len:
                    this = tmp_domains_list[n]
                    end = this[1]
                    if end - start + 1 > 600:
                        final_domain.append((start,this[0]))
                        start = this[0]
                        end = start
                    n += 1
                    if n == tmp_domains_len:
                        if end < domain[1] - 1:
                            final_domain.append((start,domain[0]))
                final_domain.append((domain[0],domain[1]))
        elif index == len(large_domain) - 1:
            if length - 1 - domain[1] +1 < 600:
                if length - 1 - domain[1] +1 < 100 :
                    final_domain.append((domain[0],length))
                    continue
                final_domain.append((domain[1],length))
                final_domain.append((domain[0],domain[1]))
            else:
                start = domain[1]
                end = start
                tmp_domains_list = []
                tmp_index = domain_locs_sorted.index(domain)
                if tmp_index == len(domain_locs_sorted) - 1:
                    final_domain.append((domain[1],length))
                    final_domain.append((domain[0],domain[1]))
                    continue
                for i in range(tmp_index+1,len(domain_locs_sorted)):
                    tmp_domains_list.append(domain_locs_sorted[i])
                tmp_domains_len = len(tmp_domains_list)    
                n = 0
                while n < tmp_domains_len:
                    this = tmp_domains_list[n]
                    end = this[1]
                    if end - start + 1 > 600:
                        final_domain.append((start,this[0]))
                        start = this[0]
                        end = start
                    n += 1
                    if n == tmp_domains_len:
                        if end < length:
                            final_domain.append((start,length))
                final_domain.append((domain[0],domain[1]))
        elif index == len(large_domain) - 2:
            #last is this , this is end
            tmp_last_list = [domain,large_domain[index-1]]
            tmp_this_list = [large_domain[index+1],domain]
            for i in range(2):
                last = tmp_last_list[i]
                this = tmp_this_list[i]
                if this[0] - last[1] + 1 < 600:
                    if this[0] - last[1] + 1 < 200:
                        final_domain.append((last[0],this[0]))
                        continue
                    final_domain.append((last[1],this[0]))
                    final_domain.append((last[0],last[1]))
                else:
                    start = last[1]
                    end = start
                    tmp_domains_list = []
                    tmp_last_index = domain_locs_sorted.index(last)
                    tmp_this_index = domain_locs_sorted.index(this)
                    if tmp_this_index - tmp_last_index == 1:
                        final_domain.append((last[1],this[0]))
                        final_domain.append((last[0],last[1]))
                        continue
                    for i in range(tmp_last_index+1,tmp_this_index):
                        tmp_domains_list.append(domain_locs_sorted[i])
                    tmp_domains_len = len(tmp_domains_list)    
                    n = 0
                    while n < tmp_domains_len:
                        tmp_this = tmp_domains_list[n]
                        end = tmp_this[1]
                        if end - start + 1 > 600:
                            final_domain.append((start,tmp_this[0]))
                            start = tmp_this[0]
                            end = start
                        n += 1
                        if n == tmp_domains_len:
                            if end <= this[0]:
                                final_domain.append((start,this[0]))
                    final_domain.append((last[0],last[1]))
        else:
            last = large_domain[index-1]
            this = domain
            if this[0] - last[1] + 1 < 600:
                if this[0] - last[1] + 1 < 100:
                    final_domain.append((last[0],this[0]))
                    continue
                final_domain.append((last[1],this[0]))
                final_domain.append((last[0],last[1]))
            else:
                start = last[1]
                end = start
                tmp_domains_list = []
                tmp_last_index = domain_locs_sorted.index(last)
                tmp_this_index = domain_locs_sorted.index(this)
                if tmp_this_index - tmp_last_index == 1:
                    final_domain.append((last[1],this[0]))
                    final_domain.append((last[0],last[1]))
                    continue
                for i in range(tmp_last_index+1,tmp_this_index):
                    #print('middle',domain_locs_sorted[i])
                    tmp_domains_list.append(domain_locs_sorted[i])
                tmp_domains_len = len(tmp_domains_list)    
                n = 0
                while n < tmp_domains_len:
                    tmp_this = tmp_domains_list[n]
                    end = tmp_this[1]
                    if end - start + 1 > 600:
                        #print('whlie',n,start,tmp_this[0])
                        final_domain.append((start,tmp_this[0]))
                        start = tmp_this[0]
                        end = start
                    n += 1
                    #print('add',n)
                    if n == tmp_domains_len:
                        #print('n == tmp_domains_len')
                        if end <= this[0]:
                            #print('last',start,this[0])
                            final_domain.append((start,this[0]))
                if index == 1:
                    continue
                final_domain.append((last[0],last[1]))
    blank = find_blank(final_domain,length)
    for b in blank:
        if b[1] - b[0] + 1 < 100:
            left_and_right = [domain for domain in final_domain if domain[1] == b[0] or domain[1] == b[1]]
            if len(left_and_right) == 2:
                left_len = left_and_right[0][1] - left_and_right[0][0] + 1
                right_len = left_and_right[1][1] - left_and_right[1][0] + 1
                if left_len > right_len:
                    final_domain.remove(left_and_right[1])
                    final_domain.append((b[0],left_and_right[1][1]))
                else:
                    final_domain.remove(left_and_right[0])
                    final_domain.append((left_and_right[0][0],b[1]))
            else:
                min_loc = min(b[0],left_and_right[0][0])
                max_loc = max(left_and_right[0][1],b[1])
                final_domain.remove(left_and_right[0])
                final_domain.append((min_loc,max_loc))
        elif b[1] - b[0] + 1 < 600:
            final_domain.append(b)
        else:
            small_domains = [domain for domain in domain_locs_sorted if domain[0] >= b[0] and domain[1] <= b[1]]
            if len(small_domains) == 0:
                final_domain.append(b)
            else:
                start = b[0]
                end = start
                small_domains_len = len(small_domains)    
                n = 0
                while n < small_domains_len:
                    tmp_this = small_domains[n]
                    end = tmp_this[1]
                    if end - start + 1 > 600:
                        #print('whlie',n,start,tmp_this[0])
                        final_domain.append((start,tmp_this[0]))
                        start = tmp_this[0]
                        end = start
                    n += 1
                    #print('add',n)
                    if n == small_domains_len:
                        #print('n == tmp_domains_len')
                        if end < b[1]:
                            #print('last',start,this[0])
                            final_domain.append((start,b[1]))
    out_file = bedpe_file_name.replace('.bedpe','') + '_final_domain_locs.bedpe'
    of = open(out_file,'w')
    of.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
    for domain in final_domain:
        of.write(chrom+'\t'+str(domain[0])+'\t'+str(domain[1])+'\t'+chrom+'\t'+str(domain[0])+'\t'+str(domain[1])+'\n')
    of.close()
    return final_domain

def create_domain_constrain(domains,pairs,length):
    sbm = np.zeros((length,length))
    rm_single_pairs = []
    pairs_len = len(pairs)
    pairs_filter = 0
    for pair in pairs:
        tmp_last = (pair[0]-1,pair[1]+1)
        tmp_next = (pair[0]+1,pair[1]-1)
        if (tmp_last in pairs) or (tmp_next in pairs):
            rm_single_pairs.append(pair)
        else:
            pairs_filter += 1
    print('pairs_len',pairs_len)
    print('pairs_filter',pairs_filter)
    for pair in rm_single_pairs:
        sbm[pair[0],pair[1]] = 1
    for domain in domains:
        file_name = str(domain[0])+"_"+str(domain[1])+".const"
        out = open(file_name,'w')
        out.write('DS:'+'\n'+'-1'+'\n'+'SS:'+'\n'+'-1'+'\n'+'Mod:'+'\n'+'-1'+'\n'+'Pairs:'+'\n')
        start = domain[0]
        end = domain[1]
        tmp_matrix = sbm[start:end,start:end]
        ones_bool = (tmp_matrix == 1)
        indices = np.where(ones_bool)
        for row, col in zip(indices[0], indices[1]):
            if row > col :
                continue
            out.write(str(row+1)+' '+str(col+1)+'\n')
        out.write('-1 -1'+'\n'+'FMN:'+'\n'+'-1'+'\n'+'Forbids:'+'\n'+'-1 -1'+'\n')
        out.close()

def creat_fasta_from_domain(domains,fasta_list):
    prefixs = []
    for domain in domains:
        prefixs.append(str(domain[0])+"_"+str(domain[1]))
        file_name = str(domain[0])+"_"+str(domain[1])+".fasta"
        out = open(file_name,'w')
        seq = fasta_list[domain[0]:domain[1]]
        seq = "".join(seq)
        out.write('>'+str(domain[0])+"_"+str(domain[1])+"\n")
        out.write(seq)
        out.close()
    with open('domain.txt','w')as f:
        for prefix in prefixs:
            f.write(prefix+"\n")
    return prefixs

def select_lowest_energy_ct_and_return_all_bedpe_and_support(ct_name,ct_start,ct_end,chrom):
    ctf = open(ct_name,'r')
    ct_pairs_dict = {}
    ct_energy = {}
    n = 0
    for line in ctf:
        if 'ENERGY' in line :
            n += 1
            structure_name = 's' + str(n)
            ct_pairs_dict[structure_name] = []
            sub = line.strip().split()
            if len(sub) < 4:
                ct_energy[structure_name] = 10000
            else:
                #print(float(sub[3]))
                ct_energy[structure_name] = float(sub[3])
        else:
            sub = line.strip().split()
            if len(sub) < 4 :
                print(line)
                continue
            if int(sub[4]) == 0 :
                continue
            left = int(ct_start) + int(sub[0])
            right = int(ct_start) + int(sub[4])
            if left > right:
                continue
            ct_pairs_dict[structure_name].append((left,right))
    max_ct = None
    min_value = np.inf
    for s in ct_energy.keys():
        if ct_energy[s] < min_value:
            min_value = ct_energy[s]
            max_ct = s
    #print(max_ct,ct_energy[max_ct])
    max_ct_dict = {}
    for s in ct_pairs_dict:
        bedpe_name = '../domain_structure/'  + str(ct_start) + "_" + str(ct_end) + "_" + s + ".bedpe"
        bf = open(bedpe_name,'w')
        bf.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
        pairs = ct_pairs_dict[s]
        for pair in pairs:
            bf.write(chrom+'\t'+str(pair[0]-1)+'\t'+str(pair[0])+'\t'+chrom+'\t'+str(pair[1]-1)+'\t'+str(pair[1])+'\n')
            if s == max_ct:
                max_ct_dict[str(pair[0])] = str(pair[1])
                max_ct_dict[str(pair[1])] = str(pair[0])
        bf.close()
    ctf.close()
    return max_ct_dict

def merge_ct_to_whole_structure(prefixs,seq_list,length,chrom):
    pairs = {}
    bedpe_file_energy = '../merge_structure'
    bedpe_file_energy_path = Path(bedpe_file_energy)
    if not bedpe_file_energy_path.exists():
        os.mkdir(bedpe_file_energy)
    for prefix in prefixs:
        file_name = '../domain_structure/' + prefix + ".ct"
        start = int(prefix.split('_')[0])
        end = int(prefix.split('_')[1])
        best_ct = select_lowest_energy_ct_and_return_all_bedpe_and_support(file_name,start,end,chrom)
        pairs = dict(pairs,**best_ct)
    wc = open(bedpe_file_energy+'/'+'whole_structure.ct','w')
    wc.write(str(length)+"\n")
    for i in range(1,length):
        right = int(pairs.get(str(i),0))
        wc.write(str(i)+' '+seq_list[i-1]+' '+str(i-1)+' '+str(i+1)+' '+str(right)+' '+str(i)+'\n')
    wc.close()
    wb = open(bedpe_file_energy+'/'+'whole_structure.bedpe','w')
    wb.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
    for pair_left in pairs.keys():
        pair_right = pairs[pair_left]
        wb.write(chrom+'\t'+str(int(pair_left)-1)+'\t'+str(int(pair_left))+'\t'+chrom+'\t'+str(int(pair_right)-1)+'\t'+str(int(pair_right))+'\n')
    wb.close()

def pad_matrix(matrix, target_rows, target_cols):
    current_rows, current_cols = matrix.shape
    rows_to_pad = max(0, target_rows - current_rows)
    cols_to_pad = max(0, target_cols - current_cols)
    padded_matrix = np.pad(matrix, ((0, rows_to_pad), (0, cols_to_pad)), mode='constant', constant_values=0)
    return padded_matrix

#################################################################################
pyname, partition_out_file_id_file, canny_bedpe, rri_matrix_10, rri_matrix_5, rri_matrix_1, length, chrom, fasta_file, sequence_id= sys.argv

print("--------------step3 start------------------")
print_time()

print("--------------------load_process_data------------------")
print_time()
if not os.path.exists(path='./step3_optimal_structure_predict'):
        os.mkdir(path='./step3_optimal_structure_predict')
os.chdir('./step3_optimal_structure_predict')

partition_file_path = '../step1_partition/'
canny_file_path = '../step2_structure_feature_detect/'
matrix_file_path = '../step2_structure_feature_detect/'
fasta_file_path = '../step0_prepare/'

length = int(length)

seq_list = load_fasta(fasta_file_path+fasta_file,[sequence_id])[sequence_id]

all_partition_matrix = create_all_partition_matrixs(partition_out_file_id_file,partition_file_path,length)
all_partition_matrix_diagonal_processing = diagonal_1000_processing(all_partition_matrix,length)

raw_canny = load_canny_bedpe(canny_bedpe,canny_file_path,length)
raw_oe_matrix_5 = extract_oe_matrix(rri_matrix_5,5,matrix_file_path)
canny_oe_mask = mask_diagonal(raw_canny,raw_oe_matrix_5)
expanded_canny = expand_matrix(canny_oe_mask,5,1,length)

expanded_matrix_10 = expand_matrix(extract_matrix(rri_matrix_10,10,matrix_file_path),10,1,length)
expanded_matrix_5 = expand_matrix(extract_matrix(rri_matrix_5,5,matrix_file_path),5,1,length)
matrix_1 =  pad_matrix(extract_matrix(rri_matrix_1,1,matrix_file_path),length,length)[:length,:length]

structure_feature_matrix = canny_partition_intersection(expanded_canny,all_partition_matrix_diagonal_processing,'structure_feature.bedpe',length,chrom)
print_time()
print("@@@@@@@@@@@@@@@@@@@@load_process_data_end@@@@@@@@@@@@@@@@@@")

print("--------------------number_structural_unit------------------")
print_time()
units_loc, units_name = number_structural_unit(structure_feature_matrix,chrom)
print("units_len",len(units_name))
print_time()
print("@@@@@@@@@@@@@@@@@@@@number_structural_unit_end@@@@@@@@@@@@@@@@@@")

print("--------------------calculate_unit_read_count------------------")
print_time()
units_count = calculate_unit_read_count(units_loc,expanded_matrix_10,expanded_matrix_5,matrix_1,structure_feature_matrix,length)
print_time()
print("@@@@@@@@@@@@@@@@@@@@calculate_unit_read_count_end@@@@@@@@@@@@@@@@@@")

print("--------------------build_structure_tree------------------")
print_time()
tree,paths = build_structure_tree(units_count,units_loc,length)
print_time()
print("@@@@@@@@@@@@@@@@@@@@build_structure_tree_end@@@@@@@@@@@@@@@@@@")

print("--------------------make_sequence_constrain------------------")
print_time()
if not os.path.exists(path='./sequence_constrain'):
        os.mkdir(path='./sequence_constrain')
os.chdir('./sequence_constrain')
n = 0
structure_bedpe_list = []
for path in paths:
    n += 1
    bedpe_file_title = 'structure' + str(n) + '.bedpe'
    structure_bedpe_list.append(bedpe_file_title)
    pairs = create_structure_bedpe(path,units_loc,bedpe_file_title,chrom)

for structure_bedpe in structure_bedpe_list:
    domain_locs = split_domain(bedpe_file_title,length,chrom)
    final_domains = connect_domain_and_fill_blank(domain_locs,bedpe_file_title,length,chrom)
    create_domain_constrain(final_domains,pairs,length)
    prefixs = creat_fasta_from_domain(final_domains,seq_list)

    fold_structure(prefixs)

    merge_ct_to_whole_structure(prefixs,seq_list,length,chrom)

os.chdir('..')
print_time()
print("@@@@@@@@@@@@@@@@@@@@make_sequence_constrain_end@@@@@@@@@@@@@@@@@@")

print("--------------step3 end------------------")
print_time()
