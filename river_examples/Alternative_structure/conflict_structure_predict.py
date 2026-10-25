import os
import numpy as np
import sys
from scipy import sparse
import itertools

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

def extract_matrix(file_name,res):
    data = np.loadtxt(file_name, usecols=[0,1,2], dtype=[('S','<i4'),('E','<i4'),('value','<f8')])
    data['S'] = data['S'] / res
    data['E'] = data['E'] / res
    data['value'][data['value'] < 0] = 0
    data['value'][np.isnan(data['value'])] = 0
    matrix = get_matrix_npz(data,data)
    return matrix

def get_shannon(matrix, n, res):
    (x,y) = np.diag_indices_from(matrix)
    shannon_ma = np.zeros_like(matrix)
    shan_out = []
    true_shan_out = []
    x = np.delete(x, 0)
    y = np.delete(y, -1)
    for i in np.arange(n-1):
        shannon_ma[x,y] = matrix[x,y].copy()
        shannon_ma[y,x] = matrix[y,x].copy()
        x = np.delete(x, 0)
        y = np.delete(y, -1)
    shannon_sum = shannon_ma.sum(axis=1)
    p_ma = shannon_ma / shannon_sum
    p_ma[np.isnan(p_ma)] = 0
    for i,arrs in enumerate(p_ma.T):
        p = arrs[arrs>0]
        shan = np.sum(p*np.log2(p)) * -1
        shan_out.append(shan)
        for iters in range(res):
            true_shan_out.append(shan)       
    return np.array(shan_out), np.array(true_shan_out)

def select_cluster_with_high_shannon(clusters_locs,shannon,out_file):
    tmp = []
    of = open(out_file,'w')
    of.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
    average = shannon.mean()
    index_list = set()
    for index,value in enumerate(shannon):
        if value >= average:
            index_list.add(index)
    for cluster in clusters_locs:
        sub = cluster.split("_")
        if int(sub[2]) - int(sub[0]) >= 3000:
            continue
        start = int(sub[2])
        end = int(sub[3])
        tmp_index_list = set([i for i in range(start,end+1)])
        if index_list & tmp_index_list:
            tmp.append(cluster)
            of.write('HS1'+"\t"+sub[0]+"\t"+sub[1]+"\t"+'HS1'+"\t"+sub[2]+"\t"+sub[3]+"\n")
    of.close()
    return tmp

class Matrix:
    def __init__(self, matrix, ostream = sys.stdout):
        self.matrix = matrix
        self.shape = matrix.shape
        self.ostream = ostream
    
    def expend_matrix(self, old_res, new_res):
        tmp_matrix = np.zeros((29903,29903))
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                time = int(old_res/new_res)
                l = row * time
                r = col * time
                l_end = l + time if l + time <= 29902 else 29903
                r_end = r + time if r + time <= 29902 else 29903
                tmp_matrix[l:l_end,r:r_end] = self.matrix[row,col]
        self.matrix = tmp_matrix

def load_canny_bedpe(infile):
    tmp_matrix = np.zeros((2991,2991))
    with open(infile,"r")as f:
        for line in f:
            if 'chrom1' in line:
                continue
            sub = line.strip().split()
            tmp_matrix[int(int(sub[1])/10),int(int(sub[4])/10)] = 1
    return tmp_matrix

def diagonal_inside_and_out_filter_conflict(conflict_clusters,structure_feature,out_file):
    n = 0
    filter_clusters = {}
    out_clusters = []
    for cluster in conflict_clusters:
        n += 1
        conflict_id = 'conflict_' + str(n)
        sub = cluster.split("_")
        tmp_matrix = structure_feature[int(sub[0]):int(sub[1]),int(sub[2]):int(sub[3])]
        matrix_sum = tmp_matrix.sum()
        if matrix_sum != 0:
            filter_clusters[conflict_id] = [int(sub[0]),int(sub[1]),int(sub[2]),int(sub[3])]
            out_clusters.append(cluster)
    of = open(out_file,'w')
    of.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
    for cluster in out_clusters:
        sub = cluster.split("_")
        of.write('HS1'+'\t'+str(sub[0])+'\t'+str(sub[1])+'\t'+'HS1'+'\t'+str(sub[2])+'\t'+str(sub[3])+'\n')
    of.close()
    return filter_clusters

def filter_conflict(conflict_clusters,structure_feature,out_file):
    filter_clusters = {}
    for cluster in list(conflict_clusters):
        conflict_list = conflict_clusters[cluster]
        sub = cluster.split("_")
        tmp_matrix = structure_feature[int(sub[0]):int(sub[1]),int(sub[2]):int(sub[3])]
        matrix_sum = tmp_matrix.sum()
        if matrix_sum == 0:
            del conflict_clusters[cluster]
            for tmp_cluster in conflict_list:
                tmp_cluster_list = conflict_clusters[tmp_cluster]
                tmp_cluster_list.remove(cluster)
                conflict_clusters[tmp_cluster] = tmp_cluster_list
    for cluster in conflict_clusters:
        cluster_list = conflict_clusters[cluster]
        if len(cluster_list) != 0:
            filter_clusters[cluster] = cluster_list
    of = open(out_file,'w')
    of.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
    for cluster in filter_clusters:
        sub = cluster.split("_")
        of.write('HS1'+'\t'+str(sub[0])+'\t'+str(sub[1])+'\t'+'HS1'+'\t'+str(sub[2])+'\t'+str(sub[3])+'\n')
    of.close()
    return filter_clusters

def select_conflict_structure(shannon_cluters,long_clusters,out_file):
    lc_set = set()
    for cluster in long_clusters:
        sub = cluster.split("_")
        tmp = (int(sub[0]),int(sub[1]),int(sub[2]),int(sub[3]))
        lc_set.add(tmp)
    distant_candidate_cluster = []
    diagonal_candidate_cluster = []
    diagonal_inside_and_out_conflict = []
    for cluster in shannon_cluters:
        sub = cluster.split("_")
        start1, end1, start2, end2 = int(sub[0]),int(sub[1]),int(sub[2]),int(sub[3])
        start1 = int(start1) - 10
        end1 = int(end1) + 10
        start2 = int(start2) - 10
        end2 = int(end2) + 10
        one = set([i for i in range(start1,end1+10)])
        two = set([i for i in range(start2,end2+10)])
        tmp = (int(sub[0]),int(sub[1]),int(sub[2]),int(sub[3]))
        if not one & two:
            if tmp in lc_set:
                distant_candidate_cluster.append(tmp)
                tmp = list(map(str, list(tmp)))
                diagonal_inside_and_out_conflict.append("_".join(tmp))
        else:
           diagonal_candidate_cluster.append(tmp)

    diagonal_candidate_cluster_pairs = list(itertools.combinations(diagonal_candidate_cluster,2))
    diagonal_candidate_cluster_pairs = [comb for comb in diagonal_candidate_cluster_pairs if comb[0] != comb[1]]
    distant_candidate_cluster_pairs = list(itertools.combinations(distant_candidate_cluster,2))
    distant_candidate_cluster_pairs = [comb for comb in distant_candidate_cluster_pairs if comb[0] != comb[1]]
    cross_conflict = {}
    vertical_horizontal_conflict = {}
    diagonal_conflict = {}
    for type in [diagonal_candidate_cluster_pairs,distant_candidate_cluster_pairs]:
        if type == diagonal_candidate_cluster_pairs:
            diagonal = True
            distant = False
        if type == distant_candidate_cluster_pairs:
            distant = True
            diagonal = False
        for pair in type:
            conflicted = False
            cross = False
            vertical_horizontal = False
            one = pair[0]
            one_left_start, one_left_end, one_right_start, one_right_end = one
            one_left_len = one_left_end - one_left_start
            one_right_len = one_right_end - one_right_start
            two = pair[1]
            two_left_start, two_left_end, two_right_start, two_right_end = two
            two_left_len = two_left_end - two_left_start
            two_right_len = two_right_end - two_right_start
            one = list(map(str, one))
            two = list(map(str, two))
            #print("_".join(one))
            #print("_".join(two))
            if ((one_left_start <= two_left_start) and (one_left_end >= two_left_end)) or ((one_left_start >= two_left_start) and (one_left_end <= two_left_end)):
                if diagonal:
                    conflicted = True
                if distant:
                    vertical_horizontal = True
            elif (one_left_start <= two_left_start) and (one_left_end >= two_left_start):
                overlap = one_left_end - two_left_start
                if (overlap > (one_left_len/2)) or (overlap > (two_left_len/2)):
                    if diagonal:
                        conflicted = True
                    if distant:
                        vertical_horizontal = True
            elif (one_left_start >= two_left_start) and (one_left_start <= two_left_end):
                overlap = two_left_end - one_left_start
                if (overlap > (one_left_len/2)) or (overlap > (two_left_len/2)):
                    if diagonal:
                        conflicted = True
                    if distant:
                        vertical_horizontal = True
            else:
                pass

            if ((one_right_start <= two_right_start) and (one_right_end >= two_right_end)) or ((one_right_start >= two_right_start) and (one_right_end <= two_right_end)):
                if diagonal:
                    conflicted = True
                if distant:
                    vertical_horizontal = True
            elif (one_right_start <= two_right_start) and (one_right_end >= two_right_start):
                overlap = one_right_end - two_right_start
                if (overlap > (one_right_len/2)) or (overlap > (two_right_len/2)):
                    if diagonal:
                        conflicted = True
                    if distant:
                        vertical_horizontal = True
            elif (one_right_start >= two_right_start) and (one_right_start <= two_right_end):
                overlap = two_right_end - one_right_start
                if (overlap > (one_right_len/2)) or (overlap > (two_right_len/2)):
                    if diagonal:
                        conflicted = True
                    if distant:
                        vertical_horizontal = True
            else:
                pass

            if ((one_right_start <= two_left_start) and (one_right_end >= two_left_end)) or ((one_right_start >= two_left_start) and (one_right_end <= two_left_end)):
                if distant:
                    cross = True
            elif (one_right_start <= two_left_start) and (one_right_end >= two_left_start):
                overlap = one_right_end - two_left_start
                if (overlap > (one_right_len/2)) or (overlap > (two_left_len/2)):
                    if distant:
                        cross = True
            elif (one_right_start >= two_left_start) and (one_right_start <= two_left_end):
                overlap = two_left_end - one_right_start
                if (overlap > (one_right_len/2)) or (overlap > (two_left_len/2)):
                    if distant:
                        cross = True
            else:
                pass 

            if ((one_left_start <= two_right_start) and (one_left_end >= two_right_end)) or ((one_left_start >= two_right_start) and (one_left_end <= two_right_end)):
                if distant:
                    cross = True
            elif (one_left_start <= two_right_start) and (one_left_end >= two_right_start):
                overlap = one_left_end - two_right_start
                if (overlap > (one_left_len/2)) or (overlap > (two_right_len/2)):
                    if distant:
                        cross = True
            elif (one_left_start >= two_right_start) and (one_left_start <= two_right_end):
                overlap = two_right_end - one_left_start
                if (overlap > (one_left_len/2)) or (overlap > (two_right_len/2)):
                    if distant:
                        cross = True
            else:
                pass
        
            if conflicted or cross or vertical_horizontal:
                tmp_one = "_".join(one)
                tmp_two = "_".join(two)
                if diagonal:
                    if diagonal_conflict.get(tmp_one):
                        diagonal_conflict[tmp_one].append(tmp_two)
                    else:
                        diagonal_conflict[tmp_one] = [tmp_two]
                    if diagonal_conflict.get(tmp_two):
                        diagonal_conflict[tmp_two].append(tmp_one)
                    else:
                        diagonal_conflict[tmp_two] = [tmp_one]
                if distant:
                    if cross:
                        if cross_conflict.get(tmp_one):
                            cross_conflict[tmp_one].append(tmp_two)
                        else:
                            cross_conflict[tmp_one] = [tmp_two]
                        if cross_conflict.get(tmp_two):
                            cross_conflict[tmp_two].append(tmp_one)
                        else:
                            cross_conflict[tmp_two] = [tmp_one]
                    if vertical_horizontal:
                        if vertical_horizontal_conflict.get(tmp_one):
                            vertical_horizontal_conflict[tmp_one].append(tmp_two)
                        else:
                            vertical_horizontal_conflict[tmp_one] = [tmp_two]
                        if vertical_horizontal_conflict.get(tmp_two):
                            vertical_horizontal_conflict[tmp_two].append(tmp_one)
                        else:
                            vertical_horizontal_conflict[tmp_two] = [tmp_one]
    #print('warn',conflict['830_1020_840_1040'])
    cross_out = out_file + "_cross.bedpe"
    vertical_horizontal_out = out_file + "_vertical_horizontal.bedpe"
    diagonal_out = out_file + "_diagonal.bedpe"
    for out in [cross_out,vertical_horizontal_out,diagonal_out]:
        writed = set()
        of = open(out,"w")
        of.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
        if out == cross_out:
            conflict = cross_conflict
        elif out == vertical_horizontal_out:
            conflict = vertical_horizontal_conflict
        elif out == diagonal_out:
            conflict = diagonal_conflict
        for cluster in conflict:
            conflict_clusters = conflict[cluster]
            #print(conflict_clusters)
            for conflict_cluster in conflict_clusters:
                if conflict_cluster in writed:
                    continue
                else:
                    writed.add(conflict_cluster)
                sub = conflict_cluster.split("_")
                of.write('HS1'+'\t'+str(sub[0])+'\t'+str(sub[1])+'\t'+'HS1'+'\t'+str(sub[2])+'\t'+str(sub[3])+'\n')
        of.close()
    diagonal_inside_and_out_out = out_file + "_diagonal_inside_and_out.bedpe"
    of = open(diagonal_inside_and_out_out,"w")
    of.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
    for cluster in diagonal_inside_and_out_conflict:
        sub = cluster.split("_")
        of.write('HS1'+'\t'+str(sub[0])+'\t'+str(sub[1])+'\t'+'HS1'+'\t'+str(sub[2])+'\t'+str(sub[3])+'\n')
    of.close()
    return diagonal_conflict,cross_conflict,vertical_horizontal_conflict,diagonal_inside_and_out_conflict

def select_filter_diagonal_non_diagonal_clusters(clusters_locs,rri_matirx,non_diagonal_out,diagonal_out):
    #candidate_clusters = []
    non_diagonal_clusters = []
    diagonal_clusters = []
    for cluster in clusters_locs:
        sub = cluster.split("_")
        if int(sub[3]) - int(sub[0]) >= 3000:
            continue
        value = np.sum(rri_matirx[int(sub[0]):int(sub[1]),int(sub[2]):int(sub[3])])
        #candidate_clusters.append((cluster,value))
        left = set(list(range(int(sub[0]),int(sub[1])+10)))
        right = set(list(range(int(sub[2])-10,int(sub[3]))))
        if left & right:
            diagonal_clusters.append(cluster)
        else:
            non_diagonal_clusters.append((cluster,value))
    values = [value for _,value in non_diagonal_clusters]
    cut_off = np.percentile(values,50)
    out_clusters = [loc for loc,value in non_diagonal_clusters if value >= cut_off]
    failed_clusters = [loc for loc,value in non_diagonal_clusters if value < cut_off]
    of = open(non_diagonal_out,"w")
    of.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
    for cluster in out_clusters:
        sub = cluster.split("_")
        of.write('HS1'+'\t'+str(sub[0])+'\t'+str(sub[1])+'\t'+'HS1'+'\t'+str(sub[2])+'\t'+str(sub[3])+'\n')
    of.close()
    of = open(diagonal_out,"w")
    of.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
    for cluster in diagonal_clusters:
        sub = cluster.split("_")
        of.write('HS1'+'\t'+str(sub[0])+'\t'+str(sub[1])+'\t'+'HS1'+'\t'+str(sub[2])+'\t'+str(sub[3])+'\n')
    of.close()
    with open('filed_non_diagonal_clusters_clusters.bedpe','w')as f:
        for cluster in failed_clusters:
            sub = cluster.split("_")
            f.write('HS1'+'\t'+str(sub[0])+'\t'+str(sub[1])+'\t'+'HS1'+'\t'+str(sub[2])+'\t'+str(sub[3])+'\n')
    return out_clusters,diagonal_clusters

def load_clusters(infile):
    clusters = []
    with open(infile,'r')as f:
        for line in f:
            if 'chrom1' in line:
                continue
            else:
                sub = line.strip().split()
                clusters.append("_".join([str(sub[1]),str(sub[2]),str(sub[4]),str(sub[5])]))
    return clusters

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
    
def number_structural_unit_within_cluster(candidate_structure_matrix,cluster_loc):
    '''
    candidate_structure_matrix          matrix(29903,29903)      where pixel == 1 is candidate
    cluster_loc          [row_start,row_end,col_start,col_end]
    unit_loc      [pixelrow_pixelcol,......]           pixel_set for unit
    unit_name             [s1,........]               unit_num
    '''
    unit_loc = {}
    unit_name = []
    pixel_unit = {}
    checked_pixels = set()
    tmp_structure_num = 0
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    for row in range(int(cluster_loc[0]),int(cluster_loc[1])):
        for col in range(int(cluster_loc[2]),int(cluster_loc[3])):
            if (candidate_structure_matrix[row,col] > 0) and (str(row)+"_"+str(col) not in checked_pixels):
                one += 1
                if (row + 1 >= int(cluster_loc[1])) or (col - 1 < int(cluster_loc[2])):
                    five += 1
                    continue
                else:
                    if (candidate_structure_matrix[row+1,col-1] > 0):
                        two += 1
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
                three += 1
                if (row + 1 >= int(cluster_loc[1])) or (col - 1 < int(cluster_loc[2])):
                    continue
                else:
                    if (candidate_structure_matrix[row+1,col-1] > 0):
                        four += 1
                        structure_id = pixel_unit[str(row)+"_"+str(col)]
                        unit_loc[structure_id].append(str(row+1)+"_"+str(col-1))
                        pixel_unit[str(row+1)+"_"+str(col-1)] = structure_id
                        checked_pixels.add(str(row+1)+"_"+str(col-1))
            else:
                continue
    print(one,two,three,four,five)
    return unit_loc,unit_name

def calculate_unit_read_count(units_loc,rri_matrix_10,rri_matrix_5,rri_matrix_1,partition_matrix,average=None):
    """
    units_loc      {s1:[start_end,start_end,...],....}            unit pixel loc
    rri_matrix               matrix(29903,29903)
    units_support       [(s1,count),...]                     unit reads counts
    """
    units_support = []
    for unit in units_loc.keys():
        pixels_num = 0
        count = 0
        locs = units_loc[unit]
        for loc in locs:
            pixels_num += 1
            row = int(loc.split("_")[0])
            col = int(loc.split("_")[1])
            count += rri_matrix_10[row,col] + rri_matrix_5[row,col] + rri_matrix_1[row,col] + ((rri_matrix_10[row,col] + rri_matrix_5[row,col] + rri_matrix_1[row,col]) / partition_matrix[row,col])
        if average:
            count /= pixels_num
        units_support.append((unit,count))
    units_support_list = [count for (_,count) in units_support]
    if len(units_support_list) == 0:
        return None
    units_support_list_96 = np.percentile(units_support_list,96)
    units_support = [(structure,count) for (structure,count) in units_support if count >= units_support_list_96]
    return units_support

def pseudoknot_judgment(pre_path,units_loc,structure):
    pseudoknot = False
    structure_left = []
    structure_right = []
    structure_locs = units_loc[structure]
    for loc in structure_locs:
        sub = loc.split("_")
        structure_left.append(int(sub[0]))
        structure_right.append(int(sub[1]))
    structure_left_start = min(structure_left)
    structure_left_end = max(structure_left)
    structure_right_start = min(structure_right)
    structure_right_end = max(structure_right)
    for pre_structure in pre_path:
        if pre_structure == 's0':
            continue
        left = []
        right = []
        pre_strcutre_locs = units_loc[pre_structure]
        for loc in pre_strcutre_locs:
            sub = loc.split("_")
            left.append(int(sub[0]))
            right.append(int(sub[1]))
        left_start = min(left)
        left_end = max(left)
        right_start = min(right)
        right_end = max(right)
        if (left_end < structure_left_start) and (structure_left_end < right_start) and (structure_right_start > right_end):
            pseudoknot = True
        elif (structure_right_start > left_end) and (structure_right_end < right_start) and (structure_left_end < left_start):
            pseudoknot = True
        else:
            continue
    return pseudoknot

class Tree_Point(object):
    def __init__(self,num=None,father=None,son=None,structure=None,path=None):
        self.num = num
        self.father = father
        self.son = son
        self.structure = structure
        self.path = path

def conflict_judgment_filter_within_cluster_loc(tmp_next,unins_loc,pre_path,cluster_loc):
    not_conflict_structure = []
    path = pre_path[:]
    region_start = min([int(i) for i in cluster_loc])
    region_end = max([int(i) for i in cluster_loc])
    region_len = region_end - region_start + 1
    dot = np.zeros(region_len)
    for structure in path:
        if structure == 's0':
            continue
        structure_locs = unins_loc[structure]
        for structure_loc in structure_locs:
            sub = structure_loc.split("_")
            dot[int(sub[0])-region_start] = 1
            dot[int(sub[1])-region_start] = 1
    for tmp_structure in tmp_next:
        pre_dot = dot
        tmp_structure_locs = unins_loc[tmp_structure]
        retain = True
        for tmp_structure_loc in tmp_structure_locs:
            sub = tmp_structure_loc.split("_")
            if (pre_dot[int(sub[0])-region_start] == 1) or (pre_dot[int(sub[1])-region_start] == 1):
                retain = False
                break
        if pseudoknot_judgment(path,unins_loc,tmp_structure):
            retain = False
        if retain:
            not_conflict_structure.append(tmp_structure)
    return not_conflict_structure

def build_structure_tree_within_cluster(structure_count,unins_loc,cluster_loc):
    paths = []
    root = Tree_Point(num='point0',father=None,son=[],structure='s0',path=['s0'])
    #print(root.path)
    sorted_structure_count = sorted(structure_count,key=lambda x:x[1],reverse=True)
    tmp_sorted_structure_count = sorted_structure_count
    if len(tmp_sorted_structure_count) == 0:
        return None,None
    #print(sorted_structure_count)
    point_num = 0
    tmp_next_round_point = []
    next_round_structure = []
    #print(tmp_next_strucure)
    pre_path = root.path[:]
    #print('pre_path',pre_path)
    while (len(next_round_structure) == 0):
                tmp_next_strucure = []
                tmp_max_value = tmp_sorted_structure_count[0][1]
                next_positions = [index for index, (_, value) in enumerate(tmp_sorted_structure_count) if value == tmp_max_value]
                #print('next_positions',next_positions)
                #print(tmp_structure_count_rm_last_structure)
                for index in next_positions:
                    tmp_next_strucure.append(tmp_sorted_structure_count[index][0])
                #print('tmp_next_strucure',tmp_next_strucure)
                next_round_structure = conflict_judgment_filter_within_cluster_loc(tmp_next_strucure,unins_loc,pre_path,cluster_loc)
                #print('tmp_path',tmp_path)
                #print('next_round_structure',next_round_structure)
                if len(next_positions) == 0:
                    pass
                else:
                    tmp_sorted_structure_count = [(structure, value) for index, (structure, value) in enumerate(tmp_sorted_structure_count) if index not in next_positions]
    #print('root.path',root.path)
    #print('next_round_structure',next_round_structure)
    for structure in next_round_structure:
        point_num += 1
        point_id = 'point' + str(point_num)
        tmp_path = root.path[:]
        tmp_path.append(structure)
        #print('tmp_path',tmp_path)
        tmp_next_round_point.append(Tree_Point(num=point_id,father=root,son=[],structure=structure,path=tmp_path))
    root.son = tmp_next_round_point
    next_round_point = tmp_next_round_point
    #print(root.num,root.father,root.son,root.structure,root.path)
    #print(next_round_point[0].num,next_round_point[0].father,next_round_point[0].son,next_round_point[0].structure,next_round_point[0].path)

    while(next_round_point):
        tmp_next_round_point = []
        for point in next_round_point:
            #print('father',point.father.num)
            #print(point.num)
            point_son = []
            tmp_next_strucure = []
            next_round_structure = []
            tmp_path = point.path[:]
            end = False
            #print('tmp_path',tmp_path)
            tmp_structure_count = sorted_structure_count
            last_structure = tmp_path[-1]
            last_structure_value = [value for (structure, value) in sorted_structure_count if structure == last_structure][0]
            tmp_structure_count_rm_last_structure = [(structure, value) for (structure, value) in tmp_structure_count if float(value) < float(last_structure_value)]
            if len(tmp_structure_count_rm_last_structure) == 0:
                paths.append(tmp_path)
                continue
            #print(tmp_structure_count_rm_last_structure)
            while (len(next_round_structure) == 0):
                tmp_next_strucure = []
                tmp_max_value = tmp_structure_count_rm_last_structure[0][1]
                next_positions = [index for index, (_, value) in enumerate(tmp_structure_count_rm_last_structure) if value == tmp_max_value]
                #print('next_positions',next_positions)
                #print(tmp_structure_count_rm_last_structure)
                for index in next_positions:
                    tmp_next_strucure.append(tmp_structure_count_rm_last_structure[index][0])
                #print('tmp_next_strucure',tmp_next_strucure)
                next_round_structure = conflict_judgment_filter_within_cluster_loc(tmp_next_strucure,unins_loc,tmp_path,cluster_loc)
                #print('tmp_path',tmp_path)
                #print('next_round_structure',next_round_structure)
                if len(next_positions) == 0:
                    pass
                else:
                    tmp_structure_count_rm_last_structure = [(structure, value) for index, (structure, value) in enumerate(tmp_structure_count_rm_last_structure) if index not in next_positions]
                #print(tmp_structure_count_rm_last_structure)
                if len(tmp_structure_count_rm_last_structure) == 0:
                    end = True
                    break
            if end:
                #print("end")
                paths.append(tmp_path)
                #print(tmp_path)
                continue
            #print('next_round_structure',next_round_structure)
            for structure in next_round_structure:
                #print('structure',structure)
                point_num += 1
                point_id = 'point' + str(point_num)
                point_path = tmp_path[:]
                point_path.append(structure)
                #tmp_next_round_point.append(Tree_Point(num=point_id,father=point,son=[],structure=structure,path=point_path))
                point_son.append(Tree_Point(num=point_id,father=point,son=[],structure=structure,path=point_path))
                tmp_next_round_point.extend(point_son)
            point.son = point_son
            #print(point.father.num,point.father.structure,point.num,point.path,point.structure,point.son[0].num,point.son[0].path,point.son[0].structure)
        next_round_point = tmp_next_round_point
        #print(next_round_point[0].num,next_round_point[0].father,next_round_point[0].son,next_round_point[0].structure,next_round_point[0].path)
    paths_mvd = [list(path) for path in set(tuple(path) for path in paths)]
    return root,paths_mvd

def make_cluster_sequence(seq,cluster_loc,out_file,file_path,diagonal=False):
    if diagonal:
        sub = cluster_loc.split("_")
        start = int(sub[0])
        end = int(sub[1])
    else:
        sub = cluster_loc.split("_")
        left_start, left_end, right_start, right_end = sub
        start = min(int(left_start),int(right_start))
        end = max(int(left_end),int(right_end))
    sequence = seq[start:end]
    sequence = "".join(sequence)
    out_path = './'+file_path+'/' + out_file
    of = open(out_path,'w')
    of.write(">"+cluster_loc+"\n")
    of.write(sequence+"\n")
    return start,end

def create_structure_bedpe(structure_list,structure_loc,out_file,file_path):
    out_path = './'+file_path+'/' + out_file
    out = open(out_path,"w")
    for structure in structure_list:
        if structure == 's0':
            continue
        locs = structure_loc[structure]
        for loc in locs:
            sub = loc.split("_")
            out.write("HS1"+"\t"+str(int(sub[0]))+"\t"+str(int(sub[0])+1)+"\t"+"HS1"+"\t"+str(int(sub[1]))+"\t"+str(int(sub[1])+1)+"\n")
    out.close()

def structure_to_rnastructure_constraint(structure_list,structure_loc,out_file,start,file_path,no_pairing_area=None):
    out_path = './'+file_path+'/' + out_file
    out = open(out_path,"w")
    out.write('DS:'+'\n')
    out.write('-1'+'\n')
    out.write('SS:'+'\n')
    out.write('-1'+'\n')
    out.write('Mod:'+'\n')
    out.write('-1'+'\n')
    out.write('Pairs:'+'\n')
    for structure in structure_list:
        if structure == 's0':
            continue
        locs = structure_loc[structure]
        for loc in locs:
            sub = loc.split("_")
            out.write(str(int(sub[0])+1-start)+" "+str(int(sub[1])+1-start)+"\n")
    out.write('-1 -1'+'\n')
    out.write('FMN:'+'\n')
    out.write('-1'+'\n')
    out.write('Forbids:'+'\n')
    if no_pairing_area:
        left = list(range(int(no_pairing_area[0]),int(no_pairing_area[1])))
        right = list(range(int(no_pairing_area[2]),int(no_pairing_area[3])))
        for l in left:
            for r in right:
                out.write(str(l+1-start)+" "+str(r+1-start)+"\n")
    out.write('-1 -1'+'\n')
    out.close

def make_prohibit_constraint(out_file,start,file_path,no_pairing_area):
    out_path = './'+file_path+'/' + out_file
    out = open(out_path,"w")
    out.write('DS:'+'\n')
    out.write('-1'+'\n')
    out.write('SS:'+'\n')
    out.write('-1'+'\n')
    out.write('Mod:'+'\n')
    out.write('-1'+'\n')
    out.write('Pairs:'+'\n')
    out.write('-1 -1'+'\n')
    out.write('FMN:'+'\n')
    out.write('-1'+'\n')
    out.write('Forbids:'+'\n')
    if no_pairing_area:
        left = list(range(int(no_pairing_area[0]),int(no_pairing_area[1])))
        right = list(range(int(no_pairing_area[2]),int(no_pairing_area[3])))
        for l in left:
            for r in right:
                out.write(str(l+1-start)+" "+str(r+1-start)+"\n")
    out.write('-1 -1'+'\n')
    out.close

def make_structure_and_seq_diagonal_in_out(cluster_id,cluster_loc,seq_list,file_path,prohibited_area):
    '''
    paths     path list [[s1,s2,s3...],[],[],[]...]
    units_loc      {s1:[28830_28949,28831_28948],...}
    calculate_matrix    oe or kr matrix (29903,29903)
    cluster_id            cluster1...
    cluster_loc            [28830,28949,28831,28948]
    '''
    out = open("./"+file_path+"/all_file_name.info",'a')
    loc_out = open("./"+file_path+"/all_file_true_start.info",'a')
    n = 0
    cluster_loc = list(map(str,cluster_loc))
    cluster_loc = "_".join(cluster_loc)
    out_prefixs = []
    n += 1
    seq_file_title = cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc + '.fasta'
    constrain_file_title = cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc + '.const'
    out.write(cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc+"\n")
    start,end = make_cluster_sequence(seq_list,cluster_loc,seq_file_title,file_path)
    make_prohibit_constraint(constrain_file_title,start,file_path,prohibited_area)
    loc_out.write(cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc+"\t"+str(start)+"\t"+str(end)+"\n")
    out_prefixs.append(cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc)
    out.close()
    loc_out.close()
    return out_prefixs

def make_structure_and_seq(cluster_id,cluster_loc,seq_list,file_path):
    '''
    paths     path list [[s1,s2,s3...],[],[],[]...]
    units_loc      {s1:[28830_28949,28831_28948],...}
    calculate_matrix    oe or kr matrix (29903,29903)
    cluster_id            cluster1...
    cluster_loc            [28830,28949,28831,28948]
    '''
    out = open("./"+file_path+"/all_file_name.info",'a')
    loc_out = open("./"+file_path+"/all_file_true_start.info",'a')
    n = 0
    cluster_loc = list(map(str,cluster_loc))
    cluster_loc = "_".join(cluster_loc)
    out_prefixs = []
    n += 1
    seq_file_title = cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc + '.fasta'
    out.write(cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc+"\n")
    start,end = make_cluster_sequence(seq_list,cluster_loc,seq_file_title,file_path)
    loc_out.write(cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc+"\t"+str(start)+"\t"+str(end)+"\n")
    out_prefixs.append(cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc)
    out.close()
    loc_out.close()
    return out_prefixs

def make_structure_bedpe_constrain_and_seq(paths,units_loc,cluster_id,cluster_loc,seq_list,file_path,prohibited_area=None,diagonal=False):
    '''
    paths     path list [[s1,s2,s3...],[],[],[]...]
    units_loc      {s1:[28830_28949,28831_28948],...}
    calculate_matrix    oe or kr matrix (29903,29903)
    cluster_id            cluster1...
    cluster_loc            [28830,28949,28831,28948]
    '''
    out = open("./"+file_path+"/all_file_name.info",'a')
    loc_out = open("./"+file_path+"/all_file_true_start.info",'a')
    n = 0
    cluster_loc = list(map(str,cluster_loc))
    cluster_loc = "_".join(cluster_loc)
    out_prefixs = []
    for path in paths:
        n += 1
        seq_file_title = cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc + '.fasta'
        bdepe_file_title = cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc + '.bedpe'
        constrain_file_title = cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc + '.const'
        out.write(cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc+"\n")
        if diagonal:
            start,end = make_cluster_sequence(seq_list,cluster_loc,seq_file_title,diagonal=True)
        else:
            start,end = make_cluster_sequence(seq_list,cluster_loc,seq_file_title,file_path)
        loc_out.write(cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc+"\t"+str(start)+"\t"+str(end)+"\n")
        create_structure_bedpe(path,units_loc,bdepe_file_title,file_path)
        structure_to_rnastructure_constraint(path,units_loc,constrain_file_title,start,file_path,no_pairing_area=prohibited_area)
        out_prefixs.append(cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc)
    out.close()
    loc_out.close()
    return out_prefixs

def diagonal_conflict_make_constrain_seq(diagonal_conflict,seq_list,structure_feature_matrix,matrix_10,matrix_5,matrix_1,partition_matrix):
    file_path = 'diagonal_conflict_clusters'
    if not os.path.exists(path='./diagonal_conflict_clusters'):
        os.mkdir(path='./diagonal_conflict_clusters')
    
    constrain_prefixs = []
    without_constrain_prefixs = []

    conflict_set = set()
    for cluster in diagonal_conflict:
        conflict_clusters = diagonal_conflict[cluster]
        for conflict_cluster in conflict_clusters:
            merge1 = (cluster,conflict_cluster)
            merge2 = (conflict_cluster,cluster)
            if (merge1 in conflict_set) or (merge2 in conflict_set):
                continue
            else:
                conflict_set.add(merge1)
    n = 0
    conflict = {}
    clusters_loc = {}
    for pair in conflict_set:
        n += 1
        conflict_id = 'conflict_' + str(n)
        conflict[conflict_id] = [pair[0],pair[1]]
        sub0 = pair[0].split("_")
        sub1 = pair[1].split("_")
        clusters_loc[pair[0]] = [str(sub0[0]),str(sub0[1]),str(sub0[2]),str(sub0[3])]
        clusters_loc[pair[1]] = [str(sub1[0]),str(sub1[1]),str(sub1[2]),str(sub1[3])]

    for conflict_id in conflict:
        conflict_clusters = conflict[conflict_id]
        n = 0
        for cluster in conflict_clusters:
            cluster_loc = list(cluster.split("_"))
            units_loc, units_name = number_structural_unit_within_cluster(structure_feature_matrix,cluster_loc)
            units_count = calculate_unit_read_count(units_loc,matrix_10,matrix_5,matrix_1,partition_matrix)
            if units_count == None:
                print(cluster,'no substructure in cluster')
                n += 1 
                cluster_id = conflict_id + '_' + 'cluster_' + str(n) 
                prefixs = make_structure_and_seq(cluster_id,cluster_loc,seq_list,file_path)
                without_constrain_prefixs.extend(prefixs)
                continue
            tree,paths = build_structure_tree_within_cluster(units_count,units_loc,cluster_loc)
            if tree == None:
                continue
            n += 1 
            cluster_id = conflict_id + '_' + 'cluster_' + str(n) 
            prefixs = make_structure_bedpe_constrain_and_seq(paths,units_loc,cluster_id,cluster_loc,seq_list,file_path)
            constrain_prefixs.extend(prefixs)
    return constrain_prefixs,without_constrain_prefixs

def create_all_partition_matrixs(fileprefixs_file):
    fileprefixs = []
    with open(fileprefixs_file,"r")as f:
        for line in f:
            fileprefixs.append(line.strip())
    tmp_matrix = np.zeros((29903,29903))
    for fileprefix in fileprefixs:
        fileprefix_replace = fileprefix.replace(".txt","")
        sub = fileprefix_replace.split("_")
        left_start = int(sub[0])
        right_start = int(sub[2])
        tmp = open(fileprefix, 'r')
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

def diagonal_processing(all_partition_matrix):
    tmp = all_partition_matrix.copy()
    x,y = np.diag_indices_from(tmp)

    for time in range(10):
        tmp[x,y] = 0
        tmp[y,x] = 0
        x = np.delete(x,-1)
        y = np.delete(y,0)

    #rows, cols = np.ogrid[:29903, :29903]
    #mask = np.abs(rows - cols) >= 2500 
    #tmp[mask] = 0
    return tmp

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

def extract_oe_matrix(file_name,res):
    data = np.loadtxt(file_name, usecols=[0,1,2], dtype=[('S','<i4'),('E','<i4'),('value','<f8')])
    data['S'] = data['S'] / res
    data['E'] = data['E'] / res
    data['value'][data['value'] < 0] = 0
    data['value'][np.isnan(data['value'])] = 0
    mat = get_matrix_npz(data,data)
    oe = get_oe(mat)
    return oe

def mask_diagonal(raw_matrix,oe_matirx):
    tmp = raw_matrix.copy()
    ones_bool = (oe_matirx <= 1)
    indices = np.where(ones_bool)
    indices
    for row,col in zip(indices[0],indices[1]):
        tmp[row,col] = 0
    return tmp

def canny_partition_intersection(canny_matrix,partition_matrix):
    tmp_matrix = np.zeros((29903,29903))
    candidate_structure_loc = np.logical_and(canny_matrix == 1, partition_matrix > 0)
    tmp_matrix[candidate_structure_loc] = 1
    return tmp_matrix

def fold_structure_without_constrain(prefixs):
    for prefix in prefixs:
        print(prefix)
        seq = prefix + '.fasta'
        const = prefix + '.const'
        out = prefix + '.ct'
        os.system("Fold-smp %s %s"%(seq,out))
        print(prefix,'-------end-----')

def fold_structure(prefixs):
    for prefix in prefixs:
        print(prefix)
        seq = prefix + '.fasta'
        const = prefix + '.const'
        out = prefix + '.ct'
        os.system("Fold-smp %s %s -c %s"%(seq,out,const))
        print(prefix,'-------end-----')

def ct2bedpe(prefixs):
    ct_files_prefix = []
    for prefix in prefixs:
        ct_file = prefix + '.ct'
        chrom = 'HS1'
        ct = open(ct_file, "r")

        base_bedpe_name = prefix
        res = None
        start = int(base_bedpe_name.split("_")[-4])

        left_start = int(base_bedpe_name.split("_")[-4]) + 1
        left_end = int(base_bedpe_name.split("_")[-3])
        right_start = int(base_bedpe_name.split("_")[-2]) + 1
        right_end = int(base_bedpe_name.split("_")[-1])

        current_structure = None
        out = None

        for index, row in enumerate(ct):
            row = row.strip()
            if not row: 
                continue
            if 'ENERGY' in row:  
                if out is not None:
                    out.close()
                structure_index = 1 if current_structure is None else current_structure + 1
                if structure_index > 1:
                    current_structure = structure_index
                    continue
                bedpe = f'{base_bedpe_name}_structure{structure_index}.bedpe'
                ct_files_prefix.append(f'{base_bedpe_name}_structure{structure_index}')
                out = open(bedpe, "w")
                out.write('chrom1\tstart1\tend1\tchrom2\tstart2\tend2\n')
                current_structure = structure_index
                continue
            if current_structure == None:
                continue
            if current_structure > 1:
                    continue
            #if index == 0:
            sub = row.split()
            if len(sub) < 6:
                continue
            if int(sub[4]) != 0:
                left = int(sub[0]) + start
                right = int(sub[4]) + start
                if (left < left_start) or (left > left_end) or (right < right_start) or (right > right_end):
                    continue
                out.write(chrom+"\t" + str(left-1) + "\t" + str(left) + "\t"+chrom+"\t" + str(right-1) + "\t" + str(right) + "\n")
        ct.close()

        #if out is not None:
        #    out.close()
        #ct.close()
    return ct_files_prefix

def load_pair_bedpe(in_bedpe_prefix):
    pairs = {}
    file = in_bedpe_prefix + '.bedpe'
    ib = open(file, "r")
    for line in ib:
        line = line.strip().split()
        if 'chrom1' in line:
            continue
        else:
            left = int(line[2])
            right = int(line[5])
            pairs[left] = right
            pairs[right] = left
    ib.close()
    return pairs

def make_structure_ric_annotation_diagonal(infile_prefix,matrix_file,res):
    pairs = load_pair_bedpe(infile_prefix)
    matrix_file = '../'+matrix_file
    matrix = extract_matrix(matrix_file,res)
    locs = list(infile_prefix.split(".")[0].split('_')[-5:-1])
    locs = [int(i) for i in locs]
    length = max(locs) - min(locs)
    start = min(locs)
    structure_locs = list(range(1,length+1))
    value_list = []
    for i in structure_locs:
        i = i + start
        if i in pairs:
            value_list.append(matrix[int((i-1)//res),int((pairs[i]-1)//res)])
        else:
            value_list.append(0)
    out_file = "_".join(map(str,locs)) + "_annotation.txt"
    with open(out_file,'w')as f:
        for i in range(len(value_list)):
            f.write(str(structure_locs[i])+"\t"+str(value_list[i])+"\n")

def make_structure_ric_annotation(seq,infile_prefix,matrix_file,res):
    pairs = load_pair_bedpe(infile_prefix)
    matrix_file = '../'+matrix_file
    matrix = extract_matrix(matrix_file,res)
    locs = list(infile_prefix.split('_')[-5:-1])
    locs = [int(i) for i in locs]
    left_len = locs[1] - locs[0]
    right_len = locs[3] - locs[2]
    left_start = locs[0]
    left_end = locs[1]
    right_start = locs[2]
    right_end = locs[3]
    structure_locs = list(range(1,left_len+1+right_len))
    value_list = []
    for i in structure_locs:
        if i <= left_len:
            i = i + left_start
        elif i > left_len:
            i = i + right_start - left_len
        if i in pairs:
            value_list.append(matrix[int((i-1)//res),int((pairs[i]-1)//res)])
        else:
            value_list.append(0)
    out_file = "_".join(map(str,locs)) + "_annotation.txt"
    with open(out_file,'w')as f:
        for i in range(len(value_list)):
            f.write(str(structure_locs[i])+"\t"+str(value_list[i])+"\n")
    ct_out = "_".join(map(str,locs)) + "_remake.ct"
    sequence = seq[left_start:left_end] + seq[right_start:right_end]
    #print(left_start,left_end,right_start,right_end)
    #print(sequence)
    with open(ct_out,'w')as f:
        f.write(str(left_len+right_len)+"\n")
        for i in range(len(sequence)):
            i = i + 1
            if i <= left_len:
                pair_i = i + left_start
                if pair_i in pairs:
                    pair_j = pairs[pair_i] - right_start + left_len
                else:
                    pair_j = 0
            elif i > left_len:
                pair_i = i + right_start - left_len
                if pair_i in pairs:
                    pair_j = pairs[pair_i] - left_start
                else:
                    pair_j = 0
            f.write(str(i)+" "+sequence[i-1]+" "+str(i-1)+" "+str(i+1)+" "+str(pair_j)+" "+str(i)+"\n")

def make_constrain(structure_list,structure_loc,left_start,left_len,right_start,right_len):
    left_const = ["<" for i in range(left_len)]
    right_const = [">" for i in range(right_len)]
    for structure in structure_list:
        if structure == 's0':
            continue
        locs = structure_loc[structure]
        for loc in locs:
            sub = loc.split("_")
            left_const[int(sub[0])-left_start] = '('
            right_const[int(sub[1])-right_start+2] = ')'
    const = "".join(left_const) + '&' + "".join(right_const)
    return const

def make_cluster_sequence_constrain(seq,cluster_loc,out_file,structure_list,structure_loc,file_path):
    sub = cluster_loc.split("_")
    left_start, left_end, right_start, right_end = int(sub[0]),int(sub[1]),int(sub[2]),int(sub[3])
    left_seq = "".join(seq[left_start:left_end+2])
    left_len = len(left_seq)
    right_seq = "".join(seq[right_start-2:right_end])
    right_len = len(right_seq)
    sequence = left_seq + '&' + right_seq
    constrain = make_constrain(structure_list,structure_loc,left_start,left_len,right_start,right_len)
    out_path = './'+file_path+'/' + out_file
    of = open(out_path,'w')
    of.write(">"+cluster_loc+"\n")
    of.write(sequence+"\n")
    of.write(constrain)

def make_structure_constrain_and_seq(paths,units_loc,cluster_id,cluster_loc,seq_list,file_path):
    '''
    paths     path list [[s1,s2,s3...],[],[],[]...]
    units_loc      {s1:[28830_28949,28831_28948],...}
    calculate_matrix    oe or kr matrix (29903,29903)
    cluster_id            cluster1...
    cluster_loc            [28830,28949,28831,28948]
    '''
    out = open("./"+file_path+"/all_file_name.info",'a')
    n = 0
    cluster_loc = list(map(str,cluster_loc))
    cluster_loc = "_".join(cluster_loc)
    out_prefixs = []
    for path in paths:
        n += 1
        seq_constrain_file_title = cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc + '.fasta'
        make_cluster_sequence_constrain(seq_list,cluster_loc,seq_constrain_file_title,path,units_loc,file_path)
        out_prefixs.append(cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc)
        out.write(cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc+"\n")
    return out_prefixs

def cross_conflict_make_constrain_seq(cross_conflict,seq_list,structure_feature_matrix,matrix_10,matrix_5,matrix_1,partition_matrix):
    file_path = 'cross_conflict_clusters'
    if not os.path.exists(path='./cross_conflict_clusters'):
        os.mkdir(path='./cross_conflict_clusters')
    total_prefixs = []

    conflict_set = set()
    for cluster in cross_conflict:
        conflict_clusters = cross_conflict[cluster]
        for conflict_cluster in conflict_clusters:
            merge1 = (cluster,conflict_cluster)
            merge2 = (conflict_cluster,cluster)
            if (merge1 in conflict_set) or (merge2 in conflict_set):
                continue
            else:
                conflict_set.add(merge1)
    n = 0
    conflict = {}
    clusters_loc = {}
    for pair in conflict_set:
        n += 1
        conflict_id = 'conflict_' + str(n)
        conflict[conflict_id] = [pair[0],pair[1]]
        sub0 = pair[0].split("_")
        sub1 = pair[1].split("_")
        clusters_loc[pair[0]] = [str(sub0[0]),str(sub0[1]),str(sub0[2]),str(sub0[3])]
        clusters_loc[pair[1]] = [str(sub1[0]),str(sub1[1]),str(sub1[2]),str(sub1[3])]

    for conflict_id in conflict:
        conflict_clusters = conflict[conflict_id]
        n = 0
        for cluster in conflict_clusters:
            cluster_loc = clusters_loc[cluster]
            units_loc, units_name = number_structural_unit_within_cluster(structure_feature_matrix,cluster_loc)
            units_count = calculate_unit_read_count(units_loc,matrix_10,matrix_5,matrix_1,partition_matrix)
            if units_count == None:
                print(cluster,'no substructure in cluster')
                continue
            tree,paths = build_structure_tree_within_cluster(units_count,units_loc,cluster_loc)
            if tree == None:
                continue
            n += 1 
            cluster_id = conflict_id + "_" + 'cluster_' + str(n) 
            prefixs = make_structure_constrain_and_seq(paths,units_loc,cluster_id,cluster_loc,seq_list,file_path)
            total_prefixs.extend(prefixs)
    return total_prefixs

def dot_reprocessing(left_dot,right_dot):
    processed_left_dot = list(left_dot[:])
    processed_right_dot = list(right_dot[:])

    longest = 0
    for i,symbol in enumerate(left_dot):
        if (symbol == '(') and (i > longest):
            longest = i
    processed_left_dot[longest] = '<'
    shortest = 99999
    for i,symbol in enumerate(right_dot):
        if (symbol == ')') and (i < shortest):
            shortest = i
    processed_right_dot[shortest] = '>'

    left_pair = []
    for i,symbol in enumerate(processed_left_dot):
        if symbol == '(':
            left_pair.append(i)
    right_pair = []
    for i,symbol in enumerate(processed_right_dot):
        if symbol == ')':
            right_pair.append(i)

    filtered_pairs = []
    for pair in zip(left_pair,right_pair[::-1]):
        left = pair[0]
        right = pair[1]
        if (left-1 not in left_pair) and (left+1 not in left_pair) and (right-1 not in right_pair) and (right+1 not in right_pair):
            filtered_pairs.append(pair)
    
    for pair in filtered_pairs:
        processed_left_dot[pair[0]] = '<'
        processed_right_dot[pair[1]] = '>'
    processed_left_dot = "".join(processed_left_dot)
    processed_right_dot = "".join(processed_right_dot)

    return processed_left_dot,processed_right_dot

def remake_constrain(fasta_file):
    seq_id = None
    left_seq = None
    right_seq = None
    left_dot = None
    right_dot = None
    outf = open(fasta_file,'r')
    for line in outf:
        if '_' in line:
            seq_id = line.strip()
        elif 'A' in line or 'T' in line or 'C' in line or 'G' in line or 'U' in line:
            line = line.strip().split("&")
            left_seq = line[0]
            right_seq = line[1]
        elif '.' in line or '(' in line or ')' in line:
            line = line.strip()
            sub = line.split("&")
            left_dot = sub[0]
            right_dot = sub[1]
    os.remove(fasta_file)

    processed_left_dot,processed_right_dot = dot_reprocessing(left_dot,right_dot)
    with open(fasta_file,'w')as f:
        f.write(seq_id+"\n")
        f.write(left_seq+'&'+right_seq+'\n')
        f.write(processed_left_dot+'&'+processed_right_dot)

def fold_structure_vienna(prefixs):
    for prefix in prefixs:
        print(prefix)
        seq = prefix + '.fasta'
        out = prefix + '.out'
        os.system("RNAcofold --constraint --enforceConstraint %s > %s"%(seq,out))
        while os.path.getsize(out) == 0:
            print('remake',prefix)
            remake_constrain(seq)
            os.system("RNAcofold --constraint --enforceConstraint %s > %s"%(seq,out))
        print(prefix,'-------end-----')

def vienna_to_bedpe_ct_add_annotation(prefixs,matrix_file,res=None):
    for prefix in prefixs:
        out_file = prefix + ".out"
        left_seq = None
        right_seq = None
        left_dot = None
        right_dot = None
        outf = open(out_file,'r')
        for line in outf:
            if '>' in line:
                continue
            elif 'A' in line or 'T' in line or 'C' in line or 'G' in line or 'U' in line:
                line = line.strip().split("&")
                left_seq = line[0][:-2]
                right_seq = line[1][2:]
            elif '.' in line or '(' in line or ')' in line:
                line = line.strip().split()
                sub = line[0].split("&")
                left_dot = sub[0][:-2]
                right_dot = sub[1][2:]
        left_len = len(left_seq)
        seq = left_seq + right_seq
        sub = prefix.split("_")
        left_start, left_end, right_start, right_end = int(sub[-4]), int(sub[-3]), int(sub[-2]), int(sub[-1])

        left_pairs_locs = []
        right_pairs_locs = []
        for i,s in enumerate(left_dot):
            if s == '.':
                continue
            elif s == '(':
                left_pairs_locs.append(i)
        for i,s in enumerate(right_dot):
            if s == '.':
                continue
            elif s == ')':
                right_pairs_locs.append(i)
        
        left_pairs = {l:r for l,r in zip(left_pairs_locs,right_pairs_locs[::-1])}
        right_pairs = {r:l for l,r in zip(left_pairs_locs,right_pairs_locs[::-1])}
        ct_out = prefix + ".ct"
        cto = open(ct_out,'w')
        cto.write(str(len(seq))+"\n")
        for i in range(len(seq)):
            if i <= (left_len - 1):
                left = i
                right = left_pairs.get(left,-left_len-1)
                if i < (left_len - 1):
                    cto.write(str(i+1)+" "+left_seq[i]+" "+str(i)+" "+str(i+2)+" "+str(right+left_len+1)+" "+str(i+1)+"\n")
                elif i == (left_len - 1):
                    cto.write(str(i+1)+" "+left_seq[i]+" "+str(i)+" "+str(0)+" "+str(right+left_len+1)+" "+str(i+1)+"\n")
            elif i >= left_len:
                right = i - left_len
                left = right_pairs.get(right,-1)
                if i > left_len:
                    cto.write(str(i+1)+" "+right_seq[right]+" "+str(i)+" "+str(i+2)+" "+str(left+1)+" "+str(i+1)+"\n")
                elif i == left_len:
                    cto.write(str(i+1)+" "+right_seq[right]+" "+str(0)+" "+str(i+2)+" "+str(left+1)+" "+str(i+1)+"\n")
        cto.close()

        bedpe_out = prefix + ".bedpe"
        bo = open(bedpe_out,'w')
        bo.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
        for left_part in left_pairs:
            left = left_part + left_start
            right = left_pairs[left_part] + right_start
            bo.write('HS1'+"\t"+str(left)+"\t"+str(left+1)+"\t"+'HS1'+"\t"+str(right)+"\t"+str(right+1)+"\n")
        bo.close

        annotation_out = prefix + "_annotation.txt"
        ao = open(annotation_out,'w')
        matrix_file_path = '../'+matrix_file
        matrix = extract_matrix(matrix_file_path,res)
        values = []
        for i in range(len(seq)):
            if i <= (left_len - 1):
                left = i
                if left in left_pairs:
                    left_zero = left + left_start
                    right_zero = left_pairs.get(left) + right_start
                    values.append(matrix[int(left_zero/10),int(right_zero/10)])
                else:
                    values.append(0)
            elif i >= left_len:
                right = i - left_len
                if right in right_pairs:
                    left_zero = right_pairs.get(right) + left_start
                    right_zero = right + right_start
                    values.append(matrix[int(left_zero/10),int(right_zero/10)])
                else:
                    values.append(0)
        for i in range(len(values)):
            ao.write(str(i+1)+"\t"+str(values[i])+"\n")
        ao.close()

def vertical_horizontal_conflict_make_constrain_seq(vertical_horizontal_conflict,seq_list,structure_feature_matrix,matrix_10,matrix_5,matrix_1,partition_matrix):
    file_path = 'vertical_horizontal_conflict_clusters'
    if not os.path.exists(path='./vertical_horizontal_conflict_clusters'):
        os.mkdir(path='./vertical_horizontal_conflict_clusters')
    total_prefixs = []

    conflict_set = set()
    for cluster in vertical_horizontal_conflict:
        conflict_clusters = vertical_horizontal_conflict[cluster]
        for conflict_cluster in conflict_clusters:
            merge1 = (cluster,conflict_cluster)
            merge2 = (conflict_cluster,cluster)
            if (merge1 in conflict_set) or (merge2 in conflict_set):
                continue
            else:
                conflict_set.add(merge1)
    n = 0
    conflict = {}
    clusters_loc = {}
    for pair in conflict_set:
        n += 1
        conflict_id = 'conflict_' + str(n)
        conflict[conflict_id] = [pair[0],pair[1]]
        sub0 = pair[0].split("_")
        sub1 = pair[1].split("_")
        clusters_loc[pair[0]] = [str(sub0[0]),str(sub0[1]),str(sub0[2]),str(sub0[3])]
        clusters_loc[pair[1]] = [str(sub1[0]),str(sub1[1]),str(sub1[2]),str(sub1[3])]

    for conflict_id in conflict:
        conflict_clusters = conflict[conflict_id]
        n = 0
        for cluster in conflict_clusters:
            cluster_loc = clusters_loc[cluster]
            units_loc, units_name = number_structural_unit_within_cluster(structure_feature_matrix,cluster_loc)
            units_count = calculate_unit_read_count(units_loc,matrix_10,matrix_5,matrix_1,partition_matrix)
            if units_count == None:
                print(cluster,'no substructure in cluster')
                continue
            tree,paths = build_structure_tree_within_cluster(units_count,units_loc,cluster_loc)
            if tree == None:
                continue
            n += 1 
            cluster_id = conflict_id + "_" + 'cluster_' + str(n) 
            prefixs = make_structure_constrain_and_seq(paths,units_loc,cluster_id,cluster_loc,seq_list,file_path)
            total_prefixs.extend(prefixs)
    return total_prefixs

def filter_conflict_and_select_diagonal(conflict_clusters,structure_feature,canny_cluster):
    sfm = Matrix(load_canny_bedpe(structure_feature))
    sfm.expend_matrix(10,1)
    canny_diagonal = []
    with open(canny_cluster,"r")as f:
        for line in f:
            if 'chrom1' in line:
                continue
            else:
                sub = line.strip().split()
                start1, end1, start2, end2 = int(sub[1]),int(sub[2]),int(sub[4]),int(sub[5])
                start1 = int(start1) - 10
                end1 = int(end1) + 10
                start2 = int(start2) - 10
                end2 = int(end2) + 10
                one = set([i for i in range(start1,end1+10)])
                two = set([i for i in range(start2,end2+10)])
                tmp = (int(sub[1]),int(sub[2]),int(sub[4]),int(sub[5]))
                if one & two:
                    canny_diagonal.append(tmp)

    n = 0
    filter_clusters = {}
    for cluster in conflict_clusters:
        n += 1
        conflict_id = 'conflict' + str(n)
        sub = cluster.split("_")
        tmp_matrix = sfm.matrix[int(sub[0]):int(sub[1])+10,int(sub[2]):int(sub[3])+10]
        matrix_sum = tmp_matrix.sum()
        if matrix_sum != 0:
            filter_clusters[conflict_id] = [int(sub[0]),int(sub[1]),int(sub[2]),int(sub[3])]
    return filter_clusters,canny_diagonal

def select_diagonal_conflict_clusters(out_conflict,canny_diagonal_list):
    region_start = min(int(out_conflict[0]),int(out_conflict[2]))
    region_end = max(int(out_conflict[1]),int(out_conflict[3]))
    region_set = set([i for i in range(region_start,region_end+10)])
    conflict_clusters = []
    for canny_diagonal in canny_diagonal_list:
        left_start, left_end, right_start, right_end = canny_diagonal.split("_")
        tmp_start = min(int(left_start),int(right_start))
        tmp_end = max(int(left_end),int(right_end))
        tmp_set = set([i for i in range(tmp_start,tmp_end+10)])
        if region_set & tmp_set:
            tmp = (int(left_start),int(left_end),int(right_start),int(right_end))
            conflict_clusters.append(tmp)
    return conflict_clusters

def select_candidate_structure_from_cluster(candidate_structure_matrix,diagonal_conflict_clusters_loc,out_clusters):
    '''
    candidate_structure_matrix      canny_merge_filter_final_matrix (29903,29903)    where pixel > 0 is candidate basepairs
    cluster_pixels           pixels list     [pixel1,pixel2,...,]
    pixels_loc            {pixel1:[HS1 20 30 HS1 40 50]}
    tmp_matrix          
    '''
    region_start = min([int(out_clusters[0]),int(out_clusters[1]),int(out_clusters[2]),int(out_clusters[3])])
    region_end = max([int(out_clusters[0]),int(out_clusters[1]),int(out_clusters[2]),int(out_clusters[3])])
    tmp_matrix = np.zeros((29903,29903))
    for cluster in diagonal_conflict_clusters_loc:
        tmp_matrix[int(cluster[0]):int(cluster[1])+1,int(cluster[2]):int(cluster[3])+1] = candidate_structure_matrix[int(cluster[0]):int(cluster[1])+1,int(cluster[2]):int(cluster[3])+1]
        tmp_start = min(int(cluster[0]),int(cluster[1]),int(cluster[2]),int(cluster[3]))
        tmp_end = max(int(cluster[0]),int(cluster[1]),int(cluster[2]),int(cluster[3]))
        if tmp_start < region_start:
            region_start = tmp_start
        if tmp_end > region_end:
            region_end = tmp_end
    return tmp_matrix,region_start,region_end

def number_structural_unit_within_triangle(candidate_structure_matrix,triangle_start,triangle_end):
    '''
    candidate_structure_matrix          matrix(29903,29903)      where pixel == 1 is candidate
    cluster_loc          [row_start,row_end,col_start,col_end]
    unit_loc      [pixelrow_pixelcol,......]           pixel_set for unit
    unit_name             [s1,........]               unit_num
    '''
    unit_loc = {}
    unit_name = []
    pixel_unit = {}
    checked_pixels = set()
    tmp_structure_num = 0
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    for row in range(int(triangle_start),int(triangle_end)):
        for col in range(int(triangle_start),int(triangle_end)):
            if (candidate_structure_matrix[row,col] > 0) and (str(row)+"_"+str(col) not in checked_pixels):
                one += 1
                if (row + 1 >= int(triangle_end)) or (col - 1 < int(triangle_start)):
                    five += 1
                    continue
                else:
                    if (candidate_structure_matrix[row+1,col-1] > 0):
                        two += 1
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
                three += 1
                if (row + 1 >= int(triangle_end)) or (col - 1 < int(triangle_start)):
                    continue
                else:
                    if (candidate_structure_matrix[row+1,col-1] > 0):
                        four += 1
                        structure_id = pixel_unit[str(row)+"_"+str(col)]
                        unit_loc[structure_id].append(str(row+1)+"_"+str(col-1))
                        pixel_unit[str(row+1)+"_"+str(col-1)] = structure_id
                        checked_pixels.add(str(row+1)+"_"+str(col-1))
            else:
                continue
    #print(one,two,three,four,five)
    return unit_loc,unit_name

def diagonal_inside_and_out_conflict_make_constrain_seq(diagonal_inside_and_out_conflict,seq_list,structure_feature_matrix,matrix_10,matrix_5,matrix_1,partition_matrix,diagonal_clusters):
    file_path = 'diagonal_inside_and_out_conflict_clusters'
    if not os.path.exists(path='./diagonal_inside_and_out_conflict_clusters'):
        os.mkdir(path='./diagonal_inside_and_out_conflict_clusters')
    non_diagonal_prefixs = []
    diagonal_constrain_prefixs = []

    for conflict_id in diagonal_inside_and_out_conflict:
        cluster_loc = diagonal_inside_and_out_conflict[conflict_id]
        n = 0
        units_loc, units_name = number_structural_unit_within_cluster(structure_feature_matrix,cluster_loc)
        units_count = calculate_unit_read_count(units_loc,matrix_10,matrix_5,matrix_1,partition_matrix)
        if units_count == None:
            print("_".join(list(map(str,cluster_loc))),'no substructure in cluster')
            continue
        tree,paths = build_structure_tree_within_cluster(units_count,units_loc,cluster_loc)
        if tree == None:
            continue
        n += 1 
        cluster_id1 = conflict_id + "_" +'cluster_' + str(n) + '_nondiagonal'
        prefixs1 = make_structure_constrain_and_seq(paths,units_loc,cluster_id1,cluster_loc,seq_list,file_path)
        non_diagonal_prefixs.extend(prefixs1)

        diagonal_conflict_clusters = select_diagonal_conflict_clusters(cluster_loc,diagonal_clusters)
        select_clusters_matrix, region_start, region_end = select_candidate_structure_from_cluster(structure_feature_matrix,diagonal_conflict_clusters,cluster_loc)
        diagonal_conflict_region = [region_start,region_end,region_start,region_end]
        diagonal_conflict_units_loc, diagonal_conflict_units_name = number_structural_unit_within_triangle(select_clusters_matrix,region_start,region_end)
        diagonal_conflict_units_count = calculate_unit_read_count(diagonal_conflict_units_loc,matrix_10,matrix_5,matrix_1,partition_matrix)
        if diagonal_conflict_units_count == None:
            print("_".join(list(map(str,cluster_loc))),'no substructure in cluster s diagonal clusters')
            n += 1 
            cluster_id2 = conflict_id + "_" +'cluster_' + str(n) + '_diagonal'
            prefixs2 = make_structure_and_seq_diagonal_in_out(cluster_id2,diagonal_conflict_region,seq_list,file_path,cluster_loc)
            diagonal_constrain_prefixs.extend(prefixs2)
            continue
        diagonal_conflict_tree,diagonal_conflict_paths = build_structure_tree_within_cluster(diagonal_conflict_units_count,diagonal_conflict_units_loc,diagonal_conflict_region)
        if diagonal_conflict_tree == None:
            continue
        n += 1 
        cluster_id2 = conflict_id + "_" +'cluster_' + str(n) + '_diagonal'
        prefixs2 = make_structure_bedpe_constrain_and_seq(diagonal_conflict_paths,diagonal_conflict_units_loc,cluster_id2,diagonal_conflict_region,seq_list,file_path,prohibited_area=cluster_loc)
        diagonal_constrain_prefixs.extend(prefixs2)
    return non_diagonal_prefixs,diagonal_constrain_prefixs

#################################################################################################################################
pyname , partition_out_file_id_file, clusters_bedpe, canny_bedpe, rri_matrix_10, rri_matrix_5, rri_matrix_1  = sys.argv

test_seq = load_fasta('NC_045512_2_md.fasta',['NC_045512.2'])['NC_045512.2']

all_partition_matrix = create_all_partition_matrixs(partition_out_file_id_file)
all_partition_matrix_diagonal_processing = diagonal_processing(all_partition_matrix)

raw_canny = load_canny_bedpe(canny_bedpe)
raw_oe_matrix_10 = extract_oe_matrix(rri_matrix_10,10)
canny_oe_mask = mask_diagonal(raw_canny,raw_oe_matrix_10)

canny = Matrix(canny_oe_mask)
canny.expend_matrix(10,1)

structure_feature_matrix = canny_partition_intersection(canny.matrix,all_partition_matrix_diagonal_processing)

#oe_matrix_10 = Matrix(extract_oe_matrix(rri_matrix_10,10))
#oe_matrix_10.expend_matrix(10,1)
matrix_10 = Matrix(extract_matrix(rri_matrix_10,10))
matrix_10.expend_matrix(10,1)

matrix_5 = Matrix(extract_matrix(rri_matrix_5,5))
matrix_5.expend_matrix(5,1)

matrix_1 = Matrix(extract_matrix(rri_matrix_1,1))

raw_matrix_10 = extract_matrix(rri_matrix_10,10)
shannon = get_shannon(raw_matrix_10,300,10)[1]

clusters_locs = load_clusters(clusters_bedpe)

high_shannon_clusters = select_cluster_with_high_shannon(clusters_locs,shannon,'clusters_with_high_shannon.bedpe')

non_diagonal_clusters,diagonal_clusters = select_filter_diagonal_non_diagonal_clusters(clusters_locs,matrix_10.matrix,'non_diagonal_clusters_clusters.bedpe','diagonal_clusters_clusters.bedpe')

diagonal_conflict_clusters,cross_conflict_clusters,vertical_horizontal_conflict_clusters,diagonal_inside_and_out_conflict_clusters = select_conflict_structure(high_shannon_clusters,non_diagonal_clusters,'conflict_clusters')

diagonal_conflict_clusters_filter = filter_conflict(diagonal_conflict_clusters,canny.matrix,'diagonal_conflict_clusters_filter.bedpe')
cross_conflict_clusters_filter = filter_conflict(cross_conflict_clusters,canny.matrix,'cross_conflict_clusters_filter.bedpe')
vertical_horizontal_conflict_clusters_filter = filter_conflict(vertical_horizontal_conflict_clusters,canny.matrix,'vertical_horizontal_conflict_clusters_filter.bedpe')
diagonal_inside_and_out_conflict_clusters_filter = diagonal_inside_and_out_filter_conflict(diagonal_inside_and_out_conflict_clusters,canny.matrix,'diagonal_inside_and_out_conflict_clusters_filter.bedpe')

diagonal_conflict_constrain_prefixs,diagonal_conflict_without_constrain_prefixs = diagonal_conflict_make_constrain_seq(diagonal_conflict_clusters_filter,test_seq,structure_feature_matrix,matrix_10.matrix,matrix_5.matrix,matrix_1.matrix,all_partition_matrix_diagonal_processing)
diagonal_conflict_total_prefixs = diagonal_conflict_constrain_prefixs + diagonal_conflict_without_constrain_prefixs
os.chdir('./diagonal_conflict_clusters')
fold_structure(diagonal_conflict_constrain_prefixs)
fold_structure_without_constrain(diagonal_conflict_without_constrain_prefixs)
ct_prefixs = ct2bedpe(diagonal_conflict_total_prefixs)
for prefix in ct_prefixs:
    locs = list(prefix.split(".")[0].split('_')[-5:-1])
    locs = [int(i) for i in locs]
    if set(list(range(locs[0],locs[1]))) & set(list(range(locs[2],locs[3]))):
        make_structure_ric_annotation_diagonal(prefix,rri_matrix_10,10)
    else:
        make_structure_ric_annotation(test_seq,prefix,rri_matrix_10,10)
os.chdir('..')

cross_conflict_total_prefixs = cross_conflict_make_constrain_seq(cross_conflict_clusters_filter,test_seq,structure_feature_matrix,matrix_10.matrix,matrix_5.matrix,matrix_1.matrix,all_partition_matrix_diagonal_processing)
os.chdir('./cross_conflict_clusters')
fold_structure_vienna(cross_conflict_total_prefixs)
vienna_to_bedpe_ct_add_annotation(cross_conflict_total_prefixs,rri_matrix_10,res=10)
os.chdir('..')

vertical_horizontal_conflict_total_prefixs = vertical_horizontal_conflict_make_constrain_seq(vertical_horizontal_conflict_clusters_filter,test_seq,structure_feature_matrix,matrix_10.matrix,matrix_5.matrix,matrix_1.matrix,all_partition_matrix_diagonal_processing)
os.chdir('./vertical_horizontal_conflict_clusters')
fold_structure_vienna(vertical_horizontal_conflict_total_prefixs)
vienna_to_bedpe_ct_add_annotation(vertical_horizontal_conflict_total_prefixs,rri_matrix_10,res=10)
os.chdir('..')

diagonal_inside_and_out_conflict_non_diagonal_prefixs,diagonal_inside_and_out_conflict_diagonal_constrain_prefixs = diagonal_inside_and_out_conflict_make_constrain_seq(diagonal_inside_and_out_conflict_clusters_filter,test_seq,structure_feature_matrix,matrix_10.matrix,matrix_5.matrix,matrix_1.matrix,all_partition_matrix_diagonal_processing,diagonal_clusters)
os.chdir('./diagonal_inside_and_out_conflict_clusters')
fold_structure_vienna(diagonal_inside_and_out_conflict_non_diagonal_prefixs)
vienna_to_bedpe_ct_add_annotation(diagonal_inside_and_out_conflict_non_diagonal_prefixs,rri_matrix_10,res=10)
fold_structure(diagonal_inside_and_out_conflict_diagonal_constrain_prefixs)
ct_prefixs = ct2bedpe(diagonal_inside_and_out_conflict_diagonal_constrain_prefixs)
for prefix in ct_prefixs:
    locs = list(prefix.split(".")[0].split('_')[-5:-1])
    locs = [int(i) for i in locs]
    if set(list(range(locs[0],locs[1]))) & set(list(range(locs[2],locs[3]))):
        make_structure_ric_annotation_diagonal(prefix,rri_matrix_10,10)
    else:
        make_structure_ric_annotation(test_seq,prefix,rri_matrix_10,10)
os.chdir('..')