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
    '''选择位于大于全局香农均值位置的clusters,且配对距离小于3000'''
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

def load_canny_bedpe(infile,length):
    tmp_len = int(length/10)+10
    tmp_matrix = np.zeros((tmp_len,tmp_len))
    with open(infile,"r")as f:
        for line in f:
            if 'chrom1' in line:
                continue
            sub = line.strip().split()
            tmp_matrix[int(int(sub[1])/10),int(int(sub[4])/10)] = 1
    return tmp_matrix

def select_diagonal_non_diagonal_clusters(clusters,non_diagonal_out,diagonal_out):
    '''筛选距离不超过3000的，score>50%clusters的非对角线clusters.以及对角clusters'''
    #candidate_clusters = []
    non_diagonal_clusters = []
    diagonal_clusters = []
    for cluster in clusters:
        if int(cluster[3]) - int(cluster[0]) >= 3000:
            continue
        #candidate_clusters.append((cluster,value))
        left = set(list(range(int(cluster[0]),int(cluster[1])+10)))
        right = set(list(range(int(cluster[2])-10,int(cluster[3]))))
        if left & right:
            diagonal_clusters.append(cluster)
        else:
            non_diagonal_clusters.append(cluster)
            
    of = open(non_diagonal_out,"w")
    of.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
    for cluster in non_diagonal_clusters:
        of.write('HS1'+'\t'+str(cluster[0])+'\t'+str(cluster[1])+'\t'+'HS1'+'\t'+str(cluster[2])+'\t'+str(cluster[3])+'\n')
    of.close()

    of = open(diagonal_out,"w")
    of.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
    for cluster in diagonal_clusters:
        of.write('HS1'+'\t'+str(cluster[0])+'\t'+str(cluster[1])+'\t'+'HS1'+'\t'+str(cluster[2])+'\t'+str(cluster[3])+'\n')
    of.close()
    return non_diagonal_clusters,diagonal_clusters

def load_clusters(cluster_file):
    clusters = []
    with open(cluster_file,'r')as f:
        for line in f:
            if 'chrom1' in line:
                continue
            else:
                sub = line.strip().split()
                clusters.append((int(sub[1]),int(sub[2]),int(sub[4]),int(sub[5])))
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

def calculate_unit_read_count(units_loc,rri_matrix_10,rri_matrix_5,rri_matrix_1,biopartition_matrix):
    """
    units_loc      {s1:[start_end,start_end,...],....}            unit pixel loc
    rri_matrix               matrix(28044,28044)
    units_support       [(s1,count),...]                     unit reads counts
    """
    merge_matrix = rri_matrix_10 + rri_matrix_5 + rri_matrix_1
    merge_99 = np.percentile(merge_matrix,99)
    merge_matrix[merge_matrix >= merge_99] = merge_99

    units_support = []
    for unit in units_loc.keys():
        pixels_num = 0
        count = 0
        locs = units_loc[unit]
        for loc in locs:
            pixels_num += 1
            row = int(loc.split("_")[0])
            col = int(loc.split("_")[1])
            count += merge_matrix[row,col] + (merge_matrix[row,col] / biopartition_matrix[row,col])
            #count += rri_matrix_5[row,col] + rri_matrix_1[row,col] + ((rri_matrix_5[row,col] + rri_matrix_1[row,col]) / partition_matrix[row,col])
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

def create_all_partition_matrixs(fileprefixs_file,length):
    fileprefixs = []
    with open(fileprefixs_file,"r")as f:
        for line in f:
            fileprefixs.append(line.strip())
    tmp_matrix = np.zeros((length,length))
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
    return tmp

def get_oe(matrix):
    #沿对角线逐对角线计算均值，让observe/均值。mask掉哪些交互很少的列。
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

def canny_partition_intersection(canny_matrix,partition_matrix,length):
    tmp_matrix = np.zeros((length,length))
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

        # 基于CT文件名生成基础BEDPE文件名
        base_bedpe_name = prefix
        res = None
        start = int(base_bedpe_name.split("_")[-4])

        left_start = int(base_bedpe_name.split("_")[-4]) + 1
        left_end = int(base_bedpe_name.split("_")[-3])
        right_start = int(base_bedpe_name.split("_")[-2]) + 1
        right_end = int(base_bedpe_name.split("_")[-1])

        # 初始化变量
        current_structure = None
        out = None

        for index, row in enumerate(ct):
            row = row.strip()
            if not row:  # 跳过空行
                continue
            if 'ENERGY' in row:  # 假设结构之间以包含"ENERGY"的行分隔
                if out is not None:
                    out.close()
                # 为每个新结构生成一个新的BEDPE文件
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
            #    continue  # 跳过CT文件的标题行
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

        # 关闭最后一个文件
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

def make_constrain_short(structure_list,structure_loc,left_start,left_len,right_start,right_len):
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

def make_cluster_sequence_constrain_short(seq,cluster_loc,out_file,structure_list,structure_loc,file_path):
    sub = cluster_loc.split("_")
    left_start, left_end, right_start, right_end = int(sub[0]),int(sub[1]),int(sub[2]),int(sub[3])
    left_seq = "".join(seq[left_start:left_end+2])
    left_len = len(left_seq)
    right_seq = "".join(seq[right_start-2:right_end])
    right_len = len(right_seq)
    sequence = left_seq + '&' + right_seq
    constrain = make_constrain_short(structure_list,structure_loc,left_start,left_len,right_start,right_len)
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
        make_cluster_sequence_constrain_short(seq_list,cluster_loc,seq_constrain_file_title,path,units_loc,file_path)
        out_prefixs.append(cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc)
        out.write(cluster_id + "_" + 'structure' + str(n) + "_" + cluster_loc+"\n")
    return out_prefixs

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

def vienna_to_bedpe_ct_add_annotation_long(prefixs,matrix_file,chrom,res=None):
    file_path = '../ct_bedpe_annotation'
    if not os.path.exists(file_path):
        os.mkdir(path=file_path)
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
        left_start, left_end, right_start, right_end = int(sub[0]), int(sub[1]), int(sub[2]), int(sub[3])

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
        ct_out = file_path + '/' + prefix + ".ct"
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

        bedpe_out = file_path + '/' + prefix + ".bedpe"
        bo = open(bedpe_out,'w')
        bo.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
        for left_part in left_pairs:
            left = left_part + left_start
            right = left_pairs[left_part] + right_start
            bo.write(chrom+"\t"+str(left)+"\t"+str(left+1)+"\t"+chrom+"\t"+str(right)+"\t"+str(right+1)+"\n")
        bo.close

        annotation_out = file_path + '/' + prefix + "_annotation.txt"
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

def short_range_cluster_make_constrain_seq(non_diagonal_clusters,diagonal_clusters,seq_list,structure_feature_matrix,matrix_10,matrix_5,matrix_1,partition_matrix):
    file_path = 'short_range_clusters'
    if not os.path.exists(path='./short_range_clusters'):
        os.mkdir(path='./short_range_clusters')
    non_diagonal_prefixs = []
    diagonal_constrain_prefixs = []

    n = 0

    for cluster in non_diagonal_clusters:
        units_loc, units_name = number_structural_unit_within_cluster(structure_feature_matrix,cluster)
        units_count = calculate_unit_read_count(units_loc,matrix_10,matrix_5,matrix_1,partition_matrix)
        if units_count == None:
            print("_".join(list(map(str,cluster))),'no substructure in cluster')
            continue
        tree,paths = build_structure_tree_within_cluster(units_count,units_loc,cluster)
        if tree == None:
            continue
        n += 1 
        cluster_id = 'cluster_' + str(n)
        prefixs = make_structure_constrain_and_seq(paths,units_loc,cluster_id,cluster,seq_list,file_path)
        non_diagonal_prefixs.extend(prefixs)

    for cluster in diagonal_clusters:
        units_loc, units_name = number_structural_unit_within_cluster(structure_feature_matrix,cluster)
        units_count = calculate_unit_read_count(units_loc,matrix_10,matrix_5,matrix_1,partition_matrix)
        if units_count == None:
            print("_".join(list(map(str,cluster))),'no substructure in cluster')
            continue
        tree,paths = build_structure_tree_within_cluster(units_count,units_loc,cluster)
        if tree == None:
            continue
        n += 1 
        cluster_id = 'cluster_' + str(n) 
        prefixs = make_structure_bedpe_constrain_and_seq(paths,units_loc,cluster_id,cluster,seq_list,file_path)
        diagonal_constrain_prefixs.extend(prefixs)
    return non_diagonal_prefixs,diagonal_constrain_prefixs

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

def pad_matrix(matrix, target_rows, target_cols):
    # 获取当前矩阵的形状
    current_rows, current_cols = matrix.shape

    # 计算需要填充的行数和列数
    rows_to_pad = max(0, target_rows - current_rows)
    cols_to_pad = max(0, target_cols - current_cols)

    # 使用 numpy.pad 进行填充
    # ((0, rows_to_pad))表示在行方向上前面填充0行，后面rows填充_to_pad行
    # ((0, cols_to_pad))表示在列方向上前面填充0列，后面填充cols_to_pad列
    # constant_values=0 表示用0填充
    padded_matrix = np.pad(matrix, ((0, rows_to_pad), (0, cols_to_pad)), mode='constant', constant_values=0)

    return padded_matrix

def select_long_range_clusters(clusters,chrom,distance=2000):
    long_range_clusters = []
    for cluster in clusters:
        tmp_distance = cluster[3] - cluster[0]
        if tmp_distance > distance:
            long_range_clusters.append(cluster)
    with open('clusters_longer_than_'+str(distance)+'.bedpe','w')as f:
        f.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
        for cluster in long_range_clusters:
            f.write(chrom+'\t'+str(cluster[0])+'\t'+str(cluster[1])+'\t'+chrom+'\t'+str(cluster[2])+'\t'+str(cluster[3])+'\n')
    return long_range_clusters

def create_fasta_file(cluster,seq_list,file_path = './tmp_fasta'):
    if not os.path.exists(file_path):
        os.mkdir(path=file_path)
    filename_1 = str(cluster[0])+"_"+str(cluster[1])+".fasta"
    filename_2 = str(cluster[2])+"_"+str(cluster[3])+".fasta"
    path_1 = file_path + "/" + filename_1
    path_2 = file_path + "/" + filename_2
    seqid_1 = filename_1[:filename_1.find('.fasta')]
    seqid_2 = filename_2[:filename_2.find('.fasta')]
    out = open(path_1, 'w')
    out.write(">"+seqid_1+"\n"+"".join(seq_list[cluster[0]:cluster[1]])+"\n")
    out.close()
    out = open(path_2, 'w')
    out.write(">"+seqid_2+"\n"+"".join(seq_list[cluster[2]:cluster[3]])+"\n")
    out.close()
    return filename_1,filename_2

def bipartition_probabilityplot(cluster, seq_list, outtype='text', file_path = './tmp_pfs_'):
    '''
    fasta1/2        seqence
    outprefix     output_name
    outtype     text/svg
    '''
    outprefix = "_".join(list(map(str,cluster)))
    
    fasta_file_name_1,fasta_file_name_2 = create_fasta_file(cluster,seq_list)

    ProbabilityPlot_cmd = 'ProbabilityPlot'
    if outtype == 'text':
        file_path = file_path + 'txt'
        ProbabilityPlot_cmd += ' -t'
    if outtype == 'svg':
        file_path = file_path + 'svg'
        ProbabilityPlot_cmd += ' --svg'
    if not os.path.exists(file_path):
        os.mkdir(path=file_path)

    fasta_1 = './tmp_fasta/' + fasta_file_name_1
    fasta_2 = './tmp_fasta/' + fasta_file_name_2
    partition_out = file_path + '/' + outprefix + '.pfs'
    
    partition_cmd = 'bipartition-smp' + " %s %s %s" % (fasta_1, fasta_2, partition_out)
    os.system(partition_cmd)
    
    if outtype == 'text':
        ProbabilityPlot_out = file_path + '/' + outprefix + '.txt'
    if outtype == 'svg':
        ProbabilityPlot_out = file_path + '/' + outprefix + '.svg'
    ProbabilityPlot_cmd += " %s %s" % (partition_out, ProbabilityPlot_out)
    os.system(ProbabilityPlot_cmd)

    return outprefix + '.txt'

def create_biopartition_matrixs(biopartition_file):
    locs = biopartition_file.replace('.txt','').split("_")
    left_start, left_end, right_start, right_end = map(int,list(locs))
    left_len = left_end - left_start
    right_len = right_end - right_start
    tmp_matrix = np.zeros((left_len,right_len))

    tmp = open("./tmp_pfs_txt/"+biopartition_file, 'r')
    for index, line in enumerate(tmp):
        if index < 2:
            continue
        else:
            tmp_sub = line.strip().split()
            left = int(tmp_sub[0]) - 1
            right = int(tmp_sub[1]) - 4 - left_len
            tmp_matrix[left, right] = float(tmp_sub[2])
    tmp.close()
    return tmp_matrix

def number_structural_unit(candidate_structure_matrix):
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
    #with open('units_check.bedpe',"w")as f:
    #    f.write('chrom1'+"\t"+'start1'+'\t'+'end1'+'\t'+'chrom2'+"\t"+'start2'+'\t'+'end2'+'\n')
    #    for unit in unit_loc:
    #        locs = unit_loc[unit]
    #        for loc in locs:
    #            sub = loc.split("_")
    #            left = int(sub[0])
    #            right = int(sub[1])
    #            f.write(chrom+"\t"+str(left)+"\t"+str(left+1)+"\t"+chrom+"\t"+str(right)+"\t"+str(right+1)+"\n")
    return unit_loc,unit_name

class TreePoint:
    def __init__(self, num=None, father=None, son=None, structure=None, path=None):
        self.num = num
        self.father = father
        self.son = son
        self.structure = structure
        self.path = path

def pseudoknot_judgment_long(pre_path, units_loc, structure, left_len):
    structure_locs = units_loc[structure]
    structure_left = [int(loc.split("_")[0]) for loc in structure_locs]
    structure_right = [int(loc.split("_")[1])+left_len for loc in structure_locs]
    structure_left_start, structure_left_end = min(structure_left), max(structure_left)
    structure_right_start, structure_right_end = min(structure_right), max(structure_right)

    for pre_structure in pre_path:
        if pre_structure == 's0':
            continue
        pre_structure_locs = units_loc[pre_structure]
        left = [int(loc.split("_")[0]) for loc in pre_structure_locs]
        right = [int(loc.split("_")[1])+left_len for loc in pre_structure_locs]
        left_start, left_end = min(left), max(left)
        right_start, right_end = min(right), max(right)

        if (left_end < structure_left_start < right_start < structure_right_start > right_end) or \
           (structure_right_start > left_end and structure_right_end < right_start and structure_left_end < left_start):
            return True
    return False

def conflict_judgment_filter_long(tmp_next,units_loc,pre_path,left_length,right_length):
    not_conflict_structure = []
    region_start = 0
    region_end = left_length + right_length - 1
    region_len = region_end - region_start + 1
    dot = np.zeros(region_len)
    for structure in pre_path:
        if structure == 's0':
            continue
        for loc in units_loc[structure]:
            sub = loc.split("_")
            dot[int(sub[0]) - region_start] = 1
            dot[int(sub[1]) - region_start + left_length] = 1

    not_conflict_structure = []
    for tmp_structure in tmp_next:
        if pseudoknot_judgment_long(pre_path, units_loc, tmp_structure, left_length):
            continue
        if all(dot[int(loc.split("_")[0]) - region_start] == 0 and dot[int(loc.split("_")[1]) - region_start+left_length] == 0 for loc in units_loc[tmp_structure]):
            not_conflict_structure.append(tmp_structure)
    return not_conflict_structure

def build_structure_tree(structure_count, units_loc, left_length, right_length):
    root = TreePoint(num='point0', father=None, son=[], structure='s0', path=['s0'])
    sorted_structure_count = sorted(structure_count, key=lambda x: x[1], reverse=True)
    if not sorted_structure_count:
        return None, None

    point_num = 0
    next_round_structure = conflict_judgment_filter_long(
        [structure for structure, _ in sorted_structure_count if _ == sorted_structure_count[0][1]],
        units_loc, root.path, left_length, right_length
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
                next_round_structure = conflict_judgment_filter_long(tmp_next_structure, units_loc, tmp_path, left_length, right_length)
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

def make_constrain(structure_list,structure_loc,left_len,right_len):
    left_const = ["<" for i in range(left_len)]
    right_const = [">" for i in range(right_len)]
    for structure in structure_list:
        if structure == 's0':
            continue
        locs = structure_loc[structure]
        for loc in locs:
            sub = loc.split("_")
            left_const[int(sub[0])] = '('
            right_const[int(sub[1])+2] = ')'
    const = "".join(left_const) + '&' + "".join(right_const)
    return const

def make_cluster_sequence_constrain(seq_list,cluster,out_file,structure_list,structure_loc,file_path):
    left_seq = "".join(seq_list[cluster[0]:cluster[1]+2])
    left_len = len(left_seq)
    right_seq = "".join(seq_list[cluster[2]-2:cluster[3]])
    right_len = len(right_seq)
    sequence = left_seq + '&' + right_seq
    constrain = make_constrain(structure_list,structure_loc,left_len,right_len)
    out_path = file_path+'/' + out_file
    of = open(out_path,'w')
    of.write(">"+"_".join([str(cluster[0]),str(cluster[1]),str(cluster[2]),str(cluster[3])])+"\n")
    of.write(sequence+"\n")
    of.write(constrain)

def make_structure_constrain_sequence(paths,units_loc,cluster,seq_list):
    '''
    paths     path list [[s1,s2,s3...],[],[],[]...]
    units_loc      {s1:[28830_28949,28831_28948],...}
    calculate_matrix    oe or kr matrix (29903,29903)
    cluster_id            cluster1...
    cluster_loc            [28830,28949,28831,28948]
    '''
    out = open("all_file_name_long.info",'a')
    file_path = './constrain_sequence'
    if not os.path.exists(file_path):
        os.mkdir(path=file_path)
    n = 0
    cluster_loc = list(map(str,cluster))
    cluster_loc = "_".join(cluster_loc)
    out_prefixs = []
    for path in paths:
        n += 1
        seq_constrain_file_title = cluster_loc + "_" +'structure' + str(n) + '.fasta'
        make_cluster_sequence_constrain(seq_list,cluster,seq_constrain_file_title,path,units_loc,file_path)
        out_prefixs.append(cluster_loc + "_" +'structure' + str(n))
        out.write(cluster_loc + "_" +'structure' + str(n)+"\n")
    return out_prefixs

########################################################################################################################################################################
pyname, partition_out_file_id_file, clusters_bedpe, canny_bedpe, rri_matrix_10, rri_matrix_5, rri_matrix_1, length, chrom, fasta_file, sequence_id, distance  = sys.argv

length = int(length)
distance = int(distance)

clusters = load_clusters(clusters_bedpe)#[1,2,3,4]

seq_list = load_fasta(fasta_file,[sequence_id])[sequence_id]

expanded_matrix_10 = expand_matrix(extract_matrix(rri_matrix_10,10),10,1,length)
expanded_matrix_5 = expand_matrix(extract_matrix(rri_matrix_5,5),5,1,length)
padded_matrix_1 = pad_matrix(extract_matrix(rri_matrix_1,1), length, length)[:length,:length]

#long_range------------------------------------------------------------------------------------------------------------------------------------------------------------
long_range_clusters = select_long_range_clusters(clusters,chrom,distance=distance)

for cluster in long_range_clusters:
    left_len = cluster[1] - cluster[0]
    right_len = cluster[3] - cluster[2]

    tmp_matrix_10 = expanded_matrix_10[cluster[0]:cluster[1],cluster[2]:cluster[3]]
    tmp_matrix_5 = expanded_matrix_5[cluster[0]:cluster[1],cluster[2]:cluster[3]]
    tmp_matrix_1 = padded_matrix_1[cluster[0]:cluster[1],cluster[2]:cluster[3]]

    biopartition_file_name = bipartition_probabilityplot(cluster,seq_list)

    biopartition_matrix = create_biopartition_matrixs(biopartition_file_name)

    units_loc, units_name = number_structural_unit(biopartition_matrix)

    units_count = calculate_unit_read_count(units_loc,tmp_matrix_10,tmp_matrix_5,tmp_matrix_1,biopartition_matrix)
    if units_count == None:
        print("_".join(list(map(str,cluster))),'no substructure in cluster')
        continue

    tree,paths = build_structure_tree(units_count,units_loc,left_len,right_len)
    if tree == None:
        continue

    prefixs = make_structure_constrain_sequence(paths,units_loc,cluster,seq_list)
    
    os.chdir('./constrain_sequence')
    fold_structure_vienna(prefixs)

    vienna_to_bedpe_ct_add_annotation_long(prefixs,rri_matrix_10,chrom,res=10)
    os.chdir('..')

#Short_range-------------------------------------------------------------------------------------------------------------------------------------------------------------
all_partition_matrix = create_all_partition_matrixs(partition_out_file_id_file,length)
all_partition_matrix_diagonal_processing = diagonal_processing(all_partition_matrix)

raw_canny = load_canny_bedpe(canny_bedpe,length)
raw_oe_matrix_10 = extract_oe_matrix(rri_matrix_10,10)
canny_oe_mask = mask_diagonal(raw_canny,raw_oe_matrix_10)
canny = expand_matrix(canny_oe_mask,10,1,length)

structure_feature_matrix = canny_partition_intersection(canny,all_partition_matrix_diagonal_processing,length)

non_diagonal_clusters,diagonal_clusters = select_diagonal_non_diagonal_clusters(clusters,'non_diagonal_clusters_clusters.bedpe','diagonal_clusters_clusters.bedpe')

non_diagonal_prefixs,diagonal_constrain_prefixs = short_range_cluster_make_constrain_seq(non_diagonal_clusters,diagonal_clusters,seq_list,structure_feature_matrix,expanded_matrix_10,expanded_matrix_5,padded_matrix_1,all_partition_matrix_diagonal_processing)
os.chdir('./short_range_clusters')
fold_structure_vienna(non_diagonal_prefixs)
vienna_to_bedpe_ct_add_annotation(non_diagonal_prefixs,rri_matrix_10,res=10)
fold_structure(diagonal_constrain_prefixs)
ct_prefixs = ct2bedpe(diagonal_constrain_prefixs)
for prefix in ct_prefixs:
    make_structure_ric_annotation_diagonal(prefix,rri_matrix_10,10)
os.chdir('..')