import numpy as np
import sys
from scipy import sparse
from datetime import datetime
################################################################################

class Matrix:
    def __init__(self, matrix, ostream = sys.stdout):
        self.matrix = matrix
        self.shape = matrix.shape
        self.ostream = ostream
    
    def matrix_to_bedpe(self, chrom_name='NR_002312.1', output=None):
        ones_bool = (self.matrix == 1)
        indices = np.where(ones_bool)
        files = open(output,'w')
        for row, col in zip(indices[0], indices[1]):
            strs = chrom_name+"\t"+str(row)+"\t"+str(row+1)+"\t"+chrom_name+"\t"+str(col)+"\t"+str(col+1)+"\n"
            files.write(strs)
        files.close()
    
    def matrix_to_hic(self, output=None):
        ones_bool = (self.matrix > 0)
        indices = np.where(ones_bool)
        files = open(output,'w')
        for row, col in zip(indices[0], indices[1]):
            files.write("0"+"\t"+'NR_002312.1'+"\t"+str(row)+"\t"+"0"+"\t"+"0"+"\t"+'NR_002312.1'+"\t"+str(col)+"\t"+"1"+"\t"+str(int(float(self.matrix[row,col])))+"\n")
        files.close()

    def save_matrix(self, name, sep_type=' ', fmt_type='%d'):
        np.savetxt(name, self.matrix, delimiter=sep_type, fmt=fmt_type)
    
    def expend_matrix(self, old_res, new_res):
        tmp_matrix = np.zeros((341,341))
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                time = int(old_res/new_res)
                l = row * time
                r = col * time
                l_end = l + time if l + time <= 340 else 341
                r_end = r + time if r + time <= 340 else 341
                tmp_matrix[l:l_end,r:r_end] = self.matrix[row,col]
        self.matrix = tmp_matrix

    def matrix_to_dump_format(self,time=1,output=None):
        ones_bool = (self.matrix > 0)
        indices = np.where(ones_bool)
        files = open(output,'w')
        for row, col in zip(indices[0], indices[1]):
            files.write(str(row)+"\t"+str(col)+"\t"+str(float(self.matrix[row,col])/(time**2))+"\n")
        files.close()

def load_canny_bedpe(infile):
    tmp_matrix = np.zeros((70,70))
    with open(infile,"r")as f:
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

def conflict_judgment_filter(tmp_next,units_loc,pre_path):
    not_conflict_structure = []
    region_start = 0
    region_end = 340
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

def build_structure_tree(structure_count, units_loc):
    root = TreePoint(num='point0', father=None, son=[], structure='s0', path=['s0'])
    sorted_structure_count = sorted(structure_count, key=lambda x: x[1], reverse=True)
    if not sorted_structure_count:
        return None, None

    point_num = 0
    next_round_structure = conflict_judgment_filter(
        [structure for structure, _ in sorted_structure_count if _ == sorted_structure_count[0][1]],
        units_loc, root.path
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
                next_round_structure = conflict_judgment_filter(tmp_next_structure, units_loc, tmp_path)
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

def extract_matrix(file_name,res):
    data = np.loadtxt(file_name, usecols=[0,1,2], dtype=[('S','<i4'),('E','<i4'),('value','<f8')])
    data['S'] = data['S'] / res
    data['E'] = data['E'] / res
    data['value'][data['value'] < 0] = 0
    data['value'][np.isnan(data['value'])] = 0
    matrix = get_matrix_npz(data,data)
    return matrix

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

def create_all_partition_matrixs(fileprefixs_file):
    fileprefixs = []
    with open(fileprefixs_file,"r")as f:
        for line in f:
            fileprefixs.append(line.strip())
    tmp_matrix = np.zeros((341,341))
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

    for time in range(5):
        tmp[x,y] = 0
        tmp[y,x] = 0
        x = np.delete(x,-1)
        y = np.delete(y,0)
    return tmp

def canny_partition_intersection(canny_matrix,partition_matrix,outfile):
    tmp_matrix = np.zeros((341,341))
    out = open(outfile,"w")
    for row in range(canny_matrix.shape[0]):
        for col in range(canny_matrix.shape[1]):
            if (canny_matrix[row,col] == 1) and (partition_matrix[row,col] > 0):
                tmp_matrix[row,col]  = partition_matrix[row,col]
                out.write("NR_002312.1"+"\t"+str(row)+"\t"+str(row+1)+"\t"+"NR_002312.1"+"\t"+str(col)+"\t"+str(col+1)+"\n")
    out.close()
    return tmp_matrix

def cluster_pixel_bedpe(bedtools_result, out_file, chrom_name):
    '''
    将bedtools找到的相互覆盖的像素进行聚类统计，将相邻的像素聚类后，输出聚类后的范围
    bedtools_result                  NR_002312.1	180	195	NR_002312.1	195	210	pixel134	NR_002312.1	180	195	NR_002312.1	180	195	pixel133                  相邻的像素位置
    out_file                         NR_002312.1	30	90	NR_002312.1	25350	25425	cluster_2                                             聚类簇
    chrom_name                       NR_002312.1                                                                                      染色体名称，好像没关系
    '''
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

def select_candidate_structure_from_cluster(candidate_structure_matrix,cluster_pixels,pixels_loc):
    '''
    checked
    '''
    tmp_matrix = np.zeros((341,341))
    for pixel in cluster_pixels:
        pixel_loc = pixels_loc[pixel]
        tmp_matrix[int(pixel_loc[1]):int(pixel_loc[2])+1,int(pixel_loc[4]):int(pixel_loc[5])+1] = candidate_structure_matrix[int(pixel_loc[1]):int(pixel_loc[2])+1,int(pixel_loc[4]):int(pixel_loc[5])+1]
    return tmp_matrix

def number_structural_unit(candidate_structure_matrix):
    '''
    candidate_structure_matrix          matrix(341,341)      where pixel > 0 is candidate
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
                f.write('NR_002312.1'+"\t"+str(left)+"\t"+str(left+1)+"\t"+'NR_002312.1'+"\t"+str(right)+"\t"+str(right+1)+"\n")
    return unit_loc,unit_name

def calculate_unit_read_count(units_loc,rri_matrix_10,rri_matrix_5,rri_matrix_1,partition_matrix,average=None):
    """
    units_loc      {s1:[start_end,start_end,...],....}            unit pixel loc
    rri_matrix               matrix(341,341)
    units_support       [(s1,count),...]                     unit reads counts
    """
    target_size = (341, 341)
    rri_matrix_10 = pad_or_crop_matrix(rri_matrix_10, target_size)
    rri_matrix_5 = pad_or_crop_matrix(rri_matrix_5, target_size)
    rri_matrix_1 = pad_or_crop_matrix(rri_matrix_1, target_size)
    merge_matrix = rri_matrix_10 + rri_matrix_5 + rri_matrix_1
    merge_99 = np.percentile(merge_matrix,99)
    merge_matrix[merge_matrix >= merge_99] = merge_99
    #merge_98 = np.percentile(merge_matrix[merge_matrix>0],98)
    #mask = (partition_matrix <= -np.log10(0.75)) & (merge_matrix <= merge_98)
    #merge_matrix[mask] = 0

    with open('merge.matrix','w')as f:
        ones_bool = (merge_matrix > 0)
        indices = np.where(ones_bool)
        for row, col in zip(indices[0], indices[1]):
            if row > col:
                continue
            f.write("0"+"\t"+"NR_002312.1"+"\t"+str(int(row))+"\t"+"0"+"\t"+"0"+"\t"+"NR_002312.1"+"\t"+str(int(col))+"\t"+"1"+"\t"+str(int(float(merge_matrix[row,col])))+"\n")
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

    #partition_matrix = partition_matrix[:341, :341]
    #non_zero_indices = partition_matrix != 0
    #final_matrix = np.zeros_like(rri_matrix_5)  # 创建一个与 rri_matrix_5 相同大小的零矩阵
    ##final_matrix[non_zero_indices] = (rri_matrix_5[non_zero_indices] + rri_matrix_1[non_zero_indices]) / partition_matrix[non_zero_indices]
    #final_matrix[non_zero_indices] = (merge_matrix[non_zero_indices]) / partition_matrix[non_zero_indices]
    ##final_matrix = (rri_matrix_5 + rri_matrix_1)/partition_matrix
    #ones_bool = (final_matrix > 0)
    #indices = np.where(ones_bool)
    #files = open('base_pairs_99mask.matrix','w')
    #for row, col in zip(indices[0], indices[1]):
    #    if row > col:
    #        continue
    #    files.write("0"+"\t"+"NR_002312.1"+"\t"+str(int(row))+"\t"+"0"+"\t"+"0"+"\t"+"NR_002312.1"+"\t"+str(int(col))+"\t"+"1"+"\t"+str(int(float(final_matrix[row,col])))+"\n")
    #files.close()

    return units_support

def create_structure_bedpe(structure_list,structure_loc,out_file):
    out = open(out_file,"w")
    for structure in structure_list:
        if structure == 's0':
            continue
        locs = structure_loc[structure]
        for loc in locs:
            sub = loc.split("_")
            out.write("NR_002312.1"+"\t"+str(int(sub[0]))+"\t"+str(int(sub[0])+1)+"\t"+"NR_002312.1"+"\t"+str(int(sub[1]))+"\t"+str(int(sub[1])+1)+"\n")
    out.close()

def structure_to_rnastructure_constraint(structure_list,structure_loc,out_file):
    out = open(out_file,"w")
    out.write('DS:'+'\n')
    out.write('-1'+'\n')
    out.write('SS:'+'\n')
    out.write('-1'+'\n')
    out.write('Mod:'+'\n')
    out.write('-1'+'\n')
    out.write('Pairs:'+'\n')
    #out.write('DS:'+'\n'+'-1'+'\n'+'SS:'+'\n'+'-1'+'\n'+'Mod:'+'\n'+'-1'+'\n'+'Pairs:'+'\n')
    for structure in structure_list:
        if structure == 's0':
            continue
        locs = structure_loc[structure]
        for loc in locs:
            sub = loc.split("_")
            out.write(str(int(sub[0])+1)+" "+str(int(sub[1])+1)+"\n")
    out.write('-1 -1'+'\n')
    out.write('FMN:'+'\n')
    out.write('-1'+'\n')
    out.write('Forbids:'+'\n')
    out.write('-1 -1'+'\n')
    #out.write('-1 -1'+'\n'+'FMN:'+'\n'+'-1'+'\n'+'Forbids:'+'\n'+'-1 -1'+'\n')
    out.close

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

def pad_or_crop_matrix(matrix, target_size):
    """
    根据目标大小对矩阵进行裁剪或填充。
    
    参数:
    matrix (np.ndarray): 输入矩阵。
    target_size (tuple): 目标大小 (行数, 列数)。
    
    返回:
    np.ndarray: 处理后的矩阵。
    """
    current_rows, current_cols = matrix.shape
    target_rows, target_cols = target_size
    
    # 如果矩阵的行数大于目标行数，则裁剪
    if current_rows > target_rows:
        matrix = matrix[:target_rows, :]
    # 如果矩阵的行数小于目标行数，则填充
    elif current_rows < target_rows:
        rows_to_pad = target_rows - current_rows
        matrix = np.pad(matrix, ((0, rows_to_pad), (0, 0)), mode='constant', constant_values=0)
    
    # 如果矩阵的列数大于目标列数，则裁剪
    if current_cols > target_cols:
        matrix = matrix[:, :target_cols]
    # 如果矩阵的列数小于目标列数，则填充
    elif current_cols < target_cols:
        cols_to_pad = target_cols - current_cols
        matrix = np.pad(matrix, ((0, 0), (0, cols_to_pad)), mode='constant', constant_values=0)
    
    return matrix

#################################################################################
pyname, partition_out_file_id_file, canny_bedpe, rri_matrix_10, rri_matrix_5, rri_matrix_1= sys.argv
print("--------------------start------------------")
print_time()

all_partition_matrix = create_all_partition_matrixs(partition_out_file_id_file)
all_partition_matrix_diagonal_processing = diagonal_processing(all_partition_matrix)

raw_canny = load_canny_bedpe(canny_bedpe)
raw_oe_matrix_5 = extract_oe_matrix(rri_matrix_5,5)
canny_oe_mask = mask_diagonal(raw_canny,raw_oe_matrix_5)

canny = Matrix(canny_oe_mask)
canny.expend_matrix(5,1)

structure_feature_matrix = canny_partition_intersection(canny.matrix,all_partition_matrix_diagonal_processing,'structure_feature.bedpe')

print("--------------------number_structural_unit------------------")
print_time()
units_loc, units_name = number_structural_unit(structure_feature_matrix)
print("units_len",len(units_name))
print_time()
print("@@@@@@@@@@@@@@@@@@@@number_structural_unit_end@@@@@@@@@@@@@@@@@@")

matrix_10 = Matrix(extract_matrix(rri_matrix_10,10))
matrix_10.expend_matrix(10,1)

matrix_5 = Matrix(extract_matrix(rri_matrix_5,5))
#matrix_5_percentile_99 = np.percentile(matrix_5.matrix, 99)
#matrix_5.matrix[matrix_5.matrix >= matrix_5_percentile_99] = matrix_5_percentile_99
matrix_5.expend_matrix(5,1)

matrix_1 = Matrix(extract_matrix(rri_matrix_1,1))
#matrix_1_percentile_99 = np.percentile(matrix_1.matrix, 99)
#matrix_1.matrix[matrix_1.matrix >= matrix_1_percentile_99] = matrix_1_percentile_99
print("--------------------calculate_unit_read_count------------------")
print_time()
units_count = calculate_unit_read_count(units_loc,matrix_10.matrix,matrix_5.matrix,matrix_1.matrix,structure_feature_matrix)
print_time()
print("@@@@@@@@@@@@@@@@@@@@calculate_unit_read_count_end@@@@@@@@@@@@@@@@@@")

print("--------------------build_structure_tree------------------")
print_time()
tree,paths = build_structure_tree(units_count,units_loc)
print_time()
print("@@@@@@@@@@@@@@@@@@@@build_structure_tree_end@@@@@@@@@@@@@@@@@@")

n = 0
for path in paths:
    n += 1
    bdepe_file_title = 'structure' + str(n) + '.bedpe'
    constrain_file_title = 'structure' + str(n) + '_constrain.txt'
    create_structure_bedpe(path,units_loc,bdepe_file_title)
    structure_to_rnastructure_constraint(path,units_loc,constrain_file_title)
