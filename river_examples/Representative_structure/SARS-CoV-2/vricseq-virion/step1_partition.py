import os
import sys

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
    
def create_fasta_file(sequence, filename, location=None, file_path = './tmp_fasta'):
    if not os.path.exists(file_path):
        os.mkdir(path=file_path)
    path = file_path + "/" + filename
    seqid = filename[:filename.find('.fasta')]
    if not location:
        out = open(path, 'w')
        out.write(">"+seqid+"\n"+sequence+"\n")
        out.close()
    else:
        sequence = sequence[location[0] : location[1] + 1]
        out = open(path, 'w')
        out.write(">"+seqid+"\n"+sequence+"\n")
        out.close()

def partition_probabilityplot(fasta1, outprefix, outtype='text', file_path = './tmp_pfs_'):
    '''
    fasta1/2        seqence
    outprefix     output_name
    outtype     text/svg
    '''
    ProbabilityPlot_cmd = 'ProbabilityPlot'
    if outtype == 'text':
        file_path = file_path + 'txt'
        ProbabilityPlot_cmd += ' -t'
    if outtype == 'svg':
        file_path = file_path + 'svg'
        ProbabilityPlot_cmd += ' --svg'
    if not os.path.exists(file_path):
        os.mkdir(path=file_path)

    fasta1 = './tmp_fasta/' + fasta1
    partition_out = file_path + '/' + outprefix + '.pfs'
    
    partition_cmd = 'partition-smp' + " %s %s" % (fasta1, partition_out)
    os.system(partition_cmd)
    
    if outtype == 'text':
        ProbabilityPlot_out = file_path + '/' + outprefix + '.txt'
    if outtype == 'svg':
        ProbabilityPlot_out = file_path + '/' + outprefix + '.svg'
    ProbabilityPlot_cmd += " %s %s" % (partition_out, ProbabilityPlot_out)
    os.system(ProbabilityPlot_cmd)
    
def split_sequence_to_collect_basepairs(sequence,window=5000, overlap=3000, out_type='text', path_prefix='./tmp_pfs_'):
    seq = sequence
    start = []
    end = []
    genome_len = len(seq)
    i = 0
    while (1):
        if i + window > genome_len:
            start.append(genome_len - window)
            end.append(genome_len - 1)
            break
        else:
            if i < overlap:
                start.append(i)
                end.append(i + window - 1)
            else:
                start.append(i)
                end.append(i + window -1)
        i += window - overlap
    fragments = list(zip(start,end))
    files = []
    for fragment in fragments:
        fasta_name = str(fragment[0]) + "_" + str(fragment[1]) + '.fasta'
        files.append(fasta_name)
        loc = (int(fragment[0]),int(fragment[1]))
        create_fasta_file(sequence=seq, filename=fasta_name, location=loc)
    pairs = list(zip(files,files))
    outprefixs = []
    for pair in pairs:
        fasta1 = pair[0]
        fasta2 = pair[1]
        outprefix = fasta1.replace('.fasta', '') + "_" +fasta2.replace('.fasta', '')
        outprefixs.append(outprefix)
        partition_probabilityplot(fasta1, outprefix, outtype=out_type, file_path=path_prefix)
    if out_type == 'text':
        path = path_prefix + 'txt'
    if out_type == 'svg':
        path = path_prefix + 'svg'
    return outprefixs, path

def save_partition_file(out_prefixs):
    with open('partition_file_id.txt','w')as f:
        for prefix in out_prefixs:
            file_name = prefix + '.txt'
            f.write(file_name+"\n")

#step1 collect prosible basepair loc
pyname, fasta_file, sequence_id, window, overlap = sys.argv

if not os.path.exists(path='./step1_partition'):
        os.mkdir(path='./step1_partition')
os.chdir('./step1_partition')

fasta_file = './step0_prepare/' + fasta_file

test_seq = load_fasta(fasta_file,[sequence_id])[sequence_id]
test_seq = ''.join(test_seq)
window = int(window)
overlap = int(overlap)
out_prefixs, partition_out_path = split_sequence_to_collect_basepairs(test_seq, window, overlap)
save_partition_file(out_prefixs)
