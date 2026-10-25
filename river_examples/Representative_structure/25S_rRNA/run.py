import os
from datetime import datetime

def print_time():
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_now)

print("all_start!!!!!!!!!!!!")
print_time()

os.system("python step0_make_step1_step2_command.py")

command1 = None
command2 = None
with open("step1_step2_command.txt",'r')as f:
    lines = f.readlines()
    for i,l in enumerate(lines):
        if i == 0:
            command1 = l.strip()
        elif i == 1:
            command2 = l.strip()

os.system("%s"%(command1))
os.system("%s"%(command2))

command3 = None
with open("step3_command.txt",'r')as f:
    for line in f:
        command3 = line.strip()

os.system("%s"%(command3))

print("all_end!!!!!!!!!")
print_time()
