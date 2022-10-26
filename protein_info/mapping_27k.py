import numpy as np
import pandas as pd
# data = np.loadtxt('./9606tab/HUMAN_9606_idmapping.txt', delimiter=',',dtype=np.str)
# dat_file = './9606tab/HUMAN_9606_idmapping.dat'
# with open(dat_file, encoding='utf-8') as f:
# 	lines = f.readlines()

lines = np.load('mapping_process.npy')

dictionary = open('protein.SHS27k.sequences.dictionary.txt')
d = dictionary.read()
count_all = 0
all_list = []
char = '\t'
for i in range(1690):
	count = 0
	for j in range(lines.shape[0]):
		str_line = lines[j]
		index_str = d[16*i:16*i+15]
		result = index_str in str_line
		if result == True:
			count = count + 1
			count_all = count_all + 1
			ind1 = str_line.find(char)
			all_list.append(str_line[0:ind1])
			# print(str_line)
			# print(index_str)

		if count >= 1:
			break
	if count == 0:
		all_list.append('None')
		# print(index_str)
print(count_all)
# np.savetxt('final.txt', all_list,fmt='%.5f', delimiter=",")
file = open('uni_final_include_none.txt','w');
file.write(str(all_list));
file.close();

