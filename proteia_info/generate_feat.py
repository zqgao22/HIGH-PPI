import numpy as np
import csv
import os
from tqdm import tqdm
import sys
import math
import argparse
import re
import torch
from itertools import combinations


def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)


def read_atoms(file, chain=".", model=1):
    pattern = re.compile(chain)

    current_model = model
    atoms = []
    ajs = []
    for line in file:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ajs_id = line[17:20]
                atoms.append((x, y, z))
                ajs.append(ajs_id)
        # elif line.startswith("MODEL"):
        #     current_model = int(line[10:14].strip())
    return atoms, ajs


def compute_contacts(atoms, threshold):
    contacts = []
    for i in range(len(atoms)-2):
        for j in range(i+2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i+1, j+1))
    return contacts


def write_output(contacts, file):
    for c in contacts:
        file.write("\t".join(map(str, c))+"\n")


def pdb_to_x(file, threshold, chain=".", model=1):
    atoms, ajs = read_atoms(file, chain, model)
    return ajs

# x_set = pdb_to_x(open('P84085.pdb', "r"), 7.5)
# replace_dict = {'ALA':'89.09', '71.07':'', '':'',}
c = 0
count1 = -1
ensp = open('ensp_uniprot.txt')
e = ensp.read()
e_sp = e.split('ENSP')
list_all = []
x_set = np.zeros(())
all_for_assign = np.loadtxt("all_assign.txt")
for liness1 in tqdm(open('protein.SHS27k.sequences.dictionary.pro3.tsv')):
    count1 = count1 + 1
    line1 = liness1.split('\t')
    li = line1[0][10:]
    for i in range(1690):
        e_zj = e_sp[i]
        res = li in e_zj
        if res == True:
            li2 = e_zj[13:-9]
            pdb_file_name = li2 + '.pdb'
            print(pdb_file_name)
            c = c + 1
            xx = pdb_to_x(open(pdb_file_name, "r"), 7.5)
            break
    x_p = np.zeros((len(xx), 7))


    # for j in range(len(xx)):
    #             if xx[j] == 'ALA':
    #                 x_p[j] = np.array([6.0,0,0, 3, 3, -0.6719000000000002, 63.31999999999999])
    #             elif xx[j] == 'CYS':
    #                 x_p[j] = np.array([5.02,1,0, 3, 3, -0.6719000000000002, 63.31999999999999])
    #             elif xx[j] == 'ASP':
    #                 x_p[j] = np.array([2.77,2,1, 3, 3, -1.1269999999999993, 100.61999999999999])
    #             elif xx[j] == 'GLU':
    #                 x_p[j] = np.array([3.22,2,1, 3, 3, -0.7368999999999997, 100.61999999999999])
    #             elif xx[j] == 'PHE':
    #                 x_p[j] = np.array([5.48,0,0, 2, 2, 0.6409999999999995, 63.31999999999999])
    #             elif xx[j] == 'GLY':
    #                 x_p[j] = np.array([5.97,1,0, 2, 2, -0.9702999999999999, 63.31999999999999])
    #             elif xx[j] == 'HIS':
    #                 x_p[j] = np.array([7.47,3,2, 3, 3, -0.6358999999999997, 91.99999999999999])
    #             elif xx[j] == 'ILE':
    #                 x_p[j] = np.array([5.94,0,0, 2, 2, 0.44439999999999996, 63.32])
    #             elif xx[j] == 'LYS':
    #                 x_p[j] = np.array([9.59,3,2, 3, 3, -0.4726999999999995, 89.34])
    #             elif xx[j] == 'LEU':
    #                 x_p[j] = np.array([5.98,0,0, 2, 2, 0.44439999999999996, 63.32])
    #             elif xx[j] == 'MET':
    #                 x_p[j] = np.array([5.74,0,0, 3, 2, 0.1514000000000002, 63.31999999999999])
    #             elif xx[j] == 'ASN':
    #                 x_p[j] = np.array([5.41,1,0, 3, 3, -1.7262999999999993, 106.40999999999998])
    #             elif xx[j] == 'PRO':
    #                 x_p[j] = np.array([6.30,0,0, 2, 2, -0.17699999999999982, 49.33])
    #             elif xx[j] == 'GLN':
    #                 x_p[j] = np.array([5.65,1,0, 3, 3, -1.336199999999999, 106.40999999999998])
    #             elif xx[j] == 'ARG':
    #                 x_p[j] = np.array([11.15,3,2, 3, 5, -1.338429999999999, 125.21999999999998])
    #             elif xx[j] == 'SER':
    #                 x_p[j] = np.array([5.68,1,0, 3, 3, -1.6094, 83.55])
    #             elif xx[j] == 'THR':
    #                 x_p[j] = np.array([5.64,1,0, 3, 3, -1.2209, 83.55])
    #             elif xx[j] == 'VAL':
    #                 x_p[j] = np.array([5.96,0,0, 2, 2, 0.05430000000000007, 63.32])
    #             elif xx[j] == 'TRP':
    #                 x_p[j] = np.array([5.89,0,0, 2, 3, 1.1222999999999996, 79.10999999999999])
    #             elif xx[j] == 'TYR':
    #                 x_p[j] = np.array([5.66,1,0, 3, 3, 0.3466000000000002, 83.55])
    # list_all.append(x_p)
    for j in range(len(xx)):
        if xx[j] == 'ALA':
            x_p[j] = all_for_assign[0,:]
        elif xx[j] == 'CYS':
            x_p[j] = all_for_assign[1,:]
        elif xx[j] == 'ASP':
            x_p[j] = all_for_assign[2,:]
        elif xx[j] == 'GLU':
            x_p[j] = all_for_assign[3,:]
        elif xx[j] == 'PHE':
            x_p[j] = all_for_assign[4,:]
        elif xx[j] == 'GLY':
            x_p[j] = all_for_assign[5,:]
        elif xx[j] == 'HIS':
            x_p[j] = all_for_assign[6,:]
        elif xx[j] == 'ILE':
            x_p[j] = all_for_assign[7,:]
        elif xx[j] == 'LYS':
            x_p[j] = all_for_assign[8,:]
        elif xx[j] == 'LEU':
            x_p[j] = all_for_assign[9,:]
        elif xx[j] == 'MET':
            x_p[j] = all_for_assign[10,:]
        elif xx[j] == 'ASN':
            x_p[j] = all_for_assign[11,:]
        elif xx[j] == 'PRO':
            x_p[j] = all_for_assign[12,:]
        elif xx[j] == 'GLN':
            x_p[j] = all_for_assign[13,:]
        elif xx[j] == 'ARG':
            x_p[j] = all_for_assign[14,:]
        elif xx[j] == 'SER':
            x_p[j] = all_for_assign[15,:]
        elif xx[j] == 'THR':
            x_p[j] = all_for_assign[16,:]
        elif xx[j] == 'VAL':
            x_p[j] = all_for_assign[17,:]
        elif xx[j] == 'TRP':
            x_p[j] = all_for_assign[18,:]
        elif xx[j] == 'TYR':
            x_p[j] = all_for_assign[19,:]
    list_all.append(x_p)
    # for j in range(len(xx)):
    #             if xx[j] == 'ALA':
    #                 x_p[j] = np.array([-0.17691335, -0.19057421, 0.045527875, -0.175985, 1.1090639, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'CYS':
    #                 x_p[j] = np.array([-0.31572455, 0.38517416, 0.17325026, 0.3164464, 1.1512344, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    #             elif xx[j] == 'ASP':
    #                 x_p[j] = np.array([0.00600859, -0.1902303, -0.049640052, 0.15067418, 1.0812483, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    #             elif xx[j] == 'GLU':
    #                 x_p[j] = np.array([-0.06940994, -0.34011552, -0.17767446, 0.251, 1.0661993, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    #             elif xx[j] == 'PHE':
    #                 x_p[j] = np.array([0.2315121, -0.01626652, 0.25592703, 0.2703909, 1.0793934, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'GLY':
    #                 x_p[j] = np.array([-0.07281224, 0.01804472, 0.22983849, -0.045492448, 1.1139168, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'HIS':
    #                 x_p[j] = np.array([0.019046513, -0.023256639, -0.06749539, 0.16737276, 1.0796973, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'ILE':
    #                 x_p[j] = np.array([0.15077977, -0.1881559, 0.33855876, 0.39121667, 1.0793937, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'LYS':
    #                 x_p[j] = np.array([0.22048187, -0.34703028, 0.20346786, 0.65077996, 1.0620389, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'LEU':
    #                 x_p[j] = np.array([0.0075188675, -0.17002057, 0.08902198, 0.066686414, 1.0804346, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'MET':
    #                 x_p[j] = np.array([0.06302169, -0.10206237, 0.18976009, 0.115588315, 1.0927621, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'ASN':
    #                 x_p[j] = np.array([0.41597384, -0.22671205, 0.31179032, 0.45883527, 1.0529875, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'PRO':
    #                 x_p[j] = np.array([0.017954966, -0.09864355, 0.028460773, -0.12924117, 1.0974121, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'GLN':
    #                 x_p[j] = np.array([0.25189143, -0.40238172, -0.046555642, 0.22140719, 1.0362468, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    #             elif xx[j] == 'ARG':
    #                 x_p[j] = np.array([-0.15621762, -0.19172126, -0.209409, 0.026799612, 1.0879921, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'SER':
    #                 x_p[j] = np.array([0.17177454, -0.16769698, 0.27776834, 0.10357749, 1.0800852, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'THR':
    #                 x_p[j] = np.array([0.27962074, -0.051454283, 0.114876375, 0.3550331, 1.0615551, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'VAL':
    #                 x_p[j] = np.array([-0.09511698, -0.11654304, 0.1440215, -0.0022315443, 1.1064949, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'TRP':
    #                 x_p[j] = np.array([0.25281385, 0.12420933, 0.0132171605, 0.09199735, 1.0842415, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    #             elif xx[j] == 'TYR':
    #                 x_p[j] = np.array([0.27962074, -0.051454283, 0.114876375, 0.3550331, 1.0615551, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # list_all.append(x_p)

    

torch.save(list_all,'x_list_7.pt')
# bv = torch.load('x_list.pt')
# cv = bv[0]
# aaa = 1
#     # contacts = pdb_to_cm(open(pdb_file_name, "r"), 7.5)
#     # write_output(contacts, open('1133.cm', "w"))
# list_all = np.array(list_all)
# np.save('edge_list.npy',list_all)
#
# a = np.load('edge_list.npy', allow_pickle = True)
# a = a.tolist()
# bb = 1

