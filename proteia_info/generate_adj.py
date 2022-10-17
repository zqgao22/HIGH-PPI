import numpy as np
import csv
import os
from tqdm import tqdm
import sys
import math
import argparse
import re
from itertools import combinations
parser = argparse.ArgumentParser(description='make_adj_set')

parser.add_argument('--distance', default=None, type=float,
                    help="distance threshold")
args = parser.parse_args()

def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)


def read_atoms(file, chain=".", model=1):
    pattern = re.compile(chain)

    current_model = model
    atoms = []
    for line in file:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atoms.append((x, y, z))
        # elif line.startswith("MODEL"):
        #     current_model = int(line[10:14].strip())
    return atoms


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


def pdb_to_cm(file, threshold, chain=".", model=1):
    atoms = read_atoms(file, chain, model)
    return compute_contacts(atoms, threshold)

c = 0
count1 = -1
ensp = open('ensp_uniprot.txt')
e = ensp.read()
e_sp = e.split('ENSP')
list_all = []
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
            contacts = pdb_to_cm(open(pdb_file_name, "r"), args.distance)
            list_all.append(contacts)

            break

    
list_all = np.array(list_all)
np.save('edge_list_12.npy',list_all)
