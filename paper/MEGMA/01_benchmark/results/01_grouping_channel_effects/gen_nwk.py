# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 14:06:17 2021

@author: Shen Wanxiang
"""

import csv
from collections import defaultdict
from pprint import pprint
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from scipy.spatial.distance import squareform

itol_header = '''TREE_COLORS
SEPARATOR TAB

#First 3 fields define the node id, type and color
#Possible types are:
#'range': defines a colored range (colored background for labels/clade)
#'clade': defines color/style for all branches in a clade
#'branch': defines color/style for a single branch
#'label': defines font color/style for the leaf label
#'label_background': defines the leaf label background color

#The following additional fields are required:
#for 'range', field 4 defines the colored range label (used in the legend)

#The following additional fields are optional:
#for 'label', field 4 defines the font style ('normal',''bold', 'italic' or 'bold-italic') and field 5 defines the numeric scale factor for the font size (eg. with value 2, font size for that label will be 2x the standard size)
#for 'clade' and 'branch', field 4 defines the branch style ('normal' or 'dashed') and field 5 defines the branch width scale factor (eg. with value 0.5, branch width for that clade will be 0.5 the standard width)

DATA
#NODE_ID TYPE COLOR LABEL_OR_STYLE SIZE_FACTOR
'''

def _getNewick(node, newick, parentdist, leaf_names):
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = _getNewick(node.get_left(), newick, node.dist, leaf_names)
        newick = _getNewick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
        newick = "(%s" % (newick)
        return newick
    
def mp2newick(mp, treefile = 'phenotype_tree'):

    leaf_names = mp.alist
    linkage_matrix = mp.Z
    tree = to_tree(linkage_matrix, rd=False)
    newick = _getNewick(tree, "", tree.dist, leaf_names = leaf_names)
    
    # write newick file for itol
    with open(treefile + '.nwk', 'w') as f:
        f.write(newick)

    # write dataset file for itol
    df = mp.df_embedding[['colors','Subtypes']]
    df['TYPE'] = 'clade'
    df['STYLE'] = 'normal'
    df = df[['TYPE', 'colors', 'STYLE']]
    with open(treefile + '.txt', 'w') as f:
        f.write(itol_header)
    df.to_csv(treefile + '.txt', mode = 'a', header=None, sep='\t')
    return df
    
def tree(): 
    return defaultdict(tree)

def tree_add(t, path):
    for node in path:
        t = t[node]

def pprint_tree(tree_instance):
    def dicts(t): return {k: dicts(t[k]) for k in t}
    pprint(dicts(tree_instance))

def dfs_to_tree(dfs):
    t = tree()
    for i in range(len(dfs)):
        row = dfs.iloc[i].dropna().tolist()
        tree_add(t, row)
    return t

def tree_to_newick(root):
    items = []
    for k in root.keys():
        s = ''
        if len(root[k].keys()) > 0:
            sub_tree = tree_to_newick(root[k])
            if sub_tree != '':
                s += '(' + sub_tree + ')'
        s += k
        items.append(s)
    return ','.join(items)

def dfs_to_weightless_newick(dfs):
    t = dfs_to_tree(dfs)
    newick_tree = tree_to_newick(t)
    return newick_tree


if __name__ == '__main__':
    
    species_list = pd.read_csv('./species.list.csv', header=None,index_col=0)[1].to_list()
    dfs = pd.Series(species_list).apply(lambda x: dict([i.split('__') for i in x.split('|')])).apply(pd.Series)
    level_dict = {'k':'kingdom', 'p':'phylum', 'c':'class' ,'o':'order' ,'f':'family' ,'g': 'genus','s': 'species'}
    dfs = dfs.rename(columns=level_dict)
    nwk_string = dfs_to_weightless_newick(dfs)
    with open("1.nwk", "w") as f:
        f.write(nwk_string)
    