import csv
from collections import defaultdict
from pprint import pprint
import pandas as pd

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
    