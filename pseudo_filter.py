import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i','--input',type=str,required=True,help='pickle file of SGDet prediction')
parser.add_argument('-o','--output',type=str,required=True,help='output file name pickle')
args=parser.parse_args()

with open(args.input,'rb') as infile: 
    new_dic = pickle.load(infile)

for ids in list(new_dic.keys()): 
    if len(new_dic[ids]['rel_scores']) > 200: 
        del new_dic[ids]['rel_scores'][200:]
        del new_dic[ids]['rel_labels'][200:]
        del new_dic[ids]['rel_pairs'][200:]

for ids in list(new_dic.keys()): 
    if len(new_dic[ids]['bbox_scores']) > 25: 
        del new_dic[ids]['bbox'][25:]
        del new_dic[ids]['bbox_scores'][25:]
        del new_dic[ids]['bbox_labels'][25:]
        del new_dic[ids]['bbox_attributes'][25:]
    for i in reversed(range(len(new_dic[ids]['rel_pairs']))): 
        a,b = new_dic[ids]['rel_pairs'][i]
        if a >= 25 or b >= 25: 
            del new_dic[ids]['rel_pairs'][i]
            del new_dic[ids]['rel_scores'][i]
            del new_dic[ids]['rel_labels'][i]
            
with open(args.output,'wb') as out: 
    pickle.dump(new_dic,out,pickle.HIGHEST_PROTOCOL)