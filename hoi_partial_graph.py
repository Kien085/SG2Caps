import pickle
import json
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i','--input',type=str,required=True,help='input file of VSGNet output')
parser.add_argument('-o','--output',type=str,required=True,help='output file name json')
parser.add_argument('-p','--obj_path',type=str,required=True,help='path of object detection')
args=parser.parse_args()

f = open(args.input,'rb')
OBJ_PATH_test_s=args.obj_path
detect_list = pickle.load(f)

proper_keys=['carry_agent', 'carry_obj', 'catch_agent', 'catch_obj', 'cut_agent', 'cut_instr', 'cut_obj', 'drink_agent', 'drink_instr', 
             'eat_agent', 'eat_instr', 'eat_obj', 'hit_agent', 'hit_instr', 'hit_obj', 'hold_agent', 'hold_obj', 'jump_agent', 
             'jump_instr', 'kick_agent', 'kick_obj', 'lay_agent', 'lay_instr', 'look_agent', 'look_obj', 'point_agent', 
             'point_instr', 'read_agent', 'read_obj', 'ride_agent', 'ride_instr', 'run_agent', 
             'sit_agent', 'sit_instr', 'skateboard_agent', 'skateboard_instr', 'ski_agent', 'ski_instr', 'smile_agent', 
             'snowboard_agent', 'snowboard_instr', 'stand_agent', 'surf_agent', 'surf_instr', 'talk_on_phone_agent', 
             'talk_on_phone_instr', 'throw_agent', 'throw_obj', 'walk_agent', 'work_on_computer_agent', 'work_on_computer_instr']

# modified for rel_pairs
sorted_dict = {}
for detect in detect_list: 
    img_id = detect['image_id']
    p_bbox = detect['person_box']
    
    result = list(detect.values())[2:]
    agent_idx = None
    agent_score = -1
    obj_bbox = None
    obj_score = -1
    instr_bbox = None
    instr_score = -1
    
    for i in range(len(result)): 
        if proper_keys[i][-5:] == 'agent': 
            if result[i] > agent_score: 
                
                agent_score = result[i]
                agent_idx = i

    if agent_score < 0.5: 
        continue
    
    if agent_idx + 1 < len(result) and proper_keys[agent_idx+1][-5:] == 'instr': 
        instr_score = result[agent_idx+1][-1]
        instr_bbox = result[agent_idx+1][:4]
        if agent_idx + 2 < len(result) and proper_keys[agent_idx+2][-3:] == 'obj': 
            obj_score = result[agent_idx+2][-1]
            obj_bbox = result[agent_idx+2][:4]
    elif agent_idx + 1 < len(result) and proper_keys[agent_idx+1][-3:] == 'obj': 
        obj_score = result[agent_idx+1][-1]
        obj_bbox = result[agent_idx+1][:4]
            
    
    if not str(img_id) in sorted_dict: 
        sorted_dict[str(img_id)] = {}
        sorted_dict[str(img_id)]['bbox'] = []
        sorted_dict[str(img_id)]['rel_pairs'] = []
        sorted_dict[str(img_id)]['rel_label'] = []
        sorted_dict[str(img_id)]['attributes'] = [] 
        sorted_dict[str(img_id)]['rel_scores'] = []
        
    if instr_bbox != None: 
        sorted_dict[str(img_id)]['bbox'].append(instr_bbox)
        sorted_dict[str(img_id)]['attributes'].append(None)
    
    sorted_dict[str(img_id)]['bbox'].append(p_bbox)
    if obj_bbox == None: 
        sorted_dict[str(img_id)]['attributes'].append(proper_keys[agent_idx][:-6])
    else: 
        sorted_dict[str(img_id)]['bbox'].append(obj_bbox)
        sorted_dict[str(img_id)]['attributes'].append(None)
        sorted_dict[str(img_id)]['attributes'].append(None)
        
        sorted_dict[str(img_id)]['rel_pairs'].append([len(sorted_dict[str(img_id)]['bbox']) - 2, \
                                                      len(sorted_dict[str(img_id)]['bbox']) - 1])
        sorted_dict[str(img_id)]['rel_label'].append(proper_keys[agent_idx][:-6])
        sorted_dict[str(img_id)]['rel_scores'].append([agent_score, instr_score, obj_score])



all_detec = {}
eval_list = [j for j in os.listdir(OBJ_PATH_test_s) if j.endswith('.json')]
for file in eval_list: 
    select = []
    img_id = int(file.split('_')[-1][:-5])
    cur_obj_path_s = OBJ_PATH_test_s + file
    with open(cur_obj_path_s) as fp:detections = json.load(fp)
    H = detections['H']
    W = detections['W']
    for det in detections['detections']: 
        new_det = {}
        if det['score'] < 0.5: 
            continue
        new_det['class_str'] = det['class_str']
        new_det['score'] = det['score']
        new_det['class_no'] = det['class_no']
        top,left,bottom,right = det['box_coords']
        left, top, right, bottom = left*W, top*H, right*W, bottom*H
        new_det['box_coords'] = [left,top,right,bottom]
        select.append(new_det)
    all_detec[str(img_id)] = select

    
partial_graph = {}
for img_id, objects in all_detec.items(): 
    class_str = []
    score = []
    class_no = []
    box_coords = []
    obj_dict = {}
    for obj in objects: 
        class_str.append(obj['class_str'])
        score.append(obj['score'])
        class_no.append(obj['class_no'])
        box_coords.append(obj['box_coords'])
    
    obj_dict['class_str'] = class_str
    obj_dict['score'] = score
    obj_dict['class_no'] = class_no
    obj_dict['box_coords'] = box_coords
    partial_graph[img_id] = obj_dict
    
for img_id, vsg_result in sorted_dict.items(): 
    new_relation = []
    new_rel_score = []
    for i, (subj, obj) in enumerate(vsg_result['rel_pairs']): 
        sub_box = vsg_result['bbox'][subj]
        ob_box = vsg_result['bbox'][obj]
        new_sub_index = None
        new_ob_index = None
        for j,detect in enumerate(partial_graph[img_id]['box_coords']): 
            
            if (np.array(sub_box) - np.array(detect)).all() < 1e-8: 
                new_sub_index = j
            if (np.array(ob_box) - np.array(detect)).all() < 1e-8: 
                new_ob_index = j
        
        if new_sub_index != None and new_ob_index != None: 
            new_relation.append((new_sub_index, new_ob_index, vsg_result['rel_label'][i]))
            new_rel_score.append(vsg_result['rel_scores'][i])
    
    partial_graph[img_id]['relation'] = new_relation
    partial_graph[img_id]['rel_score'] = new_rel_score

for img_id, objects in partial_graph.items(): 
    if not img_id in sorted_dict.keys(): 
        continue
    attributes = []
    for i, box_coords in enumerate(objects['box_coords']): 
        bbox_index = []        
        local_attributes = []
        for j, bbox in enumerate(sorted_dict[img_id]['bbox']): 
            if (np.array(box_coords) - np.array(bbox)).all() < 1e-8: 
                bbox_index.append(j)

        for index in bbox_index: 
            if sorted_dict[img_id]['attributes'][index] != None: 
                local_attributes.append(sorted_dict[img_id]['attributes'][index])
        if len(local_attributes) == 0: 
            local_attributes.append(None)
        attributes.append(local_attributes)
    
    objects['attributes'] = attributes

    
with open(args.output,'w') as f: 
    json.dump(partial_graph,f)