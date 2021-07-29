#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Modified by Kien Nguyen to print out some scene graphs he wanted to print

import tempfile, os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from imageio import imread

import argparse
import numpy as np


"""
Utilities for making visualizations.
"""


def draw_scene_graph(objs, triples, attributes, vocab=None, **kwargs):
    """
    Use GraphViz to draw a scene graph. If vocab is not passed then we assume
    that objs and triples are python lists containing strings for object and
    relationship names.
    Using this requires that GraphViz is installed. On Ubuntu 16.04 this is easy:
    sudo apt-get install graphviz
    """
    output_filename = kwargs.pop('output_filename', 'graph.png')
    orientation = kwargs.pop('orientation', 'V')
    edge_width = kwargs.pop('edge_width', 6)
    arrow_size = kwargs.pop('arrow_size', 1.5)
    binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
    ignore_dummies = kwargs.pop('ignore_dummies', True)
  
    if orientation not in ['V', 'H']:
        raise ValueError('Invalid orientation "%s"' % orientation)
    rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

    if vocab is not None:
    # Decode object and relationship names
        assert torch.is_tensor(objs)
        assert torch.is_tensor(triples)
        objs_list, triples_list = [], []
        for i in range(objs.size(0)):
            objs_list.append(vocab['object_idx_to_name'][objs[i].item()])
        for i in range(triples.size(0)):
            s = triples[i, 0].item()
            p = vocab['pred_name_to_idx'][triples[i, 1].item()]
            o = triples[i, 2].item()
            triples_list.append([s, p, o])
        objs, triples = objs_list, triples_list

    # General setup, and style for object nodes
    lines = [
        'digraph{',
        'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
        'rankdir=%s' % rankdir,
        'nodesep="0.5"',
        'ranksep="0.5"',
        'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
        'node [fillcolor="lightpink1"]',
    ]
    # Output nodes for objects
    for i, obj in enumerate(objs):
        if ignore_dummies and obj == '__image__':
            continue
        lines.append('%d [label="%s"]' % (i, obj))

    # Output relationships
    next_node_id = len(objs)
    lines.append('node [fillcolor="lightblue1"]')
    for s, p, o in triples:
        if ignore_dummies and p == '__in_image__':
            continue
        lines += [
            '%d [label="%s"]' % (next_node_id, p),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
              s, next_node_id, edge_width, arrow_size, binary_edge_weight),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
              next_node_id, o, edge_width, arrow_size, binary_edge_weight)
        ]
        next_node_id += 1

    #Output Attributes
    lines.append('node [fillcolor="lightseagreen"]')
    for a,o in attributes:
        lines += [
            '%d [label="%s"]' % (next_node_id, a),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
               o, next_node_id, edge_width, arrow_size, binary_edge_weight)
        ]
        next_node_id += 1
    lines.append('}')

  # Now it gets slightly hacky. Write the graphviz spec to a temporary
  # text file
    ff, dot_filename = tempfile.mkstemp()
    with open(dot_filename, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)
    os.close(ff)

  # Shell out to invoke graphviz; this will save the resulting image to disk,
  # so we read it, delete it, then return it.
    output_format = os.path.splitext(output_filename)[1][1:]
    os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
    os.remove(dot_filename)
    img = imread(output_filename)
    os.remove(output_filename)

    return img


if __name__ == '__main__':
    objs = []
    triples = []
    attributes = []
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--id', type=str, default='0',
                        help='id of image')
    parser.add_argument('--mode', type=str,  default='sen',
                        help='image or sen')
    opt = parser.parse_args()
    if opt.mode == 'sen':
        sg_dict = np.load('../data/spice_sg_dict2.npz', allow_pickle=True)['spice_dict'][()]
        sg_dict = sg_dict['ix_to_word']
        folder = '../data/all_ISG/' #change this to the folder with the SSGs
    else:
        sg_dict = np.load('coco_pred_sg_rela.npy')[()]
        sg_dict = sg_dict['i2w']
        folder = 'coco_img_sg/'
    sg_path = folder + opt.id + '.npy'
    sg_use = np.load(sg_path, allow_pickle=True)[()]
    if opt.mode == 'sen':
        rela = sg_use['rela_info']
        obj_attr = sg_use['obj_info']
    else:
        rela = sg_use['rela_matrix']
        obj_attr = sg_use['obj_attr']
    N_rela = len(rela)
    N_obj = len(obj_attr)
    for i in range(N_obj):
        if opt.mode == 'sen':
            singleObj = sg_dict[obj_attr[i][0]]
            if len(obj_attr[i]) >= 2:
                for j in range(len(obj_attr[i])-1):
                    attr = sg_dict[obj_attr[i][j + 1]]
                    attributes.append([attr,singleObj])
            objs.append(singleObj)
        else:
            print('obj #{0}'.format(i), end = ': ')  # maybe it means 'bounding box' but not 'object'
            N_attr = 3
            for j in range(N_attr - 1):
                print('{0} {1}, '.format(sg_dict[obj_attr[i][j + 4]],\
                    sg_dict[obj_attr[i][j+1]]), end = '')
            j = N_attr - 1
            print('{0} {1}'.format(sg_dict[obj_attr[i][j + 4]],\
                sg_dict[obj_attr[i][j+1]]))

    objsEncode = {s: i for i, s in enumerate(objs)}

    encodedAttr = []
    for a,o in attributes:
        encodObj = objsEncode[o]
        encodedAttr.append([a,encodObj])

    for i in range(N_rela):
        obj_idx = 0 if opt.mode == 'sen' else 1
        sbj = sg_dict[ int(obj_attr[int(rela[i][0])][obj_idx]) ]
        obj = sg_dict[ int(obj_attr[int(rela[i][1])][obj_idx]) ]
        rela_name = sg_dict[rela[i][2]]
        sbj = objsEncode[sbj]
        obj = objsEncode[obj]
        triples.append([sbj,rela_name,obj])

    vocab = None
    image = (plt.imshow(draw_scene_graph(objs, triples, encodedAttr, vocab, orientation='V')))
    savePath = './data/' + opt.id + '.png'
    plt.savefig(savePath)