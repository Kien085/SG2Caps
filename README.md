# SG2Caps
SG2Caps: In Defense of Scene Graphs for Image Captioning, ICCV 2021

# Acknowledgement
This code is implemented based on Ruotian Luo's implementation of image captioning in https://github.com/ruotianluo/self-critical.pytorch.

The training code is implemented based on Yang Xu's implementation of SGAE in https://github.com/yangxuntu/SGAE.

# Installation anaconda and the environment
The environment file is provided in environment_yx1.yml. Create the environment with the following:
```
conda env create -f environment_yx1.yml
```
Activate the environment with the following:
```
conda activate myenv
```

# Additional Dependency
Pseudolabel is generated using Neural Motif through Scene Graph Benchmark in Pytorch in https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch. 
HOI is generated through VSGNet https://github.com/ASMIftekhar/VSGNet
```
python pseudo_filter.py -i [SGDet pickle] -o [output path]
python hoi_partial_graph.py -i [VSG output] -o [output filename] -p [output path]
```
We will provide the processed VisualGenome Data. 

# Data
Create a data folder to hold all of the data.
Run the following to download the bottom up features needed for the SGAE dependency.
```
mkdir data/bu_data; cd data/bu_data
wget https://storage.googleapis.com/up-down-attention/trainval.zip
unzip trainval.zip

#After unzipping
cd ../..
python scripts/make_bu_data.py --output_dir data/cocobu
```


Google Drive link: https://drive.google.com/file/d/1K6h8aupnQJ2v3IZdPNvqZTeGbH3qQ954/view?usp=sharing

Download cocobu2_label.h5, cocobu2_append.json, and spice_sg_dict2.npz. 

The VSGs can be found in the all_ISG and the HOI_ISG folder.

The global features can be found in the box_max.zip after unzipping in the max_pool folder.

# Training the model
1. After setting up the environment, and downloading the additional data train with this code.
2. Make sure that the checkpoint_path already exists. If not, the model will not save.
```
python train_mem.py --id id01 --caption_model lstm_mem4 --input_json data/cocobu2_appended.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_ssg_dir data/all_ISG --input_label_h5 data/cocobu2_label.h5 --sg_dict_path data/spice_sg_dict2.npz --batch_size 100 --accumulate_number 1 --learning_rate 5e-4 --learning_rate_decay_start 0 --learning_rate_decay_every 3 --scheduled_sampling_start 0 --checkpoint_path data/id11 --save_checkpoint_every 500 --val_images_use 5000 --max_epochs 150 --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --self_critical_after 210 --train_split train --memory_size 10000 --memory_index c --step2_train_after 300 --step3_train_after 400 --use_rela 0 --gpu 0
```

2.The Google Drive also contains pre-trained models that we used to get the numbers reported in the paper. SG2Caps is id09, SG2Caps Global is id14.
(Need to find baseline, bbox, and global. Baseline and bbox are probably on the old server. Also find where RL is for SG2Caps) 

3.The details of parameters:

--id: The id of the model, useful for training multiple models or different ablations

--batch_size, --accumulate_number: You can use a lower batch size with a higher accumulate number to simulate having a larger batch size, for users without big GPUs.  

--self_critical_after: The amount of epochs before the model switches from cross entropy to reinforcement learning.

--checkpoint_path: This is where the models will be saved. Create the folders before starting the training.

Additional parameters for ablations:
--use_bbox: If set to 0, trains the model without bbox.
--use_globals: If set to 0, trains the model without global features.

4.Tranining from checkpoints.
In order to continue training from a checkpoint, just add a start_from parameter to the end of the training code.
The codes provide the ability of training from checkpoints. For example, if you want to train the model from one checkpoint, say, 22, you can use the following code to continute:
```
python train_mem.py --id id08 --caption_model lstm_mem4 --input_json data/cocobu2_appended.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_rela_dir data/cocobu_img_sg --input_ssg_dir data/all_ISG --input_label_h5 data/cocobu2_label.h5 --sg_dict_path data/spice_sg_dict2.npz --batch_size 50 --accumulate_number 2 --learning_rate 5e-4 --learning_rate_decay_start 0 --learning_rate_decay_every 3 --scheduled_sampling_start 0 --checkpoint_path storedData/id08 --save_checkpoint_every 3000 --val_images_use 5000 --max_epochs 150 --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --self_critical_after 20 --train_split train --memory_size 10000 --memory_index c --step2_train_after 100 --step3_train_after 200 --use_rela 0 --gpu 0 --start_from 22
```

# Evaluating the model
1.To evaluate your code, run the following:
```
python eval_mem.py --dump_images 0 --num_images 5000 --model data/id09/modelid090019.pth --infos_path data/id06/infos_id090019.pkl --language_eval 1 --beam_size 1 --split test --index_eval 1 --use_rela 0 --training_mode 0 --memory_cell_path data/id09/memory_cellid090019.npz --sg_dict_path data/spice_sg_dict2.npz --input_ssg_dir data/all_ISG --use_globals 0
```
Make sure that model, infos_path, and memory_cell_path are pointing to the correct files and folders.

# Creating Scene Graph Visualizations
The two python scripts inside of the vis folder will create the scene graph visualizations from the input_ssg_dir that is normally used to train the model (all_ISG, HOI_ISG). You will need to change the folder variable to point to the correct folder, depending on the SSG.

To create only one SSG for one image. The resulting image will be saved in the ./data folder
```
python test.py --mode sen --id 391895
```

For all images, run vis_sg.py. The resulting images will be saved in the ./data/vis folder. Make sure this folder exists before running.
