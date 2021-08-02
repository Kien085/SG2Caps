# SG2Caps
SG2Caps: In Defense of Scene Graphs for Image Captioning, ICCV 2021

# Acknowledgement
This code is implemented based on Ruotian Luo's implementation of image captioning in https://github.com/ruotianluo/self-critical.pytorch.

The training code is implemented based on Yang Xu's implementation of SGAE in https://github.com/yangxuntu/SGAE.

# Installation anaconda and the environment
Follow the instructions to set up the environment/data from Yang Xu's README.

# Additional Dependency
Pseudolabel is generated using Neural Motif through Scene Graph Benchmark in Pytorch in https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch. 
HOI is generated through VSGNet https://github.com/ASMIftekhar/VSGNet
```
python pseudo_filter.py -i [SGDet pickle] -o [output path]
python hoi_partial_graph.py -i [VSG output] -o [output filename] -p [output path]
```
We will provide the processed VisualGenome Data. 

# Additional Data
In addition, download cocobu2_append.json and the all_ISG folder, and place them into the data folder.

# Training the model
1. After setting up the SGAE environment, and downloading the additional data:
```
python train_mem.py --id id08 --caption_model lstm_mem4 --input_json data/cocobu2_appended.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_rela_dir data/cocobu_img_sg --input_ssg_dir data/all_ISG --input_label_h5 data/cocobu2_label.h5 --sg_dict_path data/spice_sg_dict2.npz --batch_size 50 --accumulate_number 2 --learning_rate 5e-4 --learning_rate_decay_start 0 --learning_rate_decay_every 3 --scheduled_sampling_start 0 --checkpoint_path storedData/id08 --save_checkpoint_every 3000 --val_images_use 5000 --max_epochs 150 --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --self_critical_after 20 --train_split train --memory_size 10000 --memory_index c --step2_train_after 100 --step3_train_after 200 --use_rela 0 --gpu 0
```

2.The Google Drive also contains a pre-trained model that we used to get the numbers reported in the paper, under id13.

3.The details of parameters:

--id: the id of your model, which is usually set as the same as check point, which is helpful for you to train from the check point.

--batch_size, --accumulate_number: these two parameters are set for users who do not have large gpu, if you want to set batch size as 100, you can set batch_size as 50, and set accumulate_number as 2, also you can set batch_size as 20 and accumulate_number as 5. Importantly, they are not totally equal to set batch_size as 100 and accumulate_number as 1, the bigger the bathc_size is, the higher the performance.

--self_critical_after: when reinforcement leanring begins, if this value is set as 40, it means that after training 40 epoches, the reinforcement loss is used. Generally, if you want to have a good CIDEr-D score, you should use cross entropy loss first and then use reinforcement loss.

--checkpoint_path: this is where the models will be saved. Create the folders before starting the training.

4.Tranining from checkpoints.
In order to continue training from a checkpoint, just add a start_from parameter to the end of the training code.
The codes provide the ability of training from checkpoints. For example, if you want to train the model from one checkpoint, say, 22, you can use the following code to continute:
```
python train_mem.py --id id08 --caption_model lstm_mem4 --input_json data/cocobu2_appended.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_rela_dir data/cocobu_img_sg --input_ssg_dir data/all_ISG --input_label_h5 data/cocobu2_label.h5 --sg_dict_path data/spice_sg_dict2.npz --batch_size 50 --accumulate_number 2 --learning_rate 5e-4 --learning_rate_decay_start 0 --learning_rate_decay_every 3 --scheduled_sampling_start 0 --checkpoint_path storedData/id08 --save_checkpoint_every 3000 --val_images_use 5000 --max_epochs 150 --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --self_critical_after 20 --train_split train --memory_size 10000 --memory_index c --step2_train_after 100 --step3_train_after 200 --use_rela 0 --gpu 0 --start_from 16
```

# Evaluating the model
1.To evaluate your code, run the following:
```
python eval_mem.py --dump_images 0 --num_images 5000 --model storedData/id06/modelid060026.pth --infos_path storedData/id06/infos_id060026.pkl --language_eval 1 --beam_size 1 --split test --index_eval 1 --use_rela 0 --training_mode 0 --memory_cell_path storedData/id06/memory_cellid060026.npz --sg_dict_path data/spice_sg_dict2.npz --input_ssg_dir data/all_ISG
```
Make sure that model, infos_path, and memory_cell_path are pointing to the correct files and folders.

# Creating Scene Graph Visualizations
The two python scripts inside of the vis folder will create the scene graph visualizations from the input_ssg_dir that is normally used to train the model (all_ISG, HOI_ISG). You will need to change the folder variable to point to the correct folder, depending on the SSG.

To create only one SSG for one image. The resulting image will be saved in the ./data folder
```
python test.py --mode sen --id 391895
```

For all images, run vis_sg.py. The resulting images will be saved in the ./data/vis folder. Make sure this folder exists before running.
