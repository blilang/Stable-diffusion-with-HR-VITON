配置环境：
1）运行conda create -n {env_name} python=3.8。
2）运行conda activate {env_name}。
3）运行conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia。
4）运行pip install opencv-python torchgeometry Pillow tqdm tensorboardX scikit-image scipy。

训练条件生成器：
1）往data文件夹的train文件夹中放入数据集。
2）运行python3 train_condition.py --cuda {True} --gpu_ids {gpu_ids} --Ddownx2 --Ddropout --lasttvonly --interflowloss --occlusion。

训练特征融合网络：
1）往data文件夹的train文件夹中放入数据集。
2）运行python3 train_generator.py --cuda {True} --name test -b 4 -j 8 --gpu_ids {gpu_ids} --fp16 --tocg_checkpoint {condition generator ckpt path} --occlusion。

推理流程：
1）往data文件夹的test文件夹中放入数据集。
2）运行python3 test_generator.py --occlusion --cuda {True} --test_name {test_name} --tocg_checkpoint {condition generator ckpt} --gpu_ids {gpu_ids} --gen_checkpoint {image generator ckpt} --datasetting unpaired --dataroot {dataset_path} --data_list {pair_list_textfile}。