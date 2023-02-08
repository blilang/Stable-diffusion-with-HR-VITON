训练流程：
1）在options/base_options.py中定位训练数据集。
2）用setup_model_weights.py生成初始权重。
3）在options/base_options.py调整参数。
4）运行python train.py。

推理流程
1）新建input_images文件夹，并把衣服图片放入。
2）运行python infer.py。
3）生成mask图片会在output_images文件夹内；若要生成白色背景的衣服，则生成在white_background文件夹内。