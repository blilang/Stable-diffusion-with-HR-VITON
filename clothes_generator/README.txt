训练Stable-diffusion流程：
1）在my_train_style.sh脚本文件中定位初始权重和图片数据集的位置。
2）在my_train_style.sh脚本文件中修改参数。
3）在test_prompts_style.txt中修改验证prompts。
4）运行sh my_train_style.sh。

训练Dreambooth流程：
1）在my_train_object.sh脚本文件中定位初始权重和图片数据集的位置。
2）在my_train_object.sh脚本文件中修改参数。
3）在test_prompts_object.txt中修改验证prompts。
4）运行sh my_train_object.sh。


推理流程：
1）在test_model.py中修改权重路径。
2）在test_model.py中修改prompts和推理参数。
3）运行python test_model.py。