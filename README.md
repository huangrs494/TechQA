# TechQA
本项目旨在使用xtuner技术对心理健康类数据进行微调，实现心理问题的专业问答，并搭建一个方便使用的Demo。

## 项目文件介绍

# config
微调数据的配置文件，包括基座模型internlm2-chat-7b， 微调对话数据data.json，以及一些配置参数
max_length = 2048
pack_to_max_length = True
batch_size = 8 # per_device
accumulative_counts = 2
dataloader_num_workers = 0
max_epochs = 3
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03
evaluation_freq = 500

# hf
该文件夹的数据为微调后的模型转换成huggingface格式

# workdirs
包含了微调后的模型和日志配置文件等。cli_internlm2.py为demo文件，运行后，则可进行对话体验

# merged
基座模型和微调后的模型合并后生成的文件
