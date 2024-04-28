SYSTEM = '现在你是一个心理专家，我有一些心理问题，请你用专业的知识帮我解决。'
accumulative_counts = 2
alpaca_en = dict(
    dataset=dict(
        data_files=dict(train='/root/ft/data/data.json'),
        path='json',
        type='datasets.load_dataset'),
    dataset_map_fn=None,
    max_length=2048,
    pack_to_max_length=True,
    remove_unused_columns=True,
    shuffle_before_pack=True,
    template_map_fn=dict(
        template='xtuner.utils.PROMPT_TEMPLATE.internlm2_chat',
        type='xtuner.dataset.map_fns.template_map_fn_factory'),
    tokenizer=dict(
        padding_side='right',
        pretrained_model_name_or_path=
        '/root/share/model_repos/internlm2-chat-7b',
        trust_remote_code=True,
        type='transformers.AutoTokenizer.from_pretrained'),
    type='xtuner.dataset.process_hf_dataset')
batch_size = 8
betas = (
    0.9,
    0.999,
)
custom_hooks = [
    dict(
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path=
            '/root/share/model_repos/internlm2-chat-7b',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.DatasetInfoHook'),
    dict(
        evaluation_inputs=[
            '我最近总是感到很焦虑，尤其是在学业上。我有个特别崇拜的同学，他好像在各方面都比我优秀，我总觉得自己怎么努力也追不上他，这让我压力特别大。',
            '我知道应该理性看待，但就是忍不住会去比较。我甚至晚上会因为这个睡不着觉，总想着怎样才能像他那样出色。',
        ],
        every_n_iters=500,
        prompt_template='xtuner.utils.PROMPT_TEMPLATE.internlm2_chat',
        system='现在你是一个心理专家，我有一些心理问题，请你用专业的知识帮我解决。',
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path=
            '/root/share/model_repos/internlm2-chat-7b',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.EvaluateChatHook'),
]
data_path = '/root/ft/data/data.json'
dataloader_num_workers = 0
default_hooks = dict(
    checkpoint=dict(interval=1, type='mmengine.hooks.CheckpointHook'),
    logger=dict(interval=10, type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evaluation_freq = 500
evaluation_inputs = [
    '我最近总是感到很焦虑，尤其是在学业上。我有个特别崇拜的同学，他好像在各方面都比我优秀，我总觉得自己怎么努力也追不上他，这让我压力特别大。',
    '我知道应该理性看待，但就是忍不住会去比较。我甚至晚上会因为这个睡不着觉，总想着怎样才能像他那样出色。',
]
launcher = 'none'
load_from = None
log_level = 'INFO'
lr = 0.0002
max_epochs = 3
max_length = 2048
max_norm = 1
model = dict(
    llm=dict(
        pretrained_model_name_or_path=
        '/root/share/model_repos/internlm2-chat-7b',
        quantization_config=dict(
            bnb_4bit_compute_dtype='torch.float16',
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            llm_int8_has_fp16_weight=False,
            llm_int8_threshold=6.0,
            load_in_4bit=True,
            load_in_8bit=False,
            type='transformers.BitsAndBytesConfig'),
        torch_dtype='torch.float16',
        trust_remote_code=True,
        type='transformers.AutoModelForCausalLM.from_pretrained'),
    lora=dict(
        bias='none',
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        task_type='CAUSAL_LM',
        type='peft.LoraConfig'),
    type='xtuner.model.SupervisedFinetune')
optim_type = 'torch.optim.AdamW'
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=0.0002,
        type='torch.optim.AdamW',
        weight_decay=0),
    type='DeepSpeedOptimWrapper')
pack_to_max_length = True
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=0.09,
        start_factor=1e-05,
        type='mmengine.optim.LinearLR'),
    dict(
        T_max=3,
        begin=0.09,
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min=0.0,
        type='mmengine.optim.CosineAnnealingLR'),
]
pretrained_model_name_or_path = '/root/share/model_repos/internlm2-chat-7b'
prompt_template = 'xtuner.utils.PROMPT_TEMPLATE.internlm2_chat'
randomness = dict(deterministic=False, seed=None)
resume = False
runner_type = 'FlexibleRunner'
strategy = dict(
    config=dict(
        bf16=dict(enabled=True),
        fp16=dict(enabled=False, initial_scale_power=16),
        gradient_accumulation_steps='auto',
        gradient_clipping='auto',
        train_micro_batch_size_per_gpu='auto',
        zero_allow_untested_optimizer=True,
        zero_force_ds_cpu_optimizer=False,
        zero_optimization=dict(overlap_comm=True, stage=2)),
    exclude_frozen_parameters=True,
    gradient_accumulation_steps=2,
    gradient_clipping=1,
    sequence_parallel_size=1,
    train_micro_batch_size_per_gpu=8,
    type='xtuner.engine.DeepSpeedStrategy')
tokenizer = dict(
    padding_side='right',
    pretrained_model_name_or_path='/root/share/model_repos/internlm2-chat-7b',
    trust_remote_code=True,
    type='transformers.AutoTokenizer.from_pretrained')
train_cfg = dict(by_epoch=True, max_epochs=3, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='xtuner.dataset.collate_fns.default_collate_fn'),
    dataset=dict(
        dataset=dict(
            data_files=dict(train='/root/ft/data/data.json'),
            path='json',
            type='datasets.load_dataset'),
        dataset_map_fn=None,
        max_length=2048,
        pack_to_max_length=True,
        remove_unused_columns=True,
        shuffle_before_pack=True,
        template_map_fn=dict(
            template='xtuner.utils.PROMPT_TEMPLATE.internlm2_chat',
            type='xtuner.dataset.map_fns.template_map_fn_factory'),
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path=
            '/root/share/model_repos/internlm2-chat-7b',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.dataset.process_hf_dataset'),
    num_workers=0,
    sampler=dict(shuffle=True, type='mmengine.dataset.DefaultSampler'))
visualizer = None
warmup_ratio = 0.03
weight_decay = 0
work_dir = './work_dirs/internlm2_7b_chat_qlora_e3'
