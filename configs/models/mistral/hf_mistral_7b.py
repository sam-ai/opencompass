from opencompass.models import HuggingFaceCausalLM

models = [
    # LLaMA 7B
    dict(
        type=HuggingFaceCausalLM,
        abbr='mistral_hindi_7b_hf',
        path="OdiaGenAI/mistral_hindi_7b_base_v1",
        tokenizer_path='OdiaGenAI/mistral_hindi_7b_base_v1',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
