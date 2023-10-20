from mmengine.config import read_base

with read_base():
    from .datasets.siqa.siqa_gen import siqa_datasets
    # from .datasets.winograd.winograd_ppl import winograd_datasets
    # from .models.opt.hf_opt_125m import opt125m
    from .models.mistral.hf_mistral_7b import mistral7b

datasets = [*siqa_datasets]
models = [mistral7b]