from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import EDAccEvaluator
from opencompass.datasets import siqaDataset_V2, HFDataset

siqa_reader_cfg = dict(
    input_columns=["context", "question", "answerA", "answerB", "answerC"],
    output_column="all_labels",
    test_split="validation")

siqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    # "{context}\nQuestion: {question}\nA. {answerA}\nB. {answerB}\nC. {answerC}\nAnswer:"
                    "### System:\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n\n\n### Instruction:\n{question}\nA. {answerA}\nB. {answerB}\nC. {answerC}\n\n### Input:\n{context}\n\n### Response:\n"
                )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

siqa_eval_cfg = dict(
    evaluator=dict(type=EDAccEvaluator),
    pred_role="BOT",
)

siqa_datasets = [
    dict(
        abbr="siqa",
        type=siqaDataset_V2,
        path="sam2ai/hindi_siqa_mini",
        reader_cfg=siqa_reader_cfg,
        infer_cfg=siqa_infer_cfg,
        eval_cfg=siqa_eval_cfg)
]
