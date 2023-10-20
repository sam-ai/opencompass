from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import storyclozeDataset

storycloze_reader_cfg = dict(
    input_columns=['context', 'sentence_quiz1', 'sentence_quiz2'],
    output_column='answer_right_ending',
    train_split='test',
    test_split='test')

storycloze_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            i: dict(round=[
                dict(role="HUMAN", prompt="### System:\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n\n\n\n### Instruction:\n{context}\n\n### Response:\n"),
                dict(role="BOT", prompt=f"{{sentence_quiz{i}}}"),
            ])
            for i in range(1, 3)
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

storycloze_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

# The original story cloze dataset and repo are not long maintaining.
# Using multilingual version of this dataset.
storycloze_datasets = [
    dict(
        abbr='story_cloze',
        type=storyclozeDataset,
        path='sam2ai/hindi_story_cloze_mini',
        # name='en',
        reader_cfg=storycloze_reader_cfg,
        infer_cfg=storycloze_infer_cfg,
        eval_cfg=storycloze_eval_cfg)
]
