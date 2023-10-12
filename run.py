import os
import argparse
import torch
import random
import pickle
import pandas as pd
import warnings
import debug

warnings.filterwarnings("ignore")

from src.model import InContextLearner
from src.dataset import get_dataset
from src.utils import evaluate, test_prompt


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_name', 
    type=str, choices=[
        'gpt2-0.8B',
        'gpt2-1.5B',
        'gpt-neo-2.7B',
        'gpt-j-6B',
        'rwkv-3B',
        'rwkv-7B',
        'rwkv-14B',
        'gpt-neox-20B',
        'opt-13B',
        'opt-30B',
        'llama-13B',
        'llama-30B'
    ]
)
parser.add_argument(
    '--dataset_name', 
    type=str, choices=[
        'sst2', 
        'sst5', 
        'cr', 
        'mr', 
        'subj', 
        'trec', 
        'agnews', 
        'dbpedia', 
        'rte', 
        'cb', 
        'snli', 
        'qqp'
    ]
)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--save_path', type=str, default='saved')
parser.add_argument('--n_samples_once', type=int, default=2)
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--test_amount', type=int, default=2000)

args = parser.parse_args()

path = f'{args.save_path}/{args.model_name}_{args.dataset_name}'
try:
    os.makedirs(path)
except:
    pass

result_fname = os.path.join(path, 'results.csv')
info_fname = os.path.join(path, 'info.pkl')

df = pd.DataFrame(
    {
        "method": [],
        "shot": [],
        "seed": [],
        "accuracy": [],
        "macro-precision": [],
        "macro-recall": [],
        "macro-f1": [],
        "macro-auroc": []
    }
)

model = InContextLearner(args.model_name)
dataset = get_dataset(args.dataset_name, test_amount=args.test_amount)

print(f'example prompt: {test_prompt(dataset)}')

text_keys = sorted(list(filter(lambda key: key.startswith('text'), dataset.column_names['train'])))
test_data = []
gold_labels = []
for split in ('train', 'validation', 'test'):
    gold_labels.extend(dataset[split]['label'])
    for instance in dataset[split]:
        test_example = [instance[text_key] for text_key in text_keys]
        test_data.append(test_example)
gold_labels = torch.tensor(gold_labels)

for shot in (2, 4, 8):
    for seed in range(5):
        if os.path.exists(os.path.join(path, f'info.{shot}.{seed}.pkl')):
            continue
        record = {
            'shot': shot,
            'seed': seed,
        }
        random.seed(seed)
        train_data = []
        for instance in random.choices(dataset['train'], k=shot):
            train_example = [instance[text_key] for text_key in text_keys]
            train_example.append(instance['label_text'])
            train_data.append(train_example)

        record['method'] = 'icl'
        icl_label_logprobs = model.predict(
            train_data=train_data, 
            test_data=test_data,
            label_tokens=dataset.label_tokens,
            prefix=dataset.prefix,
            delimiter=dataset.delimiter,
            prompt_texts=dataset.prompt_texts,
            batch_size=args.batch_size)['label_logprobs']
        info['label_logprobs'] = icl_label_logprobs
        results = evaluate(gold_labels, icl_label_logprobs)
        record.update(results)
        df.loc[len(df.index)] = record

        record['method'] = 'gc'
        outputs = model.generative_calibrate_predict(
            train_data=train_data,
            test_data=test_data,
            label_tokens=dataset.label_tokens,
            prefix=dataset.prefix,
            delimiter=dataset.delimiter,
            prompt_texts=dataset.prompt_texts,
            n_samples=args.n_samples,
            n_samples_once=args.n_samples_once,
            seed=seed,
            max_new_tokens=384,
            ori_label_logprobs=icl_label_logprobs,
            batch_size=args.batch_size
        )
        info['gc_sampled_sequences'] = outputs['sampled_sequences']
        info['gc_label_logprobs'] = outputs['caled_label_logprobs']
        info['gc_estimated_label_logmarginal'] = outputs['estimated_label_logmarginal']

        results = evaluate(gold_labels, outputs[f'caled_label_logprobs'])
        record.update(results)
        df.loc[len(df.index)] = record
        
        df.to_csv(result_fname, index=False)
        with open(os.path.join(path, f'info.{shot}.{seed}.pkl'), 'wb') as f:
            pickle.dump(info, f)