import datasets
import random
import os

name_mapping = {
    'sst2': ['SetFit/sst2'],
    'sst5': ['SetFit/sst5'],
    'cr': ['SetFit/SentEval-CR'],
    'mr': ['rotten_tomatoes'],
    'subj': ['SetFit/subj'],
    'trec': ['trec'],
    'agnews': ['ag_news'],
    'dbpedia': ['dbpedia_14'],
    'rte': ['SetFit/rte'],
    'cb': ['super_glue', 'cb'],
    'snli': ['snli'],
    'qqp': ['SetFit/qqp']
}

def load_dataset(dataset_name):
    try:
        return datasets.load_from_disk(f'data/{dataset_name}')
    except:
        dataset = datasets.load_dataset(*name_mapping[dataset_name])
        dataset.save_to_disk(f'data/{dataset_name}')
        return dataset

def get_dataset(dataset_name, test_amount=None):
    # sentiment analysis
    dataset = load_dataset(dataset_name)
    if dataset_name == 'sst2':
        dataset.prompt_texts = ['Review', 'Sentiment']
        dataset.label_tokens = ['negative', 'positive']

    elif dataset_name == 'sst5':
        label_tokens = ['terrible', 'bad', 'neutral', 'good', 'great']
        def format(example):
            example['label_text'] = label_tokens[example['label']]
            return example
        dataset = dataset.map(format)
        dataset.prompt_texts = ['Review', 'Sentiment']
        dataset.label_tokens = label_tokens
    
    elif dataset_name == 'cr':
        dataset.prompt_texts = ['Review', 'Sentiment']
        dataset.label_tokens = ['negative', 'positive']

    elif dataset_name == 'mr':
        label_tokens = ['negative', 'positive']
        def format(example):
            example['label_text'] = label_tokens[example['label']]
            return example
        dataset = dataset.map(format)
        dataset.prompt_texts = ['Review', 'Sentiment']
        dataset.label_tokens = label_tokens


    # subjectivity classification
    elif dataset_name == 'subj':
        dataset.prompt_texts = ['Input', 'Type']
        dataset.label_tokens = ['objective', 'subjective']
    
    # topic classification
    elif dataset_name == 'trec':
        label_tokens = [
            'abbreviation', 
            'entity', 
            'description', 
            'person', 
            'location',
            'number']
        def format(example):
            example['label'] = example['coarse_label']
            example['label_text'] = label_tokens[example['coarse_label']]
            return example
        dataset = dataset.map(format)
        dataset.prompt_texts = ['Question', 'Answer type']
        dataset.label_tokens = label_tokens

    elif dataset_name == 'agnews':
        label_tokens = [
            'world',
            'sports',
            'business',
            'technology'
        ]
        def format(example):
            example['label_text'] = label_tokens[example['label']]
            return example
        dataset = dataset.map(format)
        dataset.prompt_texts = ['Input', 'Type']
        dataset.label_tokens = label_tokens
    
    elif dataset_name == 'dbpedia':
        dataset = dataset.rename_column('content', 'text')
        label_tokens = [
            'company',
            'school',
            'artist',
            'athlete',
            'politics',
            'transportation',
            'building',
            'nature',
            'village',
            'animal',
            'plant',
            'album',
            'film',
            'book'
        ]
        def format(example):
            example['label_text'] = label_tokens[example['label']]
            return example
        dataset = dataset.map(format)
        dataset.prompt_texts = ['Input', 'Type']
        dataset.label_tokens = label_tokens
    
    # text entailment
    elif dataset_name == 'rte':
        label_tokens = ['true', 'false']
        def format(example):
            example['label_text'] = label_tokens[example['label']]
            return example
        dataset = dataset.map(format)
        dataset.prompt_texts = ['Premise', 'Hypothesis', 'Prediction']
        dataset.label_tokens = label_tokens
        dataset['test'] = dataset['validation']
        del dataset['validation']

    elif dataset_name == 'cb':
        dataset = dataset.rename_column('premise', 'text1')
        dataset = dataset.rename_column('hypothesis', 'text2')
        label_tokens = ['true', 'false', 'neither']
        def format(example):
            example['label_text'] = label_tokens[example['label']]
            return example
        dataset = dataset.map(format)
        dataset.prompt_texts = ['Premise', 'Hypothesis', 'Prediction']
        dataset.label_tokens = label_tokens
        dataset['test'] = dataset['validation']
        del dataset['validation']
    
    elif dataset_name == 'snli':
        dataset = dataset.filter(lambda e: e['label']!=-1)
        dataset = dataset.rename_column('premise', 'text1')
        dataset = dataset.rename_column('hypothesis', 'text2')
        label_tokens = ['entailment', 'neutral', 'contradiction']
        def format(example):
            example['label_text'] = label_tokens[example['label']]
            return example
        dataset = dataset.map(format)
        dataset.prompt_texts = ['Premise', 'Hypothesis', 'Prediction']
        dataset.label_tokens = label_tokens

    elif dataset_name == 'qqp':
        label_tokens = ['false', 'true']
        def format(example):
            example['label_text'] = label_tokens[example['label']]
            return example
        dataset = dataset.map(format)
        dataset.prompt_texts = ['Question1', 'Question2', 'Prediction']
        dataset.label_tokens = label_tokens
        
        dataset['test'] = dataset['validation']
        del dataset['validation']

    dataset.prefix = ''
    dataset.delimiter = '\n'

    if test_amount is not None and test_amount < len(dataset['test']):
        random.seed(1116)
        remain_ids = set(random.sample(range(len(dataset['test'])), k=test_amount))
        dataset['test'] = dataset['test'].filter(lambda e,i: i in remain_ids, with_indices=True)

    return dataset