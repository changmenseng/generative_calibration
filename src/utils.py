import re
import random
import itertools
import numpy as np
import torch
import math
from typing import Optional
from sklearn.metrics import classification_report, roc_auc_score

def construct_exemplar_template(
    delimiter,
    prompt_texts,
    field_prefix: Optional[str] = ' '
):
    exemplar_template = ''
    pfixed_prompt_texts = []
    for i, prompt_text in enumerate(prompt_texts):
        prefix = delimiter
        pfixed_prompt_text = prefix + prompt_text + ':'
        exemplar_template += pfixed_prompt_text + field_prefix + '{}'
        pfixed_prompt_texts.append(pfixed_prompt_text)
    exemplar_template += delimiter
    return exemplar_template, pfixed_prompt_texts

def construct_exemplars(
    examples,
    exemplar_template,
    default_value: Optional[str] = '',
    rstrip: Optional[bool] = False
):
    n_slots = len(re.findall('\{\}', exemplar_template))
    exemplars = []
    for example in examples:
        if isinstance(example, str):
            example = [example]
        exemplar = exemplar_template.format(*(example + [default_value] * (n_slots - len(example))))
        if rstrip:
            exemplar = exemplar.rstrip()
        exemplars.append(exemplar)
    return exemplars

def test_prompt(dataset):
    text_keys = sorted(list(filter(lambda key: key.startswith('text'), dataset.column_names['train'])))
    exemplar_template, _ = construct_exemplar_template(
        dataset.delimiter,
        dataset.prompt_texts
    )
    train_data = []
    for example in random.choices(dataset['train'], k=5):
        train_example = [example[text_key] for text_key in text_keys]
        train_example.append(example['label_text'])
        train_data.append(train_example)
    prompt = ''.join(construct_exemplars(train_data, exemplar_template))
    return prompt

def create_step(pattern):
    def step(src_state, value):
        if src_state[:2] == (0, 0): # init
            if value == pattern[0][0]:
                return (1, 0, 1)
            return (0, 0, 0)

        group_idx, idx, first_reach = src_state
        subpattern_idx = group_idx - 1
        if idx == len(pattern[subpattern_idx]) - 1: # 如果是该组的最后一个
            try: # 可能是最后一个组，此时subpattern_idx+1超界
                if value == pattern[subpattern_idx + 1][0]:
                    return (group_idx + 1, 0, 1)
            except:
                pass
            return (group_idx, idx, 0)
        else: # 如果是该组的中间
            if value == pattern[subpattern_idx][idx + 1]:
                return (group_idx, idx + 1, 1)
            try:
                return (group_idx - 1, len(pattern[subpattern_idx] - 1), 0)
            except:
                return (0, 0, 0)
    return step

def check_sequence(
    sequence,
    pattern,
    stop_value
):
    step = create_step(pattern)

    state = (0, 0, 1)
    match_span_ids = []
    success = False
    for i, value in enumerate(sequence):
        print(state)
        if value == stop_value:
            return None
        state = step(state, value)
        group_idx, idx, first_reach = state
        if group_idx > 0 and idx == len(pattern[group_idx - 1]) - 1 and first_reach:
            match_span_ids.append((i - len(pattern[group_idx - 1]) + 1, i + 1))

        if state[:2] == (len(pattern), len(pattern[-1]) - 1):
            success = True
            break
    if success:
        return match_span_ids
    return None

def macro_auroc(y_true, y_score):
    n_classes = y_score.shape[-1]
    auroc = 0
    for k, (i, j) in enumerate(itertools.combinations(range(n_classes), 2)):
        mask = (y_true == i) | (y_true == j)
        _y_true = y_true[mask]
        _y_score = y_score[mask]
        _auroc = roc_auc_score(
            y_true=(_y_true == i).long(),
            y_score=_y_score[:, i] - _y_score[:, j]
        )
        auroc = (k * auroc + _auroc) / (k + 1)
    return auroc

def evaluate(gold_labels, label_logprobs):
    results = dict()
    classification_results = classification_report(
        y_true=gold_labels, 
        y_pred=label_logprobs.argmax(-1),
        output_dict=True
    )
    results = {
        'accuracy': classification_results['accuracy'],
        'macro-precision': classification_results['macro avg']['precision'],
        'macro-recall': classification_results['macro avg']['recall'],
        'macro-f1': classification_results['macro avg']['f1-score'],
    }
    results['macro-auroc'] = macro_auroc(
        y_true=gold_labels,
        y_score=label_logprobs
    )
    return results

@torch.no_grad()
def unpack_past_key_values(past_key_values):
    batch_size = past_key_values[0][0].shape[0]
    n_layer = len(past_key_values)
    unpacked_past_key_values = []
    for k in range(batch_size):
        unpacked_past_key_values.append([])
        for i in range(n_layer):
            unpacked_past_key_values[k].append([None, None])
            for j in range(2):
                unpacked_past_key_values[k][i][j] = past_key_values[i][j][k].unsqueeze(0)
    return unpacked_past_key_values

@torch.no_grad()
def unpack_list_of_past_key_values(list_of_past_key_values):
    unpacked_past_key_values = []
    for past_key_values in list_of_past_key_values:
        unpacked_past_key_values.extend(unpack_past_key_values(past_key_values))
    return unpacked_past_key_values

@torch.no_grad()
def broadcast_past_key_values(past_key_values, batch_size, is_rnn):
    if past_key_values is None:
        return None
    if batch_size == 1:
        return past_key_values
    if is_rnn:
        # tuple of five torch.FloatTensor of shape (batch_size, hidden_size, num_hidden_layers)
        past_key_values = list(past_key_values)
        for i, item in enumerate(past_key_values):
            past_key_values[i] = item.repeat(batch_size, 1, 1)
    else:
        past_key_values = list(past_key_values)
        for i, item in enumerate(past_key_values):
            past_key_values[i] = list(item)
        n_layer = len(past_key_values)
        for i in range(n_layer):
            for j in range(2):
                past_key_values[i][j] = past_key_values[i][j].repeat(batch_size, 1, 1, 1)
    return past_key_values

@torch.no_grad()
def combine_list_of_past_key_values(list_of_past_key_values):
    res = []
    n_layer = len(list_of_past_key_values[0])
    for i in range(n_layer):
        res.append([])
        for j in range(2):
            res[i].append([])
            for past_key_values in list_of_past_key_values:
                res[i][j].append(past_key_values[i][j])
            res[i][j] = torch.cat(res[i][j], 0)
    return res

@torch.no_grad()
def prune_past_key_values(past_key_values, is_rnn, include_ids=None, exclude_ids=None, include_mask=None):
    if is_rnn:
        size = past_key_values[0].shape[0]
    else:
        size = past_key_values[0][0].shape[0]

    if include_mask is None:
        if include_ids is not None:
            include_mask = [True if i in include_ids else False for i in range(size)]
        else:
            include_mask = [False if i in exclude_ids else True for i in range(size)]
    past_key_values = list(past_key_values)

    if is_rnn:
        # tuple of five torch.FloatTensor of shape (batch_size, hidden_size, num_hidden_layers)
        for i, item in enumerate(past_key_values):
            past_key_values[i] = item[include_mask]

    else:
        for i, item in enumerate(past_key_values):
            past_key_values[i] = list(item)
        n_layer = len(past_key_values)
        for i in range(n_layer):
            for j in range(2):
                past_key_values[i][j] = past_key_values[i][j][include_mask]

    return past_key_values

def rnn2transformer_forward(
    self,
    input_ids=None, 
    past_key_values=None, 
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **kwargs
):
    '''This method overides the original `forward` of RWKV to make a consistent api.
    '''
    outputs = self.old_forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        state=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        labels=None,
        **kwargs
    )
    outputs.past_key_values = outputs.state
    return outputs

def dont_add_space_prefix___call__(self, *args, **kwargs):
    '''This method overides the orginal `__call__` of LlamaTokenizer.
    '''
    try:
        text = args[0]
    except IndexError:
        text = kwargs['text']
    return_tensors = kwargs.get('return_tensors', None)
    add_special_tokens = kwargs.get('add_special_tokens', True)
    outputs = self.old___call__(*args, **kwargs)
    spaces = {'\n', '\t', ' '}
    if isinstance(text, str):
        if text[0] in spaces and not add_special_tokens:
            if return_tensors is None:
                outputs.input_ids = outputs.input_ids[1:]
                outputs.attention_mask = outputs.attention_mask[1:]
            elif return_tensors == 'pt':
                outputs.input_ids = outputs.input_ids[:, 1:]
                outputs.attention_mask = outputs.attention_mask[:, 1:]
                
    elif isinstance(text, list):
        if text[0][0] in spaces and not add_special_tokens: # assume all the input are started with spaces
            if return_tensors is None:
                outputs.input_ids = list(map(lambda x: x[1:], outputs.input_ids))
                outputs.attention_mask = list(map(lambda x: x[1:], outputs.attention_mask))
            elif return_tensors == 'pt':
                outputs.input_ids = outputs.input_ids[:, 1:]
                outputs.attention_mask = outputs.attention_mask[:, 1:]

    return outputs