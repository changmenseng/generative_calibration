import tqdm
import json
import scipy
import math
import torch
import argparse
import itertools
import functools
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
from types import MethodType
from typing import List, Set, Optional, Union, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils import (
    construct_exemplar_template, 
    construct_exemplars,
    create_step,
    broadcast_past_key_values,
    prune_past_key_values,
    rnn2transformer_forward,
    dont_add_space_prefix___call__
)

class InContextLearner:

    def __init__(self, model_name):
        metadata = json.load(open('llm_metadata.json', 'r'))[model_name]
        self.model = AutoModelForCausalLM.from_pretrained(
            **metadata,
            torch_dtype=torch.float16, 
            load_in_8bit=True, 
            low_cpu_mem_usage=True
        )
        self.model.eval()
        self.is_rnn = 'rwkv' in model_name
        if self.is_rnn:
            self.model.old_forward = self.model.forward
            self.model.forward = MethodType(rnn2transformer_forward, self.model)

        self.is_llama = 'llama' in model_name
        if self.is_llama:
            from transformers import LlamaTokenizer
            LlamaTokenizer.old___call__ = LlamaTokenizer.__call__
            LlamaTokenizer.__call__ = dont_add_space_prefix___call__
            self.tokenizer = LlamaTokenizer.from_pretrained(
                metadata['pretrained_model_name_or_path'], 
                use_fast=True
            )
            self.tokenizer.pad_token = '-'
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                metadata['pretrained_model_name_or_path'], 
                use_fast=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.max_length = self.model.config.n_positions
        except:
            self.max_length = self.model.config.max_position_embeddings

        self.token_prefix = ' '

    def batch(
        self,
        inputs,
        batch_size,
        verbose,
    ):
        start = 0
        if verbose: pbar = tqdm.tqdm(total=len(inputs))
        while start < len(inputs):
            end = start + batch_size
            batch = inputs[start: end]
            start = end
            if verbose: pbar.update(len(batch))
            yield batch
        if verbose: pbar.close()
    
    @torch.no_grad()
    def _get_sent_logprobs(
        self,
        logits,
        input_ids,
        attention_mask = None
    ):
        token_logprobs = logits.float().log_softmax(-1)
        token_logprobs = torch.gather(token_logprobs[:,:-1], 2, input_ids.unsqueeze(-1)[:,1:]).squeeze(-1)
        if attention_mask is None:
            return token_logprobs.sum(-1)
        return (token_logprobs * attention_mask[:,1:]).sum(-1)
    
    @torch.no_grad()
    def _get_last_token_logprobs(
        self,
        logits,
        attention_mask = None
    ):
        token_logprobs = logits.float().log_softmax(-1)
        if attention_mask is None:
            return token_logprobs[:, -1]
        lens = attention_mask.sum(-1, keepdims=True)
        last_token_logprobs = torch.gather(
            token_logprobs, 1, (lens-1)[:,:,None].repeat(1,1,token_logprobs.shape[-1])).squeeze(1)
        return last_token_logprobs
    
    @torch.no_grad()
    def _sample_until_one_end(
        self,
        start_logits,
        past_key_values,
        topk: int,
        max_new_tokens: int,
        forced_sequences: List[List[int]],
        bad_word_ids: List[int],
        states: List[tuple],
        step: Callable[[tuple, int], tuple]
    ):
        
        first_tokens = {sequence[0] for sequence in forced_sequences}
        size = start_logits.shape[0]
        outputs = argparse.Namespace(
            logits=start_logits,
            past_key_values=past_key_values
        )
        end_ids = []
        sampled_ids = []
        for i in range(max_new_tokens):
            next_token_logits = outputs.logits[:, -1, :] # (size, vocab_size)
            for bad_word_id in bad_word_ids:
                next_token_logits[:, bad_word_id] = - torch.inf
            next_token_ids = []
            for j in range(size):
                sequence_idx, sub_sequence_idx, first_reach = states[j]
                if first_reach and \
                    sequence_idx == len(forced_sequences) and \
                    sub_sequence_idx == len(forced_sequences[-1]) - 1:
                    next_token_id = self.tokenizer.eos_token_id
                    end_ids.append(j)
                elif sequence_idx and first_reach and \
                    sub_sequence_idx != len(forced_sequences[sequence_idx-1])-1:
                    next_token_id = forced_sequences[sequence_idx-1][sub_sequence_idx+1]
                else:
                    if topk is None:
                        next_token_id = dist.Categorical(logits=next_token_logits[j].float()).sample().item()
                    else:
                        topk_next_token_logits, topk_ids = next_token_logits[j].float().topk(topk)
                        next_token_id = topk_ids[dist.Categorical(logits=topk_next_token_logits).sample().item()].item()
                    if next_token_id in first_tokens:
                        next_token_id = forced_sequences[sequence_idx][0]
                states[j] = step(states[j], next_token_id)
                next_token_ids.append(next_token_id)
            next_token_ids = torch.tensor(next_token_ids) # (n_sample)
            sampled_ids.append(next_token_ids)
            if end_ids:
                break
            
            outputs = self.model(
                input_ids=next_token_ids.unsqueeze(-1).cuda(0),
                past_key_values=outputs.past_key_values
            )
            outputs.logits = outputs.logits.cpu().float()

        if sampled_ids:
            sampled_ids = torch.stack(sampled_ids, -1)
        return sampled_ids, outputs.logits, outputs.past_key_values, states, end_ids

    @torch.no_grad()
    def _get_sampled_label_lm_logprobs(
        self,
        label_tokens: List[str],
        input_ids: Optional[torch.tensor] = None, # (1, length)
        past_key_values: Optional[tuple] = None,
        start_logits: Optional[torch.tensor] = None,
        topk: Optional[int] = None,
        max_new_tokens: Optional[int] = 384,
        forced_sequences: Optional[List[List[int]]] = None,
        bad_word_ids: Optional[List[int]] = None
    ):
        size = start_logits.shape[0]
        step = create_step(forced_sequences)
        _states = [(0, 0, 1) for _ in range(size)]

        _logits = start_logits
        _past_key_values = past_key_values
        
        ids_mapping = list(range(size)) # map local ids to global ids, the length should be unfinished
        sampled_ids = [[] for _ in range(size)]
        label_logprobs = [[] for _ in range(size)]

        current_length = 0
        while ids_mapping:
            _sampled_ids, _logits, _past_key_values, _states, _finished_ids = self._sample_until_one_end(
                start_logits=_logits,
                past_key_values=_past_key_values,
                topk=topk,
                max_new_tokens=max(max_new_tokens - current_length, 0),
                forced_sequences=forced_sequences,
                bad_word_ids=bad_word_ids,
                states=_states,
                step=step
            )
            _unfinished_ids = sorted(list(set(range(_logits.shape[0])) - set(_finished_ids))) 
            
            # add _sampled_ids to sampled_ids
            if len(_sampled_ids) != 0:
                current_length += _sampled_ids.shape[1]
                for _idx, idx in enumerate(ids_mapping):
                    sampled_ids[idx].append(_sampled_ids[_idx])
                
            if len(_finished_ids) == 0: # 如果_finished_ids为空，说明到最大长度但也没生成完
                for _unfinished_id in _unfinished_ids:
                    unfinished_id = ids_mapping[_unfinished_id]
                    sampled_ids[unfinished_id] = torch.cat(sampled_ids[unfinished_id], 0)
                    label_logprobs[unfinished_id] = torch.full((len(label_tokens),), torch.nan)
                break
            
            # add label_logprobs and concat sampled_ids for finished
            finished_ids = []
            for _finished_id in _finished_ids:
                finished_id = ids_mapping[_finished_id]
                finished_ids.append(finished_id)
                _logprobs = _logits.float().log_softmax(-1)
                for token in label_tokens:
                    token_id = self.tokenizer(self.token_prefix + token, add_special_tokens=False).input_ids[0]
                    label_logprobs[finished_id].append(_logprobs[_finished_id, 0, token_id])
                label_logprobs[finished_id] = torch.stack(label_logprobs[finished_id], 0)
                # sampled_ids[finished_id].append(torch.full((_sampled_ids.shape[0],), self.tokenizer.pad_token_id))
                sampled_ids[finished_id] = torch.cat(sampled_ids[finished_id], 0)

            
            # delete finished ones in _logits, _past_key_values, and _states
            _logits = _logits[_unfinished_ids]
            _past_key_values = prune_past_key_values(_past_key_values, self.is_rnn, include_ids=_unfinished_ids)
            _states = [_states[i] for i in _unfinished_ids]

            # update ids_mapping:
            for finished_id in finished_ids:
                ids_mapping.remove(finished_id)

        label_logprobs = torch.stack(label_logprobs, 0)
        return sampled_ids, label_logprobs

    @torch.no_grad()
    def generative_calibrate_predict(
        self,
        train_data: List[List[str]], 
        test_data: Union[List[str], List[List[str]]],
        label_tokens: List[str],
        prefix: Optional[str] = '',
        delimiter: Optional[str] = '\n',
        prompt_texts: Optional[List[str]] = ['Input', 'Output'],
        n_samples: Optional[int] = 100,
        n_samples_once: Optional[int] = 5,
        seed: Optional[float] = None,
        topk: Optional[int] = None,
        max_new_tokens: Optional[int] = 384,
        obj_label_logmarginal: Optional[torch.tensor] = None,
        ori_label_logprobs: Optional[torch.tensor] = None,
        batch_size: Optional[int] = 2,
        verbose: Optional[bool] = False
    ):
        '''Generative calibration
        '''
        if obj_label_logmarginal is None:
            obj_label_logmarginal = torch.full((len(label_tokens),), -math.log(len(label_tokens)))

        exemplar_template, pfixed_prompt_texts = construct_exemplar_template(
            delimiter=delimiter,
            prompt_texts=prompt_texts,
            field_prefix=self.token_prefix
        )
        exemplar_prompt = prefix + ''.join(construct_exemplars(train_data, exemplar_template))
        if ori_label_logprobs is None:
            test_prompts = construct_exemplars(test_data, exemplar_template, default_value='', rstrip=True)
            ori_label_logprobs = self._predict(
                test_prompts=test_prompts,
                exemplar_prompt=exemplar_prompt,
                label_tokens=label_tokens,
                batch_size=batch_size,
                verbose=verbose
            ).log_softmax(-1)

        generation_prompt = exemplar_prompt + pfixed_prompt_texts[0]
        generation_input_ids = self.tokenizer(generation_prompt, return_tensors='pt').input_ids.cuda(0)
        outputs = self.model(generation_input_ids)
        prompt_length = generation_input_ids.shape[1]
        pattern = self.tokenizer(pfixed_prompt_texts[1:], add_special_tokens=False).input_ids
        if self.is_llama:
            pattern = list(map(lambda x: x[1:], pattern))
        
        cnt = 0
        sampled_sequences = []
        sampled_label_lm_logprobs = []
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        prev_n_samples = 0
        bad_word_ids = [self.tokenizer.eos_token_id]
        double_delimiter_ids = self.tokenizer(delimiter*2, add_special_tokens=False).input_ids
        if len(double_delimiter_ids) == 1:
            bad_word_ids.append(double_delimiter_ids[0])
        if verbose:
            pbar = tqdm.tqdm(total=n_samples, desc='Sampling')
        while cnt < n_samples:
            curr_n_samples = min(n_samples_once, n_samples-cnt)
            if curr_n_samples != prev_n_samples or self.is_rnn:
                past_key_values = broadcast_past_key_values(outputs.past_key_values, curr_n_samples, self.is_rnn)
                start_logits = outputs.logits[:, -1:, :].repeat(curr_n_samples, 1, 1)
            
            batch_sampled_sequences, batch_sampled_label_lm_logprobs = self._get_sampled_label_lm_logprobs(
                label_tokens=label_tokens,
                past_key_values=past_key_values,
                start_logits=start_logits,
                topk=topk,
                max_new_tokens=min(max_new_tokens, self.max_length - prompt_length),
                forced_sequences=pattern,
                bad_word_ids=bad_word_ids
            )
            success_mask = ~batch_sampled_label_lm_logprobs.isnan().any(-1)
            batch_sampled_label_lm_logprobs = batch_sampled_label_lm_logprobs[success_mask]
            batch_sampled_sequences = [batch_sampled_sequences[i] for i in success_mask.argwhere().squeeze(-1).tolist()]

            cnt += success_mask.long().sum().item()
            sampled_label_lm_logprobs.append(batch_sampled_label_lm_logprobs)
            sampled_sequences.extend(batch_sampled_sequences)

            prev_n_samples = curr_n_samples
            if verbose:
                pbar.update(success_mask.long().sum().item())
        if verbose:
            pbar.close()

        sampled_label_lm_logprobs = torch.cat(sampled_label_lm_logprobs, 0)
        estimated_label_logmarginal = (sampled_label_lm_logprobs.logsumexp(0) - math.log(n_samples)).log_softmax(-1)

        caled_label_logprobs = (ori_label_logprobs + obj_label_logmarginal - estimated_label_logmarginal).log_softmax(-1)

        outputs = {
            'ori_label_logprobs': ori_label_logprobs,
            'caled_label_logprobs': caled_label_logprobs,
            'estimated_label_logmarginal': estimated_label_logmarginal,
            'sampled_sequences': sampled_sequences
        }
        return outputs

    @torch.no_grad()
    def _predict(
        self,
        test_prompts: List[str],
        label_tokens: List[str],
        exemplar_prompt: Optional[str] = '',
        batch_size: Optional[int] = 2,
        verbose: Optional[bool] = False
    ):
        past_key_values = None
        exemplar_prompt_length = 0
        if exemplar_prompt:
            exemplar_prompt_ids = self.tokenizer(
                exemplar_prompt, return_tensors='pt', truncation=True, max_length=self.max_length-128).input_ids
            past_key_values = self.model(
                input_ids=exemplar_prompt_ids.cuda(0)
            ).past_key_values
            exemplar_prompt_length = exemplar_prompt_ids.numel()

        label_logprobs = []
        prev_batch_size = 0
        for batch in self.batch(test_prompts, batch_size, verbose):
            batch_size = len(batch)
            if batch_size != prev_batch_size or self.is_rnn:
                broadcasted_past_key_values = broadcast_past_key_values(past_key_values, batch_size, self.is_rnn)
            batch_encoded_inputs = self.tokenizer(
                batch, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                add_special_tokens=False,
                max_length=min(self.max_length - exemplar_prompt_length, 384))
            batch_logits = self.model(
                input_ids=batch_encoded_inputs.input_ids.cuda(0),
                past_key_values=broadcasted_past_key_values
            ).logits.float().cpu()
            batch_last_token_logprobs = self._get_last_token_logprobs(
                logits=batch_logits,
                attention_mask=batch_encoded_inputs.attention_mask
            ).log_softmax(-1)
            batch_label_logprobs = []
            for token in label_tokens:
                token_id = self.tokenizer(self.token_prefix + token, add_special_tokens=False).input_ids[0]
                batch_label_logprobs.append(batch_last_token_logprobs[:, token_id])
            batch_label_logprobs = torch.stack(batch_label_logprobs, -1)
            label_logprobs.append(batch_label_logprobs)
            prev_batch_size = batch_size
        label_logprobs = torch.cat(label_logprobs, 0)
        
        return label_logprobs

    @torch.no_grad()
    def predict(
        self,
        train_data: List[List[str]], 
        test_data: Union[List[str], List[List[str]]],
        label_tokens: List[str],
        prefix: Optional[str] = '',
        delimiter: Optional[str] = '\n',
        prompt_texts: Optional[List[str]] = ['Input', 'Output'],
        batch_size: Optional[int] = 2,
        verbose: Optional[bool] = False
    ):
        '''Regular In-context Learning
        '''

        exemplar_template, _ = construct_exemplar_template(
            delimiter=delimiter,
            prompt_texts=prompt_texts,
            field_prefix=self.token_prefix
        )
        exemplar_prompt = prefix + ''.join(construct_exemplars(train_data, exemplar_template))
        
        test_prompts = construct_exemplars(test_data, exemplar_template, default_value='', rstrip=True)
        label_logprobs = self._predict(
            test_prompts=test_prompts,
            label_tokens=label_tokens,
            exemplar_prompt=exemplar_prompt,
            batch_size=batch_size,
            verbose=verbose
        ).log_softmax(-1)

        return {'label_logprobs': label_logprobs}