import copy
import json
import os
import re
import warnings
from collections import OrderedDict, UserDict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from packaging import version
from kiwipiepy import Kiwi
from transformers import BertTokenizer
from transformers import __version__
from transformers.dynamic_module_utils import custom_object_save
from transformers.utils import (
    EntryNotFoundError,
    ExplicitEnum,
    PaddingStrategy,
    PushToHubMixin,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    TensorType,
    add_end_docstrings,
    copy_func,
    get_file_from_repo,
    is_flax_available,
    is_offline_mode,
    is_remote_url,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    logging,
    to_py_obj,
    torch_required,
)

from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils.generic import _is_jax, _is_numpy, _is_tensorflow, _is_torch, _is_torch_device

VERY_LARGE_INTEGER = int(1e30)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(1e20)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]


# Slow tokenizers used to be saved in three separated files
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
FULL_TOKENIZER_FILE = "tokenizer.json"
_re_tokenizer_file = re.compile(r"tokenizer\.(.*)\.json")


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in
    an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"
class MyWordpieceTokenizer(object):
    def __init__(self, vocab, unk_token, mask_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.kiwi = Kiwi(num_workers=1, model_path=None, load_default_dict=True, integrate_allomorph=True, model_type='sbg', typos=None, typo_cost_threshold=2.5)


    def convert(self,test_keyword):
        BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

        # 초성 리스트. 00 ~ 18
        CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

        # 중성 리스트. 00 ~ 20
        JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

        # 종성 리스트. 00 ~ 27 + 1(1개 없음)
        JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        JONGSUNG_LIST = [' ', 0x11A8, 0x11A9, 0x11AA, 0x11AB, 0x11AC, 0x11AD, 0x11AE, 0x11AF, 0x11B0, 0x11B1, 0x11B2, 0x11B3, 0x11B4, 0x11B5, 0x11B6, 0x11B7, 0x11B8, 0x11B9, 0x11BA, 0x11BB, 0x11BC, 0x11BD, 0x11BE, 0x11BF, 0x11C0, 0x11C1, 0x11C2]

        split_keyword_list = list(test_keyword)
        #print(split_keyword_list)

        result = list()
        for keyword in split_keyword_list:
            # 한글 여부 check 후 분리
            if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
                char_code = ord(keyword) - BASE_CODE
                char1 = int(char_code / CHOSUNG)
                """
                if char1 >= len(CHOSUNG_LIST):
                    print(char1, " ", ord(keyword))
                """
                try:
                    result.append(CHOSUNG_LIST[char1])
                except:
                    #print(char1," ", CHOSUNG_LIST[-1], " ", keyword)
                    #print(test_keyword)
                    result.append(keyword)
                    continue
                #print('초성 : {}'.format(CHOSUNG_LIST[char1]))
                char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
                result.append(JUNGSUNG_LIST[char2])
                #print('중성 : {}'.format(JUNGSUNG_LIST[char2]))
                char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
                if char3==0:
                    result.append('⑨')
                else:
                    result.append(chr(JONGSUNG_LIST[char3]))
                #print('종성 : {}'.format(JONGSUNG_LIST[char3]))
            else:
                result.append(keyword)
        # result
        #print("".join(result))
        return("".join(result))
    def tokens(self, sentence):
        new_sentence = whitespace_tokenize(sentence)
        return_list = []
        for tokens in new_sentence:
            morph_tokens = self.kiwi.tokenize(tokens)
            #print(morph_tokens)
            if len(morph_tokens) < 0:
                continue
            elif len(morph_tokens) == 1:
                return_list.append(morph_tokens[0].form)
                return_list.append(" ")
            elif len(morph_tokens) == 2:
                return_list.append(morph_tokens[0].form)
                return_list.append(morph_tokens[1].form)
                return_list.append(" ")
            elif len(morph_tokens) >= 3:
                return_list.append(morph_tokens[0].form)
                for i in range(1,len(morph_tokens)-1):
                    return_list.append(morph_tokens[i].form)
                return_list.append(morph_tokens[-1].form)
                return_list.append(" ")
        return ''.join(return_list)
    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.
        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.
        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.
        Returns:
            A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            #print(token)
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            if ''.join(chars[0:len(chars)]) in self.vocab:
                output_tokens.append(''.join(chars[0:len(chars)]))
                #print(''.join(chars[0:len(chars)]))
            else:   
                is_bad = False
                start = 0
                sub_tokens = []
                while start < len(chars):
                    end = len(chars)
                    cur_substr = None
                    if "##" + ''.join(chars[start:end]) in self.vocab:
                        #print(substr)
                        cur_substr = "##"+''.join(chars[start:end])
                        #output_tokens.append(''.join(chars[start:end]))
                    else:
                        while start < end:
                            if start == 0 and end == len(chars):
                                substr = "".join(chars[start:end])
                            else:    
                                substr = "".join(chars[start:end])
                            #print(substr)
                            if start > 0:
                                substr = "##" + substr
                            if substr in self.vocab:
                                cur_substr = substr
                                #if end == len(chars):
                                #    cur_substr = substr[:-1]
                                #else:
                                #    cur_substr = substr
                                
                                break
                            end -= 1
                    if cur_substr is None:
                        last_substr = "".join(chars[start:len(chars)])
                        if last_substr in self.vocab:
                            cur_substr = last_substr
                        else:
                            cur_substr = self.unk_token
                        sub_tokens.append(cur_substr)
                        break
                      
                    sub_tokens.append(cur_substr)
                    start = end
                output_tokens.extend(sub_tokens)
        return output_tokens
    

class KrongBertTokenizer(BertTokenizer):
    def setMorph(self,path):
        self.morph = {}
        #morph -> A+B+C(or num),1,300,400,500
        f = open(path+os.sep+'morph.txt')
        for i in f.readlines():
            temp = i.replace("\n", "").replace(" ","").split(',')
            #print(temp)
            self.morph[str(temp[0])] = str(temp[1])
        print(self.morph['2'])
    def batch_encode_plus(self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    )-> BatchEncoding:
        
     
        """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.
        <Tip warning={true}>
        This method is deprecated, `__call__` should be used instead.
        </Tip>
        Args:
            batch_text_or_text_pairs (`List[str]`, `List[Tuple[str, str]]`, `List[List[str]]`, `List[Tuple[List[str], List[str]]]`, and for not-fast tokenizers, also `List[List[int]]`, `List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded. This can be a list of
                string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
                details in `encode_plus`).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )
        temp_return = self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )
        #print(temp_return)
        #print(type(temp_return))
        #BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)
        tokens = temp_return.data
        encoding = temp_return._encodings
        
        if self.morph != None:
            new_token_type_ids = []
            seg_ids = tokens['token_type_ids'].copy()
            for i in tokens['input_ids']:
                temp_type_ids = []
                for j in i:
                    temp_type_ids.append(int(self.morph[str(j)]))
                new_token_type_ids.append(temp_type_ids)
            tokens['token_type_ids']=new_token_type_ids
            tokens['seg_ids'] = seg_ids

        #print(temp_return)
        return BatchEncoding(tokens, encoding, tensor_type=return_tensors)
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.
        <Tip warning={true}>
        This method is deprecated, `__call__` should be used instead.
        </Tip>
        Args:
            text (`str`, `List[str]` or `List[int]` (the latter only for not-fast tokenizers)):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        temp_return =  self._encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )
        tokens = temp_return.data
        encoding = temp_return._encodings
        
        if self.morph != None:
            new_token_type_ids = []
            temp_type_ids = []
            seg_ids = tokens['token_type_ids'].copy()
            #print(tokens['input_ids'])
            for j in tokens['input_ids']:
                temp_type_ids.append(int(self.morph[str(j)]))
            new_token_type_ids+=temp_type_ids
            tokens['token_type_ids']=new_token_type_ids
            tokens['seg_ids'] = seg_ids
        return BatchEncoding(tokens, encoding, tensor_type=return_tensors)


if __name__ == '__main__':
    tokenizer= KrongBertTokenizer.from_pretrained('ourbert_new_new',strip_accents=False,  lowercase=False)
    tokenizer.setMorph('ourbert_new_new')
    
    print(tokenizer.batch_encode_plus([['안녕하세요.',' 반갑습니다']], padding='max_length'))
    print(tokenizer.encode_plus('안녕하세요.',' 반갑습니다', padding='max_length'))

