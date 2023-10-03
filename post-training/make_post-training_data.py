import spacy
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./path_of_splinter_model')

nlp = spacy.load('en_core_web_sm')

import json
import re
import pickle
import os
import random
import argparse
import copy

sentence_to_doc = dict()

STOPWORDS = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
             'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
             'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
             'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'also', 'could', 'would']

answerable_data = []
unanswerable_data = []


def detect_data(sentence:str):
    start_sent_tokens = tokenizer.tokenize(sentence)
    start_sent_doc = nlp(sentence)
    noun_spans = [chunk.text for chunk in start_sent_doc.noun_chunks]
    for ent in start_sent_doc.ents:
        noun_spans.append(ent.text)
    noun_spans = list(set(noun_spans))
    temp_dict = dict()
    for noun_span in noun_spans:
        if noun_span in STOPWORDS:
            continue
        span_tokens = tokenizer.tokenize(noun_span)
        if len(span_tokens) == 0:
            continue
        flag = 0
        for i in range(len(start_sent_tokens)):
            if start_sent_tokens[i] == span_tokens[0] and i + len(span_tokens) < len(start_sent_tokens):
                flag = 1
                for j in range(i+1, i+len(span_tokens)):
                    if start_sent_tokens[j] != span_tokens[j-i]:
                        flag = 0
                        break
            if flag == 1:
                temp_dict[noun_span] = [i, i+len(span_tokens)-1]
                break
    for key in temp_dict.keys():
        span_tokens = tokenizer.tokenize(key)
        flag = 0
        for i in range(len(sentence_to_doc[sentence])):
            if span_tokens[0] == sentence_to_doc[sentence][i] and i + len(span_tokens) < len(sentence_to_doc[sentence]):
                flag = 1
                for j in range(i+1, i+len(span_tokens)):
                    if span_tokens[j-i] != sentence_to_doc[sentence][j]:
                        flag = 0
                        break
            if flag == 1:
                temp_dict[key].extend([i, i+len(span_tokens)-1])
                break
    answerable_data_number, answerable_data_number = 0, 0
    for key in temp_dict.keys():
        if len(temp_dict[key]) == 4:
            __flag = 0
            question_start, question_end, answer_start, answer_end = temp_dict[key]
            if question_end + 1 == len(start_sent_tokens) or (question_end + 2 == len(start_sent_tokens) and start_sent_tokens[question_end+1] in [',', '.']):
                __flag = 1
            if __flag == 0:
                continue
            if len(start_sent_tokens) > 30:
                continue
            input_tokens = ['[CLS]'] + start_sent_tokens[:question_start] + ['[QUESTION]', '.', '[SEP]']
            token_type_ids = [0]*len(input_tokens)
            if answer_end + len(start_sent_tokens) + 2 >= 384:
                doc_end = answer_end + random.randint(1, 100)
                doc_start = doc_end - (384 - 2 - len(start_sent_tokens))
                answer_end = answer_end - doc_start
                answer_start = answer_start - doc_start
                sentence_to_doc_sentence = copy.deepcopy(sentence_to_doc[sentence][doc_start:doc_end])
            else:
                sentence_to_doc_sentence = copy.deepcopy(sentence_to_doc[sentence])
            input_tokens +=  sentence_to_doc_sentence[:383-len(input_tokens)] + ['[SEP]']
            attention_mask = [1]*len(input_tokens)
            token_type_ids.extend([1]*(len(input_tokens) - len(token_type_ids)))
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            for _ in range(384 - len(input_tokens)):
                input_ids.append(0)
                attention_mask.append(0)
                token_type_ids.append(0)
            if len(input_ids) == 384:
                final_answer_start = answer_start+4+len(start_sent_tokens[:question_start])
                final_answer_end = answer_end+4+len(start_sent_tokens[:question_start])
                answer_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(key))
                if input_ids[final_answer_start:final_answer_end+1] == answer_token_ids:
                    answerable_data.append([input_ids, attention_mask, token_type_ids, final_answer_start, final_answer_end])
                    answerable_data_number += 1
        elif len(temp_dict[key]) == 2:
            __flag = 0
            question_start, question_end = temp_dict[key]
            if question_end + 1 == len(start_sent_tokens) or (question_end + 2 == len(start_sent_tokens) and start_sent_tokens[question_end+1] in [',', '.']):
                __flag = 1
            if __flag == 0:
                continue
            if len(start_sent_tokens) > 25:
                short_start_sent_tokens = start_sent_tokens[-25:]
                sub_length = len(start_sent_tokens) - len(short_start_sent_tokens)
                question_start, question_end = question_start - sub_length, question_end - sub_length
                start_sent_tokens = short_start_sent_tokens
            input_tokens = ['[CLS]'] + start_sent_tokens[:question_start] + ['[QUESTION]', '.', '[SEP]']
            token_type_ids = [0]*len(input_tokens)
            input_tokens +=  sentence_to_doc[sentence][:383-len(input_tokens)] + ['[SEP]']
            attention_mask = [1]*len(input_tokens)
            token_type_ids.extend([1]*(len(input_tokens) - len(token_type_ids)))
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            for _ in range(384 - len(input_tokens)):
                input_ids.append(0)
                attention_mask.append(0)
                token_type_ids.append(0)
            if len(input_ids) == 384:
                unanswerable_data.append([input_ids, attention_mask, token_type_ids, 0, 0])
                answerable_data_number += 1
    return answerable_data_number, answerable_data_number
        
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ccnews')

args = parser.parse_args()
base_dataset_dir = './path_of_your_unlabeled_data'
dataset_map = {'ccnews':'ccnews_text.txt',
               'openwebtext':'openwebtext.txt',}
assert args.dataset in dataset_map.keys()
answerable_data_total_number, unanswerable_data_total_number = 0, 0
answerable_file_number, unanswerable_file_number = 1, 1

os.makedirs(os.path.join('pretraining_data/unanswerable', args.dataset), exist_ok=True)
os.makedirs(os.path.join('pretraining_data/answerable', args.dataset), exist_ok=True)

with open(os.path.join(base_dataset_dir, dataset_map[args.dataset]), 'r', encoding='UTF-8') as f:
    for line in f:
        line = line.strip()
        doc = nlp(line)
        sentence_to_doc.clear()
        for sent in doc.sents:
            sentence_to_doc[sent.text] = tokenizer.tokenize(line[sent.end_char:].strip(' '))
            if len(sentence_to_doc[sent.text]) < 300:
                break
            answerable_data_number, answerable_data_number = detect_data(sent.text)
            answerable_data_total_number += answerable_data_number
            unanswerable_data_total_number += answerable_data_number
        print('answerable_data_total_number:{0}, unanswerable_data_total_number:{1}'.format(answerable_data_total_number, unanswerable_data_total_number))
        if answerable_data_total_number // 192000 == answerable_file_number:
            with open('pretraining_data/answerable/{0}/pretrain_data_{1}.pkl'.format(args.dataset, str(answerable_file_number)), 'wb') as f:
                pickle.dump(answerable_data[:192000], f)
            answerable_file_number += 1
            temp = answerable_data[192000:]
            answerable_data.clear()
            answerable_data = temp
        if unanswerable_data_total_number // 192000 == unanswerable_file_number:
            with open('pretraining_data/unanswerable/{0}/pretrain_data_{1}.pkl'.format(args.dataset, str(unanswerable_file_number)), 'wb') as f:
                pickle.dump(unanswerable_data[:192000], f)
            unanswerable_file_number += 1
            temp = unanswerable_data[192000:]
            unanswerable_data.clear()
            unanswerable_data = temp

with open('pretraining_data/answerable/{0}/pretrain_data_{1}.pkl'.format(args.dataset, str(answerable_file_number)), 'wb') as f:
    pickle.dump(answerable_data, f)
answerable_data.clear()

with open('pretraining_data/unanswerable/{0}/pretrain_data_{1}.pkl'.format(args.dataset, str(unanswerable_file_number)), 'wb') as f:
    pickle.dump(unanswerable_data, f)
unanswerable_data.clear()
