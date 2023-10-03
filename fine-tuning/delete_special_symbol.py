import json
import os
import jsonlines
import re
import spacy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bioasq')

args = parser.parse_args()

assert args.dataset in ['bioasq','squad','searchqa','hotpotqa','newsqa','naturalquestions','textbookqa','triviaqa']

base_dir = 'data/mrqa-few-shot'
nlp = spacy.load('en_core_web_md')
datasets_name = os.listdir(base_dir)

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

special_tokens = []
map_tokens = dict()

def index_substr(span, substr):
    start = 0
    try:
        start = span.index(substr)
    except:
        return -1
    return start

def get_title_or_alt(span):
    if '<img' in span:
        start = index_substr(span, 'alt=')
        if start == -1:
            return ''
        flag, span_start = False, 0
        for i in range(start, len(span)):
            if span[i] == '"':
                flag = True
                span_start = i
            if span[i] == '"' and flag == True:
                return span[span_start+1:i]
    elif '<a href=' in span:
        start = index_substr(span, 'title=')
        if start == -1:
            return ''
        flag, span_start = False, 0
        for i in range(start, len(span)):
            if span[i] == '"':
                flag = True
                span_start = i
            if span[i] == '"' and flag == True:
                return span[span_start+1:i]
            

with open('fine-tuning/special_span.txt', 'r', encoding='UTF-8') as f:
    for line in f:
        line = line.strip()
        if len(line) > 6 and line[:6] == '____: ':
            special_tokens.append(line[6:])
        elif len(line) > 13 and line[:13] == 'sssssssssss: ':
            if '<a href=' in line or '<img' in line:
                map_tokens[line[13:]] = get_title_or_alt(line[13:])


def check(keys, new_context):
    for key in keys:
        if key not in new_context:
            return False
    return True

def replace_special_tokens(context, answer_map):
    for token in ['[DOC]', '[TLE]', '[PAR]', '[SEP]']:
        # 以|代替之前特殊的token
        new_context = context.replace(token, '|')
        if check(answer_map.keys(), new_context):
            context = new_context
    for token in special_tokens:
        # 去除html标签
        new_context = context.replace(token, ' ')
        if check(answer_map.keys(), new_context):
            context = new_context
    for key in map_tokens.keys():
        if key in context:
            new_context = context.replace(key, map_tokens[key])
            if check(answer_map.keys(), new_context):
                context = new_context
    context = re.sub(r'(\| )+', '| ', context)
    context = context.strip('\n').strip(' ')
    return context

def make_new_dict(context, question, answer_map, answers, qid):
    context = re.sub(r'[ ]+', ' ', context)
    context = context.strip('\n').strip(' ')
    question = re.sub(r'[ ]+', ' ', question)
    question = question.strip('\n').strip(' ')
    answer_map['ANSWERANSWER'] = re.sub(r'[ ]+', ' ', answer_map['ANSWERANSWER'])
    answer_map['ANSWERANSWER'] = answer_map['ANSWERANSWER'].strip('\n').strip(' ')
    answer_doc = nlp(answer_map['ANSWERANSWER'])
    answer_tokens = []
    for token in answer_doc:
        answer_tokens.append([token.text, token.idx])
    new_data = dict()
    new_data['context'] = context
    new_data['qas'] = []
    new_data['context_tokens'] = []
    qa_detected_answers = dict()
    qa_detected_answers['text'] = answer_map['ANSWERANSWER']
    qa_detected_answers['token_spans'] = []
    qa_detected_answers['char_spans'] = []
    temp_context_tokens = []
    context_doc = nlp(context)
    for token in context_doc:
        temp_context_tokens.append([token.text, token.idx])
    answer_tokens_offset = 0
    for i, temp_data in enumerate(temp_context_tokens):
        text, id = temp_data[0], temp_data[1]
        if text == 'ANSWERANSWER':
            qa_detected_answers['token_spans'].append([len(new_data['context_tokens']), len(new_data['context_tokens'])+len(answer_tokens)-1])
            for answer_text, answer_id in answer_tokens:
                new_data['context_tokens'].append([answer_text, answer_id+answer_tokens_offset+id])
            qa_detected_answers['char_spans'].append([new_data['context_tokens'][-len(answer_tokens)][1], new_data['context_tokens'][-len(answer_tokens)][1] + len(answer_map['ANSWERANSWER']) - 1])
            if i + 1 < len(temp_context_tokens):
                answer_tokens_offset = new_data['context_tokens'][-1][1] + len(new_data['context_tokens'][-1][0]) + 1 - temp_context_tokens[i+1][1]
        else:
            new_data['context_tokens'].append([text, id+answer_tokens_offset])
    qa = dict()
    qa['question'] = question
    qa['answers'] = answers
    qa['qid'] = qid
    qa['question_tokens'] = []
    question_doc = nlp(question)
    for token in question_doc:
        qa['question_tokens'].append([token.text, token.idx])
    qa['detected_answers'] = []
    qa['detected_answers'].append(qa_detected_answers)
    new_data['qas'].append(qa)
    new_data['context'] = new_data['context'].replace('ANSWERANSWER', answer_map['ANSWERANSWER'])
    return new_data.copy()

for dataset_name in datasets_name:
    if args.dataset != dataset_name:
        continue
    files = os.listdir(os.path.join(base_dir, dataset_name))
    for file in files:
        if '_qass' not in file and '_delete' not in file:
            fin = jsonlines.open(os.path.join(base_dir, dataset_name, file[:-6]+'_delete_special_symbol.jsonl'), 'w')
            flag_ = True
            print(file)
            with open(os.path.join(base_dir, dataset_name, file), 'r', encoding='UTF-8') as f:
                # next(f)
                for line in f:
                    if flag_ == True:
                        fin.write(line)
                        flag_ = False
                        continue
                    data = json.loads(line.strip())
                    origin_context = data['context']
                    for i, qa in enumerate(data['qas']):
                        context = origin_context
                        question = qa['question']
                        answers = qa['answers']
                        qid= qa["id" if "id" in qa else "qid"]
                        detected_answer = qa['detected_answers'][0]
                        answer_text = detected_answer['text']
                        
                        # 把答案替换成ANSWERANSWER
                        new_context = ''
                        context_start = 0
                        detected_answer_char_spans = detected_answer['char_spans']
                        sorted_detected_answer_char_spans = sorted(detected_answer_char_spans, key=lambda x:x[0])
                        for start, end in sorted_detected_answer_char_spans:
                            new_context += context[context_start:start] + ' ANSWERANSWER '
                            context_start = end + 1
                        new_context += context[context_start:]
                        context = new_context
                        
                        answer_map = dict()
                        answer_map['ANSWERANSWER'] = answer_text
                        context = replace_special_tokens(context, answer_map)
                        new_data = make_new_dict(context, question, answer_map, answers, qid)
                        fin.write(new_data)
            fin.close()
