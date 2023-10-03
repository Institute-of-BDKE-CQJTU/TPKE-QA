import os
import json
import argparse
import spacy
nlp = spacy.load('en_core_web_md')

nlp.add_pipe("entityLinker", last=True)

base_dir = 'data/mrqa-few-shot'

datasets_name = os.listdir(base_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bioasq')

args = parser.parse_args()

assert args.dataset in ['bioasq','squad','searchqa','hotpotqa','newsqa','naturalquestions','textbookqa','triviaqa']

question_qid_to_entity = dict()
context_qid_to_entity = dict()

for dataset_name in datasets_name:
    if args.dataset != dataset_name:
        continue
    files = os.listdir(os.path.join(base_dir, dataset_name))
    for file in files:
        if 'kg' not in file and '_qass' not in file and '_align' not in file and '_delete' not in file:
            with open(os.path.join(base_dir, dataset_name, file), 'r', encoding='UTF-8') as f:
                next(f)
                for line in f:
                    data = json.loads(line.strip())
                    context = data['context']
                    context_doc = nlp(context)
                    all_linked_entities = context_doc._.linkedEntities
                    context_entity_ids = []
                    for entity in all_linked_entities:
                        context_entity_ids.append('Q'+str(entity.get_id())+':'+str(entity.get_label()))
                    for qa in data['qas']:
                        question = qa['question']
                        qid= qa["id" if "id" in qa else "qid"]
                        context_qid_to_entity[qid] = context_entity_ids
                        question_doc = nlp(question)
                        all_linked_entities = question_doc._.linkedEntities
                        question_qid_to_entity[qid] = []
                        for entity in all_linked_entities:
                            question_qid_to_entity[qid].append('Q'+str(entity.get_id())+':'+str(entity.get_label()))

    fin = open(os.path.join('fine-tuning/spacy_entity', '{}_context.txt'.format(args.dataset)), 'w', encoding='UTF-8')
    for qid in context_qid_to_entity:
        fin.write(qid+'\t')
        for entity_id in context_qid_to_entity[qid]:
            fin.write(entity_id+'\t')
        fin.write('\n')
    fin = open(os.path.join('fine-tuning/spacy_entity', '{}_question.txt'.format(args.dataset)), 'w', encoding='UTF-8')
    for qid in question_qid_to_entity:
        fin.write(qid+'\t')
        for entity_id in question_qid_to_entity[qid]:
            fin.write(entity_id+'\t')
        fin.write('\n')
        
    