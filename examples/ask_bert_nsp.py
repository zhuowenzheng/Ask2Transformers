""" Script para hacer classification de temas preguntandole directamente a Roberta.

Método de uso:
    (Mediante un fichero como input) python3 get_topics.py topics.txt input_file.txt
    (Mediante el input standard)     python3 get_topics.py topics.txt 
                                     cat input_file.txt | python3 get_topics.py topics.txt

El topics.txt debe contener los diferentes topics que se van a usar y deberan estar separados por saltos de linea.
"""
from transformers import BertForNextSentencePrediction, AutoTokenizer
import torch
import sys
from pprint import pprint

if len(sys.argv) < 2:
    print('Usage:\tpython3 get_topics.py topics.txt input_file.txt\n\tpython3 get_topics.py topics.txt < input_file.txt')
    exit(1)

model = BertForNextSentencePrediction.from_pretrained('bert-large-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')

model.eval()

def get_topic(context, topics, tokenizer=tokenizer, model=model):
    with torch.no_grad():
        sentences = [f"{context} {tokenizer.sep_token} Topic or domain about \"{topic}\"." for topic in topics]
        input_ids = tokenizer.batch_encode_plus(sentences, pad_to_max_length=True)
        outputs = model(torch.tensor(input_ids['input_ids']))[0]
        output_probs = torch.softmax(model(torch.tensor(input_ids['input_ids']))[0][:,0], 0).numpy()
        output = sorted(list(zip(output_probs, topics)), reverse=True)
    return output

with open(sys.argv[1], 'rt') as f:
    topics = [topic.rstrip() for topic in f]

input_stream = open(sys.argv[2], 'rt') if len(sys.argv) == 3 else sys.stdin

for line in input_stream:
    line = line.rstrip()
    topic_dist = get_topic(line, topics)
    print(line)
    pprint(topic_dist)
    print()