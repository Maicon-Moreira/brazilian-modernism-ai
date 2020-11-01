import os
import tensorflow_datasets as tfds
import pickle
import json


# get text
text_lines = []

for book in os.listdir('./books/'):
    with open(f'./books/{book}', encoding='iso 8859-1') as f:
        text_lines += f.readlines()

text_lines = [a.lower() for a in text_lines]

filtered_text_lines = []

for line in text_lines:
    if line != '\n':
        spaces = line.count(' ')
        total = len(line)

        empty_space = spaces/total

        # if more than half of the line is empty, ignore it
        if empty_space >= 0.5:
            pass

        else:
            filtered_text_lines.append(line)

with open('./text.txt', 'w') as f:
    f.writelines(filtered_text_lines)


# tokenizer
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    filtered_text_lines, target_vocab_size=2**13)


# save tokenizer
with open('tokenizer', 'wb') as f:
    f.write(pickle.dumps(tokenizer))


# save vocabulary
with open('vocabulary.json', 'w') as f:
    f.write(json.dumps(tokenizer.subwords))


# lines to a huge string
text = ''.join(filtered_text_lines)


# text tokens
tokens = tokenizer.encode(text)


# save tokens
with open('tokens.json', 'w') as f:
    f.write(json.dumps(tokens))