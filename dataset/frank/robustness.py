import json
import random
from transformers import pipeline
import nltk
from nltk.corpus import wordnet
# Load the original FRANK dataset

frank_data=[]
for line in open('frank-data-sample-10.json', 'r'):
    line = json.loads(line)  # This should be a list of dictionary records
    frank_data.append(line)

# Ensure you have the WordNet data
nltk.download('wordnet')
nltk.download('punkt')

def replace_with_synonyms(text):
    words = nltk.word_tokenize(text)
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        # Choose a synonym randomly if it exists, else keep the word
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_words.append(synonym)
        else:
            new_words.append(word)
    return ' '.join(new_words)

# Load a paraphrasing model
paraphrase_model = pipeline("text2text-generation", model="t5-base")

def paraphrase_text(text):
    paraphrased = paraphrase_model(text, max_length=100, num_return_sequences=1)[0]['generated_text']
    return paraphrased

def introduce_typos(text, typo_rate=0.1):
    words = list(text)
    num_typos = int(len(words) * typo_rate)
    for _ in range(num_typos):
        idx = random.randint(0, len(words) - 1)
        words[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
    return ''.join(words)

def perturb_dataset(data, perturb_func, output_file):
    with open(output_file, 'w') as f:
        for record in data:
            new_record = record.copy()  # Avoid modifying the original
            new_record['sentences'] = [perturb_func(sentence) for sentence in record['sentences']]
            
            # Write each JSON object in a single line
            json_line = json.dumps(new_record)
            f.write(json_line + '\n')

# Define all perturbation functions
perturbations = {
    "FRANK_synonym.json": replace_with_synonyms,
    "FRANK_paraphrased.json": paraphrase_text,
    "FRANK_typo.json": lambda x: introduce_typos(x, typo_rate=0.1)
}

# Generate datasets
for filename, perturb_func in perturbations.items():
    perturb_dataset(frank_data, perturb_func, filename)
