import json
import random
from transformers import pipeline
import nltk
from nltk.corpus import wordnet

frank_data = []
count_zero_labels = 0
count_other = 0
max_zero_labels = 50
max_other = 50
realsumm_data=[]

with open('dataset/frank/frank-data-100.json', 'r') as file:
    for line in file:
        line = json.loads(line)
        frank_data.append(line)

with open('dataset/realsumm/realsumm-data-100.json', 'r') as file:
    for line in file:
        line = json.loads(line)
        realsumm_data.append(line)

# Ensure required NLTK data is available for synonym replacement and tokenization
nltk.download('wordnet')
nltk.download('punkt')

# Function to replace words in text with their synonyms
def replace_with_synonyms(text):
    words = nltk.word_tokenize(text) # Tokenize the sentence into words
    new_words = [] # List to store words after replacing with synonyms
    for word in words:
        synonyms = wordnet.synsets(word)
        # Choose a synonym randomly if it exists, else keep the word
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_words.append(synonym)
        else:
            new_words.append(word)
    return ' '.join(new_words)

# Load a paraphrasing model from Hugging Face
paraphrase_model = pipeline("text2text-generation", model="t5-base")

# Function to paraphrase text using the pre-trained T5 model
def paraphrase_text(text):
    paraphrased = paraphrase_model(text, max_length=100, num_return_sequences=1)[0]['generated_text']
    return paraphrased

# Function to introduce typos into the text at a specified rate
def introduce_typos(text, typo_rate=0.1):
    words = list(text)
    num_typos = int(len(words) * typo_rate) # Calculate number of typos based on the typo rate
    for _ in range(num_typos):
        idx = random.randint(0, len(words) - 1) # Randomly choose an index to introduce a typo
        words[idx] = random.choice('abcdefghijklmnopqrstuvwxyz') # Replace the character with a random letter
    return ''.join(words)

# Function to apply a perturbation function to the entire dataset and save the modified dataset
def perturb_dataset(data, perturb_func, output_file):
    with open(output_file, 'w') as f:
        for record in data:
            new_record = record.copy()  # Avoid modifying the original
            new_record['sentences'] = [perturb_func(sentence) for sentence in record['sentences']]
            
            # Convert the modified record back to JSON and write it to the output file
            json_line = json.dumps(new_record)
            f.write(json_line + '\n')

# Define all perturbation functions
perturbations_frank = {
    "dataset/frank/frank_synonym.json": replace_with_synonyms,
    "dataset/frank/frank_paraphrased.json": paraphrase_text,
    "dataset/frank/frank_typo.json": lambda x: introduce_typos(x, typo_rate=0.1)
}
perturbations_realsumm = {
    "dataset/realsumm/realsumm_synonym.json": replace_with_synonyms,
    "dataset/realsumm/realsumm_paraphrased.json": paraphrase_text,
    "dataset/realsumm/realsumm_typo.json": lambda x: introduce_typos(x, typo_rate=0.1)
}

# Generate datasets
for filename, perturb_func in perturbations_frank.items():
    perturb_dataset(frank_data, perturb_func, filename)

for filename, perturb_func in perturbations_realsumm.items():
    perturb_dataset(realsumm_data, perturb_func, filename)

