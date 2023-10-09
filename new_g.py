import math
import re

def preprocess(filename):
    preprocessed = []
    with open(filename, 'r') as f:
        data = f.read()
    sentences = [s.strip() for s in data.split('\n') if len(s.strip()) > 0]
    for sentence in sentences:
        sentence = sentence.strip()
        sentence = re.sub(r'<unk>', '<UNK>', sentence)
        sentence = re.sub(r'\b\d+\b', '<NUM>', sentence)
        sentence = re.sub(r' N ', ' <NUM> ', sentence)
        sentence = re.sub(r'[^\w\s<UNK><NUM>]', '', sentence)
        sentence = ' '.join([word.lower() if not re.match(r'<[A-Z]+>', word) else word for word in sentence.split()])
        preprocessed.append(sentence.split())
    return preprocessed

def preprocess_dataset(filename_wiki):
    # Open the file and read its contents
    with open(filename_wiki, 'r', encoding='utf-8') as f:
        contents = f.read()

    # Split the contents into a list of articles
    articles = contents.split('\n')[:-1]

    # Define a function to preprocess each article
    def preprocess_text(text):
        # Replace all numbers with NUM tag
        text = re.sub(r'\d+', 'NUM', text)
        # Split text into words and punctuation marks
        words = re.findall(r'\w+|[^\w\s]', text)
        # Handle unknown words and punctuation marks by giving them UNK tag
        words = ['UNK' if len(word) > 20 else word for word in words]
        return words

    # Preprocess each article and add it to a list of lists dataset
    dataset = [preprocess_text(article) for article in articles]

    return dataset

class LanguageModel:
    def __init__(self, q_number=2):
        self.n = q_number
        self.vocab = set()
        self.counts = {}
        self.total_count = 0

    def train(self, corpus):
        for sentence in corpus:
            for i in range(len(sentence) - self.n + 1):
                context = tuple(sentence[i:i + self.n - 1])
                word = sentence[i + self.n - 1]
                if context not in self.counts:
                    self.counts[context] = {}
                if word not in self.counts[context]:
                    self.counts[context][word] = 0
                self.counts[context][word] += 1
                self.total_count += 1
                self.vocab.add(word)

    def get_probability(self, context, word):
        if context in self.counts and word in self.counts[context]:
            return self.counts[context][word] / sum(self.counts[context].values())
        else:
            return 0

    def get_perplexity(self, sentences):
        log_prob_sum = 0
        word_count = 0
        for sentence in sentences:
            for i in range(len(sentence) - self.n + 1):
                context = tuple(sentence[i:i + self.n - 1])
                word = sentence[i + self.n - 1]
                log_prob_sum += math.log(self.get_probability(context, word) or 1)
                word_count += 1
        return math.exp(-log_prob_sum / word_count)

file_train = 'ptb.train.txt'
file_test = 'ptb.test.txt'
file_train_wiki = 'wiki.test.raw'
file_test_wiki = 'wiki.train.raw'
train_sentences = preprocess(file_train)
test_sentences = preprocess(file_test)
train_sentences_wiki = preprocess_dataset(file_train_wiki)
test_sentences_wiki = preprocess_dataset(file_test_wiki)

print(train_sentences[0][0])




lm = LanguageModel(q_number=2)
lm.train(train_sentences)
lm_wiki = LanguageModel(q_number=2)
lm_wiki.train(train_sentences_wiki)
print(f"Complexity of ptb dataset: {lm.get_perplexity(test_sentences)}")
print(f"Complexity of wiki dataset: {lm_wiki.get_perplexity(test_sentences_wiki)}")