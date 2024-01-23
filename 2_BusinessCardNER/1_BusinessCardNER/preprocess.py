import spacy
from spacy.tokens import DocBin
import pickle

nlp = spacy.blank("en")
training_data = pickle.load(open('./Data/Train_data.pickle', 'rb'))
test_data = pickle.load(open('./Data/Test_data.pickle', 'rb'))

# the DocBin will store the example documents (train data)
db_train = DocBin()
for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db_train.add(doc)
db_train.to_disk("./Data/train.spacy")

# the DocBin will store the example documents (test data)
db_test = DocBin()
for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db_test.add(doc)
db_test.to_disk("./Data/test.spacy")