import spacy
from spacy.training.example import Example
from spacy.util import minibatch
import random

train_data = [
    ("What is the price of 10 bananas?", {"entities": [(21, 23, "QUANTITY"), (24, 31, "PRODUCT")]}),
    ("I would like to buy 3 apples.", {"entities": [(22, 23, "QUANTITY"), (24, 30, "PRODUCT")]}),
    ("Can you tell me the cost of 7 laptops?", {"entities": [(29, 30, "QUANTITY"), (31, 38, "PRODUCT")]}),
    ("Do you sell 2 pineapples?", {"entities": [(13, 14, "QUANTITY"), (15, 25, "PRODUCT")]}),
    ("Please give me 4 chairs.", {"entities": [(17, 18, "QUANTITY"), (19, 25, "PRODUCT")]}),
    ("How much is 5 milk bottles?", {"entities": [(12, 13, "QUANTITY"), (14, 26, "PRODUCT")]}),
    ("Is the price of 6 tomatoes reasonable?", {"entities": [(17, 18, "QUANTITY"), (19, 27, "PRODUCT")]}),
    ("Tell me how much 8 oranges cost.", {"entities": [(18, 19, "QUANTITY"), (20, 27, "PRODUCT")]}),
    ("He needs 9 USB keys.", {"entities": [(9, 10, "QUANTITY"), (11, 19, "PRODUCT")]}),
    ("We ordered 15 smartphones.", {"entities": [(11, 13, "QUANTITY"), (14, 25, "PRODUCT")]}),
    ("Give me the value of 12 desks.", {"entities": [(23, 25, "QUANTITY"), (26, 31, "PRODUCT")]}),
    ("What's the price for 5 monitors?", {"entities": [(24, 25, "QUANTITY"), (26, 34, "PRODUCT")]}),
    ("Can I get 6 fans for my office?", {"entities": [(11, 12, "QUANTITY"), (13, 17, "PRODUCT")]}),
    ("I want to know the cost of 14 mugs.", {"entities": [(29, 31, "QUANTITY"), (32, 36, "PRODUCT")]}),
    ("How expensive are 13 headphones?", {"entities": [(19, 21, "QUANTITY"), (22, 32, "PRODUCT")]}),
    ("We need 11 notebooks urgently.", {"entities": [(8, 10, "QUANTITY"), (11, 20, "PRODUCT")]}),
    ("Is it possible to get 7 chairs?", {"entities": [(25, 26, "QUANTITY"), (27, 33, "PRODUCT")]}),
    ("I'm looking to purchase 4 speakers.", {"entities": [(26, 27, "QUANTITY"), (28, 36, "PRODUCT")]}),
    ("Find the cost for 2 kettles.", {"entities": [(20, 21, "QUANTITY"), (22, 29, "PRODUCT")]}),
    ("What would 9 keyboards cost?", {"entities": [(11, 12, "QUANTITY"), (13, 22, "PRODUCT")]}),
    ("She bought 16 plates yesterday.", {"entities": [(11, 13, "QUANTITY"), (14, 20, "PRODUCT")]}),
    ("How much are 17 pencils?", {"entities": [(13, 15, "QUANTITY"), (16, 23, "PRODUCT")]}),
    ("Estimate the price of 6 chairs.", {"entities": [(24, 25, "QUANTITY"), (26, 32, "PRODUCT")]}),
    ("Give me info on 3 staplers.", {"entities": [(16, 17, "QUANTITY"), (18, 26, "PRODUCT")]}),
    ("Can I get 8 water bottles?", {"entities": [(11, 12, "QUANTITY"), (13, 27, "PRODUCT")]}),
]

nlp = spacy.load('en_core_web_md')
# nlp = spacy.blank('en')

if 'ner' not in nlp.pipe_names: 
    ner = nlp.add_pipe('ner')
else :
    ner = nlp.get_pipe('ner')

for _, annotations in train_data:
    for ent in annotations['entities']: 
        if ent[2] not in ner.labels: 
            ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes): 
    optimizer = nlp.begin_training()

    epochs = 50
    for epoch in range(epochs): 
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=2)
        for batch in batches: 
            examples = []
            for text, annotations in batch: 
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, drop=0.5, losses=losses)
        print(f"Epoch {epoch + 1}, Losses: {losses}")

nlp.to_disk('custom_ner_model')

trained_nlp = spacy.load('custom_ner_model')

test_texts = [
    "How much for 3 oranges", 
    "I want 15 chairs for the conference", 
    "Can you give me the price for 6 desks ?"
]

for text in test_texts: 
    doc = trained_nlp(text)
    print(f"Text : {text}")
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])