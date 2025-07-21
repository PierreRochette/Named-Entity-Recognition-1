import spacy 

texts = [
    "Massoud goes for a walk in Strasbourg", 
    "Lucas is going to the train station",
    "Bernard Arnault is the CEO of LVMH", 
    "Solomon Hykes is the guy behind Docker", 
    "Pierre Rochette is the guy who followed this tutorial" 
]

nlp = spacy.load('en_core_web_md') # Change this line if the model is different

ner_labels = nlp.get_pipe('ner').labels # Labels that are supported by the model by default
print(ner_labels)

categories = ["ORG", "PERSON", "LOC"]

docs = [nlp(text) for text in texts]

for doc in docs: 
    entities = []
    for ent in doc.ents: 
        if ent.label_ in categories: 
            entities.append((ent.text, ent.label_))

    print(entities)