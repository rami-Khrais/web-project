import pandas as pd
import spacy

# Load spaCy's pre-trained NER model
nlp = spacy.load('en_core_web_sm')

# List of possible entity labels in spaCy
entity_labels = [
    'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 
    'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'
]
counter = 1
def extract_entities(text):
    """Extract entities from the text using spaCy and return a dictionary of entity types with their names."""
    global counter
    print('article number = ', counter)
    counter += 1
    doc = nlp(text)
    entities = {label: [] for label in entity_labels}
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    # Join multiple entities with a comma
    for label in entities:
        entities[label] = ', '.join(entities[label]) if entities[label] else None
    return entities

def add_ner_columns(df, text_column):
    """Add NER columns to the DataFrame based on the specified text column."""
    # Initialize new columns with None
    for label in entity_labels:
        df[label] = None

    # Extract entities and update the DataFrame
    for i, row in df.iterrows():
        entities = extract_entities(row[text_column])
        for label in entity_labels:
            df.at[i, label] = entities[label]

    return df


df = pd.read_csv('cleaned_data_sentment.csv')

# Add NER columns
df = add_ner_columns(df, 'Article text')
df.to_csv('data_ner.csv',index=False)
print(df)

