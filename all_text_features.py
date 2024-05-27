import pandas as pd
import spacy
import numpy as np

# Load spaCy's pre-trained model
nlp = spacy.load('en_core_web_sm')
counter = 1
def text_length_features(text):
    """Calculate text length features."""
    global counter
    print('article number = ',counter)
    counter += 1
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = len(list(nlp(text).sents))
    avg_word_length = char_count / word_count if word_count else 0
    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    return word_count, char_count, sentence_count, avg_word_length, avg_sentence_length

def pos_tag_features(text):
    """Calculate part-of-speech tag distribution."""
    doc = nlp(text)
    pos_counts = doc.count_by(spacy.attrs.POS)
    total = sum(pos_counts.values())
    pos_features = {nlp.vocab.strings[k]: v / total for k, v in pos_counts.items()}
    return pos_features


def lexical_diversity_features(text):
    """Calculate lexical diversity features."""
    words = text.split()
    unique_words = set(words)
    type_token_ratio = len(unique_words) / len(words) if words else 0
    hapax_legomena = sum(1 for word in unique_words if words.count(word) == 1)
    return type_token_ratio, hapax_legomena

def extract_features(df, text_column):
    """Extract features from the text column."""
    # Initialize feature columns
    features = ['word_count', 'char_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length',
                'type_token_ratio', 'hapax_legomena']
    
    for pos in nlp.get_pipe("tagger").labels:
        features.append(f'pos_{pos.lower()}')

    for feature in features:
        df[feature] = None
    
    # Extract features for each row
    for i, row in df.iterrows():
        text = row[text_column]
        wc, cc, sc, awl, asl = text_length_features(text)
        ttr, hl = lexical_diversity_features(text)
        pos_feats = pos_tag_features(text)
        
        df.at[i, 'word_count'] = wc
        df.at[i, 'char_count'] = cc
        df.at[i, 'sentence_count'] = sc
        df.at[i, 'avg_word_length'] = awl
        df.at[i, 'avg_sentence_length'] = asl
        df.at[i, 'type_token_ratio'] = ttr
        df.at[i, 'hapax_legomena'] = hl

        for pos, value in pos_feats.items():
            df.at[i, f'pos_{pos.lower()}'] = value

    return df


df = pd.read_csv('data_ner.csv')

# Extract features
df = extract_features(df, 'Article text')

df.to_csv('data_all_featuers.csv',index=False)
print(df)

