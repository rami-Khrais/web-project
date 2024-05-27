# import libraries
import pandas as pd

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer


# download nltk corpus (first time only)
import nltk

nltk.download('vader_lexicon')

data_path = '/web_analysis/archive/CNN_Articels_clean_2/cleaned_data.csv'
df = pd.read_csv(data_path)

# create preprocess_text function
def preprocess_text(text):

    # Tokenize the text

    tokens = word_tokenize(text.lower())




    # Remove stop words

    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]




    # Lemmatize the tokens

    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]




    # Join the tokens back into a string

    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

# apply the function df

#df['processed_article_text'] = df['Article text'].apply(preprocess_text)
# initialize NLTK sentiment analyzer

analyzer = SentimentIntensityAnalyzer()


# create get_sentiment function

def get_sentiment(text):

    scores = analyzer.polarity_scores(text)

    print(scores['compound'])

    return scores['compound'] 




# apply get_sentiment function
print('new column started')
df['polarity'] = df['Article text'].apply(get_sentiment)
df.to_csv(data_path.replace('cleaned_data.csv','cleaned_data_sentment.csv'),index=False)
