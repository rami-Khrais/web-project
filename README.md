# Leveraging Knowledge Graphs for Enhanced Text Classification of CNN Articles

## Project Overview

This project aims to enhance the text classification of CNN articles by leveraging knowledge graphs. The workflow involves data preprocessing, sentiment analysis, named entity recognition, feature extraction, knowledge graph generation, graph embedding, and finally, applying machine learning models for classification.

## Project Pipeline

The project pipeline consists of several stages, each implemented in a dedicated Python script. Below is a description of each file and its role in the pipeline:

1. **first_pre_processing.py**:
    - **Purpose**: This script processes the initial dataset to remove any null values and clean the author column, ensuring that only the author names are retained without any extra information.
    - **Output**: A new data file with cleaned data and no null values.

2. **sent_analysis.py**:
    - **Purpose**: This script calculates the polarity scores for the articles, performing sentiment analysis to assess the sentiment conveyed in each article.
    - **Output**: A dataset with an additional column for polarity scores.

3. **ner_features.py**:
    - **Purpose**: This script generates new columns representing every available entity in the articles using Named Entity Recognition (NER).
    - **Output**: A dataset with additional columns for each identified entity (e.g., names, locations, organizations).

4. **all_text_features.py**:
    - **Purpose**: This script extracts various text features from the articles, including:
        - **text_length_features**: Length of the text featuers.
        - **pos_tag_features**: Part-of-speech tag features.
        - **lexical_diversity_features**: Lexical diversity measures.
    - **Output**: A dataset with new columns for each extracted text feature.

5. **data_pooling.py**:
    - **Purpose**: This script generates a TTL (Turtle) knowledge graph file representing the dataset with the new features.
    - **Output**: A TTL knowledge graph file.

6. **data_embedding.py**:
    - **Purpose**: This script creates graph walks and embeddings from the knowledge graph. These embeddings represent the relationships and structures within the knowledge graph.
    - **Output**: Graph walks and their corresponding embeddings.

7. **ml.py**:
    - **Purpose**: This script applies machine learning models to the generated embeddings to predict the class (category) of the articles.
    - **Output**: Classification model and evaluation results.

## File Descriptions

- `first_pre_processing.py`: Preprocesses the data by removing null values and cleaning the author column.
- `sent_analysis.py`: Performs sentiment analysis to calculate polarity scores.
- `ner_features.py`: Generates columns for each available entity using NER.
- `all_text_features.py`: Extracts various text features, including text length, POS tags, and lexical diversity.
- `data_pooling.py`: Creates a TTL knowledge graph file representing the dataset with the new features.
- `data_embedding.py`: Generates graph walks and embeddings from the knowledge graph.
- `ml.py`: Applies machine learning models to the embeddings for text classification.


## Conclusion

By following this pipeline, we leverage the structured information in knowledge graphs to enhance the classification of CNN articles. This approach combines traditional text processing techniques with advanced graph-based methods to improve the accuracy and reliability of text classification.

