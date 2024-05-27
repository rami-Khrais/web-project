import math
import pandas as pd
import owlready2
import re

onto = owlready2.get_ontology("article_ontology.owl").load()
data = pd.read_csv("cleaned_output.csv")


data = data[data['Category'] != 'vr']
data = data[data['Category'] != 'style']
data = data[data['Url'].str.startswith('https://') & data['Date published'].str[0].str.isdigit()]
data = data.reset_index(drop=True)

news_data = data[data['Category'] == 'news']

# Sample 10,000 rows from the "news" category
news_sample = news_data.sample(n=2800, random_state=1)

sport_data = data[data['Category'] == 'sport']

# Sample 5000 rows from the "sport" category
sport_sample = sport_data.sample(n=2800, random_state=1)


rest_data = data[data['Category'] != 'news']
rest_data = rest_data[rest_data['Category'] != 'sport']

new_data = pd.concat([news_sample, sport_sample])
new_data = pd.concat([new_data, rest_data])

new_data = new_data.sample(frac=1, random_state=1).reset_index(drop=True)

for index, row in new_data.iterrows():
    print(index)
    if type(row['Article text'])!=str:
        continue

    article = onto.Article(iri = row['Url'])

    category = onto.Category(row['Category'].replace(' ','_'))
    section = onto.Section(row['Section'].replace(' ','_').replace('-','_'))
    clean_string = re.sub(r'[^a-zA-Z0-9]', '', row['Author'].replace(' ',''))
    author = onto.Author(re.sub(r'[^\w\s]', '', clean_string))

    category.hasName.append(row['Category'])
    author.hasName.append(row['Author'].replace('"',''))
    section.hasName.append(row['Section'])

    #Object property assertion for article class
    article.hasCategory.append(category)
    article.hasSection.append(section)
    article.hasAuthor.append(author)

    #Data property assertion for article class
    if type(row['Date published'])==str:
        article.hasDatePublished.append(row['Date published'])

    if type(row['Headline'])==str:
        article.hasHeadline.append(row['Headline'].replace('"','').replace('\'',''))

    if type(row['Description'])==str:
        article.hasDescription.append(row['Description'].replace('"','').replace('\'',''))

    if type(row['Keywords'])==str:
        article.hasKeywords.append(row['Keywords'].replace('"','').replace('\'',''))

    if type(row['Second headline'])==str:
        article.hasSecondHeadline.append(row['Second headline'].replace('"','').replace('\'',''))

    if type(row['Article text'])==str:
        clean_string = re.sub(r'[^a-zA-Z0-9]', '', row['Article text'])
        article.hasArticleText.append(re.sub(r'[^\w\s]', '', clean_string))

    if math.isnan(row['polarity'])==False:
        article.hasPolarity.append(row['polarity'])

    if type(row['PERSON'])==str:
        article.hasPERSON.append(row['PERSON'].replace('"','').replace('\'',''))

    if type(row['NORP'])==str:
        article.hasNORP.append(row['NORP'].replace('"','').replace('\'',''))

    if type(row['FAC'])==str:
        article.hasFAC.append(row['FAC'].replace('"','').replace('\'',''))

    if type(row['ORG'])==str:
        article.hasORG.append(row['ORG'].replace('"','').replace('\'',''))

    if type(row['GPE'])==str:
        article.hasGPE.append(row['GPE'].replace('"','').replace('\'',''))

    if type(row['LOC'])==str:
        article.hasLOC.append(row['LOC'].replace('"','').replace('\'',''))

    if type(row['PRODUCT'])==str:
        article.hasPRODUCT.append(row['PRODUCT'].replace('"','').replace('\'',''))

    if type(row['EVENT'])==str:
        article.hasEVENT.append(row['EVENT'].replace('"','').replace('\'',''))

    if type(row['WORK_OF_ART'])==str:
        article.hasWORK_OF_ART.append(row['WORK_OF_ART'].replace('"','').replace('\'',''))

    if type(row['LAW'])==str:
        article.hasLAW.append(row['LAW'].replace('"','').replace('\'',''))

    if type(row['LANGUAGE'])==str:
        article.hasLANGUAGE.append(row['LANGUAGE'].replace('"','').replace('\'',''))

    if type(row['DATE'])==str:
        article.hasDATE.append(row['DATE'].replace('"','').replace('\'',''))

    if type(row['TIME'])==str:
        article.hasTIME.append(row['TIME'].replace('"','').replace('\'',''))

    if type(row['PERCENT'])==str:
        article.hasPERCENT.append(row['PERCENT'].replace('"','').replace('\'',''))

    if type(row['MONEY'])==str:
        article.hasMONEY.append(row['MONEY'].replace('"','').replace('\'',''))

    if type(row['QUANTITY'])==str:
        article.hasQUANTITY.append(row['QUANTITY'].replace('"','').replace('\'',''))

    if type(row['ORDINAL'])==str:
        article.hasORDINAL.append(row['ORDINAL'].replace('"','').replace('\'',''))

    if type(row['CARDINAL'])==str:
        article.hasCARDINAL.append(row['CARDINAL'].replace('"','').replace('\'',''))

    if math.isnan(row['word_count'])==False:
        article.hasWordCount.append(row['word_count'])

    if math.isnan(row['char_count'])==False:
        article.hasCharCount.append(row['char_count'])

    if math.isnan(row['sentence_count'])==False:
        article.hasSentenceCount.append(row['sentence_count'])

    if math.isnan(row['avg_word_length'])==False:
        article.avg_word_length.append(row['avg_word_length'])

    if math.isnan(row['avg_sentence_length'])==False:
        article.avg_sentence_length.append(row['avg_sentence_length'])

    if math.isnan(row['type_token_ratio'])==False:
        article.type_token_ratio.append(row['type_token_ratio'])

    if math.isnan(row['hapax_legomena'])==False:
        article.hapax_legomena.append(row['hapax_legomena'])

    if math.isnan(row['pos_sym'])==False:
        article.pos_sym.append(row['pos_sym'])

    if math.isnan(row['pos_space'])==False:
        article.pos_space.append(row['pos_space'])

    if math.isnan(row['pos_punct'])==False:
        article.pos_punct.append(row['pos_punct'])

    if math.isnan(row['pos_propn'])==False:
        article.pos_propn.append(row['pos_propn'])

    if math.isnan(row['pos_adv'])==False:
        article.pos_adv.append(row['pos_adv'])

    if math.isnan(row['pos_pron'])==False:
        article.pos_pron.append(row['pos_pron'])

    if math.isnan(row['pos_verb'])==False:
        article.pos_verb.append(row['pos_verb'])

    if math.isnan(row['pos_det'])==False:
        article.pos_det.append(row['pos_det'])

    if math.isnan(row['pos_noun'])==False:
        article.pos_noun.append(row['pos_noun'])

    if math.isnan(row['pos_adp'])==False:
        article.pos_adp.append(row['pos_adp'])

    if math.isnan(row['pos_cconj'])==False:
        article.pos_cconj.append(row['pos_cconj'])

    if math.isnan(row['pos_adj'])==False:
        article.pos_adj.append(row['pos_adj'])

    if math.isnan(row['pos_num'])==False:
        article.pos_num.append(row['pos_num'])

    if math.isnan(row['pos_aux'])==False:
        article.pos_aux.append(row['pos_aux'])

    if math.isnan(row['pos_part'])==False:
        article.pos_part.append(row['pos_part'])

    if math.isnan(row['pos_sconj'])==False:
        article.pos_sconj.append(row['pos_sconj'])

    if math.isnan(row['pos_x'])==False:
        article.pos_x.append(row['pos_x'])

    if math.isnan(row['pos_intj'])==False:
        article.pos_intj.append(row['pos_intj'])

onto.save('article_ontology_with_data.ttl', format = 'ntriples')
new_data.to_csv('new.csv')
