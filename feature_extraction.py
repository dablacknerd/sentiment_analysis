import pandas as pd
import os
from functions import *

columns_1 = [
    'sum_pos_adj','sum_neg_adj','sum_pos_adv','sum_neg_adv',
    'sum_pos_nou','sum_neg_nou','sum_pos_vrb','sum_neg_vrb',
    'adj_score_strength_pos','adj_score_strength_neg',
    'adv_score_strength_pos','adv_score_strength_neg',
    'nou_score_strength_pos','nou_score_strength_neg',
    'vrb_score_strength_pos','vrb_score_strength_neg',
    'adj_ratio_pos_pos','adj_ratio_pos_neg',
    'adv_ratio_pos_pos','adv_ratio_pos_neg',
    'nou_ratio_pos_pos','nou_ratio_pos_neg',
    'vrb_ratio_pos_pos','vrb_ratio_pos_neg'
]

columns_2 =[]
for i in range(1,14):
    for col in columns_1:
        columns_2.append( col + '_' + str(i))

columns_3 = columns_1 + columns_2

filr = os.path.join(os.getcwd(),"chennai_reviews.csv")
df = pd.read_csv(filr,encoding = "ISO-8859-1")
new_columns = list(df.columns[0:5])
df2 = df[new_columns].copy()
df_reviews_only = df2[df2.columns[0:3]].copy()
df_sentiment = pd.DataFrame(df2['Sentiment'].values,columns=['Sentiment'])
df_rating = pd.DataFrame(df2['Rating_Percentage'].values,columns=['Rating'])

review_text = df2['Review_Text'].values
full_features = []
for review in review_text:
    full_features.append(derive_features_a_b_c(review))

segment_features =[]

for review in review_text:
    sent_tokens = return_tokenized_sentences(review)
    #segment_counter = 1
    features = []
    for sent in sent_tokens:
        sentence_wordnet_pos = return_wordnet_pos(sent)
        syn_set_list = return_sentence_synset_list(sent,sentence_wordnet_pos)
        segment_polarity_info = return_sentence_polarity_info(syn_set_list)
        features.append(derive_features_e(segment_polarity_info))
        #segment_counter += 1
        #print(segment_counter)
    segment_features.append(finalize_segment_features(features))

combined_features = []
for i in range(0,len(segment_features)):
    combined_features.append(derive_full_features(full_features[i],segment_features[i]))

df3 = pd.DataFrame(full_features,columns=columns_1)
df4 = pd.DataFrame(segment_features,columns=columns_2)
df5 = pd.DataFrame(combined_features,columns=columns_3)

writer = pd.ExcelWriter('chennai_reviews_2.xlsx', engine='xlsxwriter')
df_reviews_only.to_excel(writer, sheet_name='Reviews Only',index=False)
df3.to_excel(writer, sheet_name='Full Features',index=False)
df4.to_excel(writer, sheet_name='Segment Features',index=False)
df5.to_excel(writer, sheet_name='Combined Features',index=False)
df_sentiment.to_excel(writer, sheet_name='Review Sentiment',index=False)
df_rating.to_excel(writer, sheet_name='Review Ratings',index=False)
writer.save()

print("Feature extraction complete.")
print("See chennai_reviews_2.xlsx for extracted features")
