from nltk import sent_tokenize,word_tokenize,pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import numpy as np

def return_tokenized_sentences(review):
    return sent_tokenize(review)

def return_wordnet_pos(sentence):
    word_tokens = word_tokenize(sentence)
    pos_tagged_sentence = pos_tag(word_tokens)
    result = []
    for obj in pos_tagged_sentence:
        if obj[1][0:2] == 'NN':
            result.append((obj[0],'n'))
        elif obj[1][0:2] == 'VB':
            result.append((obj[0],'v'))
        elif obj[1][0:2] == 'JJ':
            result.append((obj[0],'a'))
        elif obj[1][0:2] == 'RB':
            result.append((obj[0],'r'))
        else:
            continue
    return result

def return_sentence_synset_list(sentence,sentence_wordnet_pos):
    syn_set_list =[]
    for obj in sentence_wordnet_pos:
        syn_set = lesk(word_tokenize(sentence),obj[0],obj[1])
        if syn_set is None:
            continue
        else:
            syn_set_name = syn_set.name()
            syn_set_list.append((obj[0],obj[1],syn_set_name))
    return syn_set_list

def return_sentence_polarity_info(syn_set_list):
    final_result = []
    for item in syn_set_list:
        result = swn.senti_synset(item[2])
        pos_score = result.pos_score()
        neg_score = result.neg_score()
        obj_score = result.obj_score()
        if pos_score > neg_score:
            dict_result = {
                "token":item[0],
                "part_of_speech":item[1],
                "polarity":"+ve",
                "score":result.pos_score(),
                "obj_score":result.obj_score()
            }
            final_result.append(dict_result)
        elif neg_score > pos_score:
            dict_result = {
                "token":item[0],
                "part_of_speech":item[1],
                "polarity":"-ve",
                "score":result.neg_score(),
                "obj_score":result.obj_score()
            }
            final_result.append(dict_result)
        else:
            continue
    return final_result

def derive_features_e(segment_polarity_info):
    pos_adj =[]
    neg_adj =[]
    pos_adv =[]
    neg_adv =[]
    pos_nou =[]
    neg_nou =[]
    pos_vrb =[]
    neg_vrb =[]
    subj_list =[]
    term_counter = 0

    for obj in segment_polarity_info:
        subj_list.append(obj['obj_score'])
        term_counter += 1
        if obj['part_of_speech'] == 'a':
            if obj['polarity'] == '+ve':
                pos_adj.append(obj['score'])
            else:
                neg_adj.append(obj['score'])
        elif obj['part_of_speech'] == 'r':
            if obj['polarity'] == '+ve':
                pos_adv.append(obj['score'])
            else:
                neg_adv.append(obj['score'])
        elif obj['part_of_speech'] == 'n':
            if obj['polarity'] == '+ve':
                pos_nou.append(obj['score'])
            else:
                neg_nou.append(obj['score'])
        elif obj['part_of_speech'] == 'v':
            if obj['polarity'] == '+ve':
                pos_vrb.append(obj['score'])
            else:
                neg_vrb.append(obj['score'])
        else:
            continue
    #calculate segment features a
    sum_pos_adj = np.array(pos_adj).sum()
    sum_neg_adj = np.array(neg_adj).sum()
    sum_pos_adv = np.array(pos_adv).sum()
    sum_neg_adv = np.array(neg_adv).sum()
    sum_pos_nou = np.array(pos_nou).sum()
    sum_neg_nou = np.array(neg_nou).sum()
    sum_pos_vrb = np.array(pos_vrb).sum()
    sum_neg_vrb = np.array(neg_vrb).sum()
    sub_sum = np.array(subj_list).sum()
    #sub_score = round(sub_sum/term_counter,2)
    sum_pos_terms = sum_pos_adj + sum_pos_adv + sum_pos_nou + sum_pos_vrb
    sum_neg_terms = sum_neg_adj + sum_neg_adv + sum_neg_nou + sum_neg_vrb

    #calculate segment features b
    if sum_pos_terms > 0.0:
        adj_score_strength_pos = round(sum_pos_adj/sum_pos_terms,2)
        adv_score_strength_pos = round(sum_pos_adv/sum_pos_terms,2)
        nou_score_strength_pos = round(sum_pos_nou/sum_pos_terms,2)
        vrb_score_strength_pos = round(sum_pos_vrb/sum_pos_terms,2)
    else:
        adj_score_strength_pos = 0.0
        adv_score_strength_pos = 0.0
        nou_score_strength_pos = 0.0
        vrb_score_strength_pos = 0.0
    if sum_neg_terms > 0.0:
        adj_score_strength_neg = round(sum_neg_adj/sum_neg_terms,2)
        adv_score_strength_neg = round(sum_neg_adv/sum_neg_terms,2)
        nou_score_strength_neg = round(sum_neg_nou/sum_neg_terms,2)
        vrb_score_strength_neg = round(sum_neg_vrb/sum_neg_terms,2)
    else:
        adj_score_strength_neg = 0.0
        adv_score_strength_neg = 0.0
        nou_score_strength_neg = 0.0
        vrb_score_strength_neg = 0.0

    #calculate segment features c
    if term_counter > 0:
        adj_ratio_pos_pos = round(float(len(pos_adj))/float(term_counter),2)
        adv_ratio_pos_pos = round(float(len(pos_adv))/float(term_counter),2)
        nou_ratio_pos_pos = round(float(len(pos_nou))/float(term_counter),2)
        vrb_ratio_pos_pos = round(float(len(pos_vrb))/float(term_counter),2)
        adj_ratio_pos_neg = round(float(len(neg_adj))/float(term_counter),2)
        adv_ratio_pos_neg = round(float(len(neg_adv))/float(term_counter),2)
        nou_ratio_pos_neg = round(float(len(neg_nou))/float(term_counter),2)
        vrb_ratio_pos_neg = round(float(len(neg_vrb))/float(term_counter),2)
    else:
        adj_ratio_pos_pos = 0.0
        adv_ratio_pos_pos = 0.0
        nou_ratio_pos_pos = 0.0
        vrb_ratio_pos_pos = 0.0
        adj_ratio_pos_neg = 0.0
        adv_ratio_pos_neg = 0.0
        nou_ratio_pos_neg = 0.0
        vrb_ratio_pos_neg = 0.0

    return [
            sum_pos_adj,
            sum_neg_adj,
            sum_pos_adv,
            sum_neg_adv,
            sum_pos_nou,
            sum_neg_nou,
            sum_pos_vrb,
            sum_neg_vrb,
            adj_score_strength_pos,
            adj_score_strength_neg,
            adv_score_strength_pos,
            adv_score_strength_neg,
            nou_score_strength_pos,
            nou_score_strength_neg,
            vrb_score_strength_pos,
            vrb_score_strength_neg,
            adj_ratio_pos_pos,
            adj_ratio_pos_neg,
            adv_ratio_pos_pos,
            adv_ratio_pos_neg,
            nou_ratio_pos_pos,
            nou_ratio_pos_neg,
            vrb_ratio_pos_pos,
            vrb_ratio_pos_neg
    ]

def derive_features_a_b_c(review):
    sentence_tokens = return_tokenized_sentences(review)
    adj_pos = []
    adj_neg = []
    vrb_pos = []
    vrb_neg = []
    adv_pos = []
    adv_neg = []
    nou_pos = []
    nou_neg = []
    term_counter = 0

    for sentence in sentence_tokens:
        sentence_wordnet_pos = return_wordnet_pos(sentence)
        syn_set_list = return_sentence_synset_list(sentence,sentence_wordnet_pos)
        for item in syn_set_list:
            term_counter += 1
            result = swn.senti_synset(item[2])
            pos_score = result.pos_score()
            neg_score = result.neg_score()
            obj_score = result.obj_score()

            if pos_score > neg_score and item[1] == 'a':
                adj_pos.append(pos_score)
            elif pos_score < neg_score and item[1] == 'a':
                adj_neg.append(neg_score)
            elif pos_score > neg_score and item[1] == 'r':
                adv_pos.append(pos_score)
            elif pos_score < neg_score and item[1] == 'r':
                adv_neg.append(neg_score)
            elif pos_score > neg_score and item[1] == 'v':
                vrb_pos.append(pos_score)
            elif pos_score < neg_score and item[1] == 'v':
                vrb_neg.append(neg_score)
            elif pos_score > neg_score and item[1] == 'n':
                nou_pos.append(pos_score)
            elif pos_score < neg_score and item[1] == 'n':
                nou_neg.append(neg_score)
            else:
                continue
    sum_pos_adj = np.array(adj_pos).sum()
    sum_neg_adj = np.array(adj_neg).sum()
    sum_pos_adv = np.array(adv_pos).sum()
    sum_neg_adv = np.array(adv_neg).sum()
    sum_pos_nou = np.array(nou_pos).sum()
    sum_neg_nou = np.array(nou_neg).sum()
    sum_pos_vrb = np.array(vrb_pos).sum()
    sum_neg_vrb = np.array(vrb_neg).sum()
    sum_pos_terms = sum_pos_adj + sum_pos_adv + sum_pos_nou + sum_pos_vrb
    sum_neg_terms = sum_neg_adj + sum_neg_adv + sum_neg_nou + sum_neg_vrb

    #calculate segment features b
    if sum_pos_terms > 0.0:
        adj_score_strength_pos = round(sum_pos_adj/sum_pos_terms,2)
        adv_score_strength_pos = round(sum_pos_adv/sum_pos_terms,2)
        nou_score_strength_pos = round(sum_pos_nou/sum_pos_terms,2)
        vrb_score_strength_pos = round(sum_pos_vrb/sum_pos_terms,2)
    else:
        adj_score_strength_pos = 0.0
        adv_score_strength_pos = 0.0
        nou_score_strength_pos = 0.0
        vrb_score_strength_pos = 0.0
    if sum_neg_terms > 0.0:
        adj_score_strength_neg = round(sum_neg_adj/sum_neg_terms,2)
        adv_score_strength_neg = round(sum_neg_adv/sum_neg_terms,2)
        nou_score_strength_neg = round(sum_neg_nou/sum_neg_terms,2)
        vrb_score_strength_neg = round(sum_neg_vrb/sum_neg_terms,2)
    else:
        adj_score_strength_neg = 0.0
        adv_score_strength_neg = 0.0
        nou_score_strength_neg = 0.0
        vrb_score_strength_neg = 0.0
    if term_counter > 0:
        adj_ratio_pos_pos = round(float(len(adj_pos))/float(term_counter),2)
        adv_ratio_pos_pos = round(float(len(adv_pos))/float(term_counter),2)
        nou_ratio_pos_pos = round(float(len(nou_pos))/float(term_counter),2)
        vrb_ratio_pos_pos = round(float(len(vrb_pos))/float(term_counter),2)
        adj_ratio_pos_neg = round(float(len(adj_neg))/float(term_counter),2)
        adv_ratio_pos_neg = round(float(len(adv_neg))/float(term_counter),2)
        nou_ratio_pos_neg = round(float(len(nou_neg))/float(term_counter),2)
        vrb_ratio_pos_neg = round(float(len(vrb_neg))/float(term_counter),2)
    else:
        adj_ratio_pos_pos = 0.0
        adv_ratio_pos_pos = 0.0
        nou_ratio_pos_pos = 0.0
        vrb_ratio_pos_pos = 0.0
        adj_ratio_pos_neg = 0.0
        adv_ratio_pos_neg = 0.0
        nou_ratio_pos_neg = 0.0
        vrb_ratio_pos_neg = 0.0

    return [
         sum_pos_adj,sum_neg_adj,sum_pos_adv,sum_neg_adv,
         sum_pos_nou,sum_neg_nou,sum_pos_vrb,sum_neg_vrb,
         adj_score_strength_pos,adj_score_strength_neg,
         adv_score_strength_pos,adv_score_strength_neg,
         nou_score_strength_pos,nou_score_strength_neg,
         vrb_score_strength_pos,vrb_score_strength_neg,
         adj_ratio_pos_pos,adj_ratio_pos_neg,
         adv_ratio_pos_pos,adv_ratio_pos_neg,
         nou_ratio_pos_pos,nou_ratio_pos_neg,
         vrb_ratio_pos_pos,vrb_ratio_pos_neg
    ]

def derive_full_features(review_features,segment_features):
    return review_features + segment_features

def finalize_segment_features(features):
    segment_count = len(features)
    f =[]
    if segment_count < 13:
        segment_makeup_count = 13 - segment_count
        t = 24 * segment_makeup_count
        z = [0.0 for i in range(0,t)]
        for feature in features:
            f = f + feature
        f = f + z
    else:
        for feature in features:
            f = f + feature
    return f
