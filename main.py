import math
import random
import re
import time
import json
from collections import defaultdict

import pandas as pd
import numpy as np

import spacy


STOP_WORDS = {',', 'для', '.', 'мм', 'с', ' ', 'в', 'и', 'см', 'на', 'м', 'по'}
pos_to_value = {
    'NOUN': 1.4,
    'PROPN': 1.6,
    'ADJ': 0.9,
    'NUM': 0.9,
    'ADP': 0
}

nlp = spacy.load("ru_core_news_sm")
df = pd.read_excel('СТЕ_Иркутск.xlsx')
indexed_df = df.set_index('ID СТЕ')


def word_distance(word_index):
    return 5 * math.exp(-((word_index - 1) / 2.5) ** 2)


def preprocess_parameter_names(parameter_name):
    width = ['ширин']
    height = ['высот']
    power = ['мощн']
    depth = ['глуб']
    diameter = ['диаметр']
    radius = ['радиус']
    amount = ['колич']
    length = ['длина']
    if len([p for p in width if p in parameter_name]) != 0 \
                and len([p for p in height if p in parameter_name]) != 0 \
                and len([p for p in length if p in parameter_name]) != 0:
        return 'габариты'
    if [p for p in width if p in parameter_name]:
        return 'ширина'
    if [p for p in height if p in parameter_name]:
        return 'высота'
    if [p for p in power if p in parameter_name]:
        return 'мощность'
    if [p for p in depth if p in parameter_name]:
        return 'глубина'
    if [p for p in diameter if p in parameter_name]:
        return 'диаметр'
    if [p for p in radius if p in parameter_name]:
        return 'радиус'
    if [p for p in amount if p in parameter_name]:
        return 'количество'
    if [p for p in length if p in parameter_name]:
        return 'длина'
    return None


def preprocess_char_value(value, unit=''):
    value = float(value)
    if any([i in unit for i in ['мм', 'миллиметр']]):
        unit = 'мм'
    elif any([i in unit for i in ['см', 'сантиметр']]):
        value, unit = int(value * 10), 'мм'
    elif any([i in unit for i in ['м', 'метр']]):
        value, unit = int(value * 1000), 'мм'
    elif any([i in unit for i in ['г', 'грамм']]):
        unit = 'г'
    elif any([i in unit for i in ['кг', 'килограмм']]):
        value, unit = int(value * 1000), 'г'
    return value, unit


def find_num_properties(root):
    def rec_find(node, find_list):
        if len(find_list) == 0:
            return [[node]]
        ways = []
        for child in node.children:
            if child.dep_ == find_list[0]:
                new_ways = rec_find(child, find_list[1:])
                for way in new_ways:
                    ways.append([node] + way)
        return ways

    # 1
    to_find = ['nmod', 'nummod']
    ways1 = [w for w in rec_find(root, to_find) if (max(wrd.i for wrd in w) - min(wrd.i for wrd in w) <= 2)]
    # 2
    to_find1, to_find2 = ['nmod'], ['nmod', 'nummod']
    ways2_1 = [w[1:] for w in rec_find(root, to_find1)]
    ways2_2 = [w[1:] for w in rec_find(root, to_find2)]
    ways2 = [w1 + w2 for w1 in ways2_1 for w2 in ways2_2]
    ways2 = [w for w in ways2 if (max(wrd.i for wrd in w) - min(wrd.i for wrd in w) <= 2)]
    # 3
    to_find = ['nmod', 'nmod', 'nummod']
    ways3 = [w[1:] for w in rec_find(root, to_find) if (max(wrd.i for wrd in w) - min(wrd.i for wrd in w) <= 2)]
    all_ways = [w for w in ways1] + [w for w in ways2] + [w for w in ways3]
    return all_ways

with open('name_index_table.json') as f:  # name to products
    name_index_table = json.loads(f.read())
    
with open('char_index_table.json') as f:  # char to products
    char_index_table = json.loads(f.read())


def get_products(raw_query):

    # query processing
    query = nlp(raw_query)
    query = [w if w.pos_ != 'X' else nlp(f',{w.text},')[1] for w in query]
    docs_dist_coefs = defaultdict(float)
    for word in query:
        if word.lemma_ not in STOP_WORDS and preprocess_parameter_names(word.text) is None and word.lemma_ in name_index_table:
            for doc in name_index_table[word.lemma_]:
                docs_dist_coefs[doc] += word_distance(len(list(word.ancestors)) + 1) * pos_to_value.get(word.pos_, 0)
    products_to_coef = {}
    for doc, coef in docs_dist_coefs.items():
        if coef <= 1e-6:
            continue
        products_to_coef[doc] = coef
    products_to_coef = sorted(products_to_coef.items(), key=lambda x: products_to_coef[x[0]], reverse=True)
    products_to_coef = [(k, v) for k, v in products_to_coef if v > 6]
    products = [k for k, v in products_to_coef]
    res = indexed_df.loc[products].copy()
    res['word_dist'] = [v for k, v in products_to_coef]


    # chars processing
    res['char_flags'] = np.zeros(len(res))
    product_metrics = {}
    products_1000 = products[:1000]
    for product_id in products_1000:
        chars = indexed_df.loc[product_id]['Характеристики']
        try:
            json_ = json.loads(chars)
        except:
            continue
        for char in json_:
            for w in query:
                if 'Value' in char and w.text in char['Value'].lower() and char['Value'] in char_index_table and \
                        product_id in char_index_table[char['Value']]:
                    res.loc[product_id, 'char_flags'] += 1
                preprocessed_name = preprocess_parameter_names(char['Name'])
                if preprocessed_name is None:
                    continue
                if w.dep_ == 'ROOT' or w.dep_ == 'nmod':
                    for way in find_num_properties(w):
                        if len(way) == 3:
                            p, u, v = way
                            p = preprocess_parameter_names(p.text)
                            v, u = preprocess_char_value(v.text, u.text)
                            if p is None:
                                continue
                            if p == preprocessed_name and 'Value' in char and 'Unit' in char:
                                new_val, _ = preprocess_char_value(char['Value'], char['Unit'])
                                smape_val = abs(new_val - v) / (abs(new_val) + abs(v))
                                product_metrics[product_id] = max(product_metrics.get(product_id, 0),
                                                                  1 / (1 + smape_val / 0.35) ** 2)

    res['char_metrics'] = np.ones(len(res)) * 6
    res.loc[list(product_metrics.keys()), 'char_metrics'] += np.array(list(product_metrics.values()))

    # f-beta score
    BETA = 3
    res['word_num_char'] = (1 + BETA ** 2) * res['word_dist'] * res['char_metrics'] / (BETA ** 2 * res['char_metrics'] + res['word_dist'])
    BETA2 = 0.4
    res['word_num_cat_char'] = (1 + BETA2 ** 2) * res['word_num_char'] * res['char_flags'] / (BETA2 ** 2 * res['word_num_char'] + res['char_flags'])

    sorted_res = res.sort_values(by='word_num_cat_char', ascending=False).reset_index()
    sorted_res = sorted_res[['Название СТЕ', 'Категория', 'Характеристики', 'ID СТЕ', 'Код КПГЗ']]
    sorted_res['Характеристики'] = sorted_res['Характеристики'].apply(web_format)
    return sorted_res.values.tolist()


def web_format(chars):
    try:
        chars = json.loads(chars)
    except:
        return []
    return [[ch['Name'], ch['Value']] for ch in chars if 'Value' in ch]