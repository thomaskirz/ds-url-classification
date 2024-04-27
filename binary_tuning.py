import re
from pathlib import Path

import nltk.corpus
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Constants
DOWNSCALE = 0.001
TRAINING_SET_FRAC = 0.8

# Function to convert URLs to ngrams
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

def generate_ngrams_from_url(url: str, n: [int]) -> [str]:
    # turn to lowercase string without non-alphabetical characters and remove http
    url = url.lower()
    non_alpha = re.compile(r'[^a-z]')
    http = re.compile(r'http')
    url = non_alpha.sub('', url)
    url = http.sub('', url)

    ngrams = []
    for i in n:
        ngrams.extend([t for t in re.findall(rf'(?=(.{{{i}}}))', url) if t not in stopwords])
    return ngrams


def generate_grams_from_url_enriched(url: str, n: [int]) -> [str]:
    # add url length, number of special characters, numerical count, word count, url depth and number of subdomains
    # (dots before the first slash)
    url_without_protocol = url[url.find('//') + 2:] if '//' in url else url
    extra_features = [len(url),
                      len(re.findall(r'[^a-zA-Z0-9]', url)),
                      len(re.findall(r'\d', url)),
                      len(re.findall(r'\w+', url)),
                      len(re.findall(r'/', url)),
                      len(re.findall(r'\.', url_without_protocol[:url_without_protocol.find('/')]))]
    return list(map(str, extra_features)) + generate_ngrams_from_url(url, n)


# def tokens_from_tokenizer(url: str, tokenizer: PreTrainedTokenizer, min_length: int = 2, bigrams: bool = True) -> str:
#     url = url.lower()
#     www = re.compile('www.')
#     non_alpha = re.compile(r'[^a-z]')
#     http = re.compile(r'http')
#     url = www.sub('', url)
#     url = non_alpha.sub('', url)
#     url = http.sub('', url)
#
#     tokens = tokenizer.tokenize(url)
#     tokens = [token for token in tokens if len(token) >= min_length]
#     if len(tokens) > 0:
#         end = re.compile(r'</w>$')
#         tokens[-1] = end.sub('', tokens[-1])
#         if bigrams:
#             tokens.extend([a + b for a, b in zip(tokens[:-1], tokens[1:])])
#     return tokens


def ngram_analyzer(ns: [int]):
    return lambda url: generate_ngrams_from_url(url, ns)


def ngram_analyzer_enriched(ns: [int]):
    return lambda url: generate_grams_from_url_enriched(url, ns)


# def tokenizer_analyzer(tokenizer: PreTrainedTokenizer, min_length: int = 2, bigrams: bool = True):
#     return lambda url: tokens_from_tokenizer(url, tokenizer, min_length, bigrams)


def load_dataset():
    df = pd.read_csv('data/odp.csv')
    df.columns = ['index', 'url', 'label_name']
    df.drop('index', axis=1, inplace=True)

    labels = list(set(df['label_name']))
    labels.sort()

    # some rows lack a URL so we remove those
    print(f'Dropping {df.url.isnull().sum()} empty URL entries from ODP dataset')
    df.drop(df[df.url.isnull()].index, inplace=True)

    # Sample so that 50% of the set is the positive class and the rest is distributed equally
    datasets = []
    for label in labels:
        positive = df[df.label_name == label].sample(frac=DOWNSCALE).assign(bin_label=1)
        other_topics = df[df.label_name != label].copy()
        other_topics['count'] = 1. / other_topics.groupby('label_name')['label_name'].transform('count')
        negative = other_topics.sample(len(positive), weights=other_topics['count']).assign(bin_label=0)
        negative.drop(labels=['count'], axis=1, inplace=True)
        datasets.append(pd.concat([positive, negative]))

    # Analyse labels in dataset
    label_ids = {l: i for i, l in enumerate(labels)}

    df['label_id'] = df['label_name'].map(label_ids)

    df['label_name'].value_counts().plot(kind='bar')

    df['label_name'].value_counts()

    return labels, datasets


if __name__ == '__main__':
    load_dataset()

    # gpt_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    # Build classifier pipeline
    # Using configuration from baseline_tuning.ipynb

    gs_results = pd.DataFrame(columns=['params',  # 'vect__analyzer', 'tfidf', 'binarizer', 'scaler', 'clf',
                                       'macro_score', 'mean_std_score', 'mean_fit_time'])

    Path('baseline_binary_tuning').mkdir(parents=True, exist_ok=True)

    for label, dataset in [list(zip(*load_dataset()))[0]]:
        # Train/test split
        training_set = dataset.sample(frac=TRAINING_SET_FRAC)
        test_set = dataset.drop(index=training_set.index)

        X_train, X_test, y_train, y_test = training_set['url'], test_set['url'], training_set['bin_label'], test_set[
            'bin_label']

        gs_pipe = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('scaler', StandardScaler(with_mean=False)),
                            ('clf', LinearSVC(dual="auto"))
                            ])
        parameters = {
            'vect__analyzer': [ngram_analyzer([3]), ngram_analyzer([5]), ngram_analyzer([6]), ngram_analyzer([8]),
                               ngram_analyzer([3, ..., 8])],
        }

        gs = GridSearchCV(gs_pipe, parameters, cv=3, n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)

        gs_results[f'params_{label}'] = gs.cv_results_['params']
        gs_results[f'mean_score_{label}'] = gs.cv_results_['mean_test_score']
        gs_results[f'std_score_{label}'] = gs.cv_results_['std_test_score']
        gs_results[f'mean_fit_time_{label}'] = gs.cv_results_['mean_fit_time']

        # write raw results to file
        pd.DataFrame(gs.cv_results_).to_csv(f'baseline_binary_tuning/gs_results_{label}.csv')

    # mean values
    gs_results['params'] = gs_results['params_0']
    gs_results['macro_score'] = gs_results[[col for col in gs_results.columns if col.startswith('mean_score')]].mean(axis=1)
    gs_results['mean_std_score'] = gs_results[[col for col in gs_results.columns if col.startswith('std_score')]].mean(axis=1)
    gs_results['mean_fit_time'] = gs_results[[col for col in gs_results.columns if col.startswith('mean_fit_time')]] \
        .mean(axis=1)