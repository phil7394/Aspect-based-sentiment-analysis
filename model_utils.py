import numpy as np
import pandas
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.multiclass import unique_labels


def apply_aspdep_weight(train_df, weight, test_df=None):
    train_text = train_df[' text'].values.astype('U')
    train_aspdep = train_df['asp_dep_words'].values.astype('U')

    text_count_vect = CountVectorizer()
    tr_text_counts = text_count_vect.fit_transform(train_text)
    text_voc = text_count_vect.vocabulary_

    asp_dep_vect = CountVectorizer(vocabulary=text_voc)
    tr_aspdep_counts = asp_dep_vect.fit_transform(train_aspdep)
    tr_count_vec = tr_text_counts + weight * tr_aspdep_counts
    tr_tfidf_vec = TfidfTransformer(use_idf=True).fit_transform(tr_count_vec)

    if test_df is not None:
        test_text = test_df[' text'].values.astype('U')
        test_aspdep = test_df['asp_dep_words'].values.astype('U')
        te_text_counts = text_count_vect.transform(test_text)
        te_aspdep_counts = asp_dep_vect.transform(test_aspdep)
        te_count_vec = te_text_counts + weight * te_aspdep_counts
        te_tfidf_vec = TfidfTransformer(use_idf=True).fit_transform(te_count_vec)
        return te_tfidf_vec

    return tr_tfidf_vec


def extract_aspect_related_words(sdp, ardf):
    print("Extracting aspect related words from text...")
    cols = list(ardf)
    cols.append('asp_dep_words')
    ar_df = pandas.DataFrame(columns=cols)
    count = 0
    for index, row in ardf.iterrows():
        count += 1
        print(count)
        dep_set = set()
        result = list(sdp.raw_parse(row[' text']))
        parse_triples_list = [item for item in result[0].triples()]
        for governor, dep, dependent in parse_triples_list:
            if governor[0] in row[' aspect_term'] or dependent[0] in row[' aspect_term']:
                dep_set.add(governor[0])
                dep_set.add(dependent[0])
        ar_row = [row[c] for c in cols[:-1]]
        ar_row.append(' '.join(list(dep_set)))
        ar_df.loc[len(ar_df.index)] = ar_row
        # print
    return ar_df


# Deprecated: use k_fold_cv()
def get_cv_metrics(text_clf, train_data, train_class, k_split):
    accuracy_scores = cross_val_score(text_clf,  # steps to convert raw messages      into models
                                      train_data,  # training data
                                      train_class,  # training labels
                                      cv=k_split,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                                      scoring='accuracy',  # which scoring metric?
                                      n_jobs=-1,  # -1 = use all cores = faster
                                      )
    cv_predicted = cross_val_predict(text_clf,
                                     train_data,
                                     train_class,
                                     cv=k_split)

    return np.mean(accuracy_scores), classification_report(train_class, cv_predicted)


def over_sample(X_train, y_train, over_sample_size):
    if over_sample_size is not None:
        sample_map = {k: over_sample_size for k in [-1, 0, 1]}
        sm = SMOTE(ratio=sample_map, random_state=0)
    else:
        sm = SMOTE(random_state=0)
    X_res, y_res = sm.fit_sample(X_train, y_train)
    return X_res, y_res


def split_train_test(idx, X, y):
    X_train = X[idx[0]]
    y_train = y[idx[0]]
    X_test = X[idx[1]]
    y_test = y[idx[1]]
    return X_train, y_train, X_test, y_test


def get_report(precision, recall, fscore, support, labels):
    last_line_heading = 'avg / total'
    digits = 2
    target_names = [u'%s' % l for l in labels]
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
    rows = zip(target_names, precision, recall, fscore, support)
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)

    report += u'\n'

    # compute averages
    report += row_fmt.format(last_line_heading,
                             np.average(precision, weights=support),
                             np.average(recall, weights=support),
                             np.average(fscore, weights=support),
                             np.sum(support),
                             width=width, digits=digits)

    return report


def get_train_test_indices(X, y, k, shuffle):
    train_test_indices = []
    skf = StratifiedKFold(n_splits=k, random_state=0, shuffle=shuffle)
    for train_index, test_index in skf.split(X, y):
        train_test_indices.append((train_index, test_index))
    return train_test_indices


def k_fold_cv(clf, X, y, k=10, over_sample_class=False, over_sample_size=None, shuffle=False):
    if X.shape[0] == len(y):
        train_test_indices = get_train_test_indices(X, y, k, shuffle)
        labels = unique_labels(y)

        k_predictions = []
        k_test_indices = []
        for i in range(0, k):
            X_train, y_train, X_test, y_test = split_train_test(train_test_indices[i], X, y)
            if over_sample_class:
                X_train, y_train = over_sample(X_train, y_train, over_sample_size)
            y_pred = clf.fit(X_train, y_train).predict(X_test)
            k_predictions.append(y_pred)
            k_test_indices.append(train_test_indices[i][1])

        k_test_indices = np.concatenate(k_test_indices)
        inv_test_indices = np.empty(len(k_test_indices), dtype=int)
        inv_test_indices[k_test_indices] = np.arange(len(k_test_indices))
        predictions = np.concatenate(k_predictions)[inv_test_indices]
        accuracy = accuracy_score(y, predictions)
        p, r, f, s = precision_recall_fscore_support(y, predictions)
        report = get_report(p, r, f, s, labels)
        return accuracy, report


def read_embeddings(embd_file):
    embd_train_data = []
    embd_train_class = []
    with open(embd_file) as ef:
        embd_list = ef.readline().split(' ')
        embd_train_data.append(embd_list[1:-1])
        embd_train_class.append(embd_list[-1])

    return np.array(embd_train_data), np.array(embd_train_class)
