import numpy as np
import pandas
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

import PreProcessor
import model_utils
from stacked_generalization import StackedGeneralizer
from standard_deviation_clf import StandardDeviationClassifier


def train_MultinomialNB(filePath):
    '''TRAINING'''
    train_df = pandas.read_csv(filePath, sep='\t')
    train_class = train_df[' class'].as_matrix()
    # for i in range(0, 15, 1):
    # train_data = model_utils.apply_aspdep_weight(train_df, 0.1 * i)
    train_data = model_utils.apply_aspdep_weight(train_df, 0.6)
    text_clf = MultinomialNB(alpha=0.6, fit_prior=True, class_prior=None).fit(train_data,
                                                                              train_class)  # 0.3, 0.6   Accuracy:  0.7375251109738484
    joblib.dump(text_clf, 'model_dumps/Multinomial_nb_model.pkl')

    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.k_fold_cv(text_clf, train_data, train_class, k=10, over_sample_class=True,
                                                 shuffle=True)
    # print("asp_wt: {}".format(0.1 * i))
    print("Accuracy: ", accuracy)
    print(clf_report)


def train_BernoulliNB(filePath):
    '''TRAINING'''
    train_df = pandas.read_csv(filePath, sep='\t')
    train_class = train_df[' class'].as_matrix()
    # for i in range(0, 15, 1):
    #     train_data = model_utils.apply_aspdep_weight(train_df, 0.1 * i)
    train_data = model_utils.apply_aspdep_weight(train_df, 0.5)
    text_clf = BernoulliNB(alpha=0.1, fit_prior=True, class_prior=None).fit(train_data,
                                                                            train_class)  # 1.2 Accuracy:  0.7036196690112986
    joblib.dump(text_clf, 'model_dumps/data_1/wt_aspect/Bernoulli_nb_model.pkl')

    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.k_fold_cv(text_clf, train_data, train_class, k=10, over_sample_class=True,
                                                 shuffle=True)
    # print("asp_wt: {}".format(0.1 * i))
    print("Accuracy: ", accuracy)
    print(clf_report)


def train_SGD(filePath):
    '''TRAINING'''
    train_df = pandas.read_csv(filePath, sep='\t')
    train_class = train_df[' class'].as_matrix()
    # for i in range(0, 15, 1):
    #     train_data = model_utils.apply_aspdep_weight(train_df, 0.1 * i)
    train_data = model_utils.apply_aspdep_weight(train_df, 1.1)
    text_clf = linear_model.SGDClassifier(loss='squared_loss', penalty='l2', alpha=1e-3, random_state=607,
                                          max_iter=20, tol=1e-2).fit(train_data,
                                                                     train_class)  # Accuracy:  0.7710460732026527
    joblib.dump(text_clf, 'model_dumps/data_1/wt_aspect/SGD_model.pkl')

    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.k_fold_cv(text_clf, train_data, train_class, k=10, over_sample_class=True,
                                                 shuffle=True)
    # print("asp_wt: {}".format(0.1 * i))
    print("Accuracy: ", accuracy)
    print(clf_report)


def train_SVC(filePath):
    '''TRAINING'''
    train_df = pandas.read_csv(filePath, sep='\t')
    # train_data, train_class = model_utils.read_embeddings(filePath)
    train_class = train_df[' class'].as_matrix()
    for i in range(15, 30, 1):
        train_data = model_utils.apply_aspdep_weight(train_df, 0.1 * i)
    # train_data = model_utils.apply_aspdep_weight(train_df, 1.7)
        text_clf = SVC(C=1, cache_size=2000, class_weight=None, coef0=0.0,
                       decision_function_shape='ovr', degree=0, gamma=0.9, kernel='rbf',
                       max_iter=-1, probability=False, random_state=None, shrinking=True,
                       tol=0.003, verbose=False).fit(train_data, train_class)
        joblib.dump(text_clf, 'model_dumps/data_2/wt_aspect/SVC_model.pkl')  # Accuracy: ', 0.7505889749930229

        """PERFORMANCE EVALUATION"""
        os_size = 2000
        accuracy, clf_report = model_utils.k_fold_cv(text_clf, train_data, train_class, k=10, over_sample_class=True, over_sample_size=os_size,
                                                     shuffle=True)
        print("asp_wt: {}".format(0.1 * i))
        print("Accuracy: ", accuracy)
        print(clf_report)


def train_XGBClassifier(filePath):
    '''TRAINING'''
    train_df = pandas.read_csv(filePath, sep='\t')
    train_class = train_df[' class'].as_matrix()
    # for i in range(0, 15, 1):
    #     train_data = model_utils.apply_aspdep_weight(train_df, 0.1 * i)
    train_data = model_utils.apply_aspdep_weight(train_df, 0.6)
    text_clf = XGBClassifier(learning_rate=1.1, n_estimators=20, max_depth=60,
                             min_child_weight=6, gamma=0.5, subsample=1.0, colsample_bytree=1.0,
                             objective='binary:logistic', scale_pos_weight=1,
                             silent=False).fit(train_data, train_class)
    joblib.dump(text_clf, 'model_dumps/data_1/wt_aspect/XGB_model.pkl')

    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.k_fold_cv(text_clf, train_data, train_class, k=10, over_sample_class=True,
                                                 shuffle=True)
    # print("asp_wt: {}".format(0.1 * i))
    print("Accuracy: ", accuracy)
    print(clf_report)


def train_RF(filePath):
    '''TRAINING'''
    train_df = pandas.read_csv(filePath, sep='\t')
    train_class = train_df[' class'].as_matrix()
    # for i in range(0, 15, 1):
    #     train_data = model_utils.apply_aspdep_weight(train_df, 0.1 * i)
    train_data = model_utils.apply_aspdep_weight(train_df, 1.0)

    # for estimators in [200,300,400]:
    #     for maxDepth in range (160,191,10):
    text_clf = RandomForestClassifier(n_estimators=400, max_depth=190, random_state=607, n_jobs=-1).fit(train_data,
                                                                                                        train_class)
    joblib.dump(text_clf, 'model_dumps/data_1/wt_aspect/RandomForest_model.pkl')

    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.k_fold_cv(text_clf, train_data, train_class, k=10, over_sample_class=True,
                                                 shuffle=True)
    # print("Accuracy: ", accuracy, "Estimators: ", 400, "Max Depth: ", 190)
    # print("asp_wt: {}".format(0.1 * i))
    print("Accuracy: ", accuracy)
    print(clf_report)


def train_polarity_clf(filePath):
    train_df = pandas.read_csv(filePath, sep='\t')
    train_class = train_df[' class'].as_matrix()
    train_data = train_df['opin_polarity'].as_matrix().reshape(-1, 1)
    text_clf = StandardDeviationClassifier().fit(train_data, train_class)  # Accuracy: 0.5192921161422769
    joblib.dump(text_clf, 'model_dumps/data_1/wt_aspect/Polarity_model.pkl')

    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.k_fold_cv(text_clf, train_data, train_class, k=10, over_sample_class=True,
                                                 shuffle=True)
    print("Accuracy: ", accuracy)
    print(clf_report)


def train_ET(filePath):
    '''TRAINING'''
    train_df = pandas.read_csv(filePath, sep='\t')
    train_class = train_df[' class'].as_matrix()
    # for i in range(0, 15, 1):
    #     train_data = model_utils.apply_aspdep_weight(train_df, 0.1 * i)
    train_data = model_utils.apply_aspdep_weight(train_df, 1.3)
    text_clf = ExtraTreesClassifier(n_estimators=120, max_depth=127, random_state=0, n_jobs=-1).fit(train_data,
                                                                                                    train_class)
    joblib.dump(text_clf, 'model_dumps/data_1/wt_aspect/ExtraTrees_model.pkl')
    # 'Accuracy: ', 0.7410500269345847
    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.k_fold_cv(text_clf, train_data, train_class, k=10, over_sample_class=True,
                                                 shuffle=True)
    # print("asp_wt: {}".format(0.1 * i))
    print("Accuracy: ", accuracy)
    print(clf_report)


# def train_gcForest(filePath):


def train_StackedGeneralizer(filePath):
    """TRAINING"""
    train_df = pandas.read_csv(filePath, sep='\t')
    train_class = train_df[' class'].as_matrix()
    # for i in range(10, 21, 1):
    #     train_data = model_utils.apply_aspdep_weight(train_df, 0.1 * i)
    train_data = model_utils.apply_aspdep_weight(train_df, 1.8)

    base_models = [joblib.load('model_dumps/data_1/wt_aspect/SVC_model.pkl'),
                   joblib.load('model_dumps/data_1/wt_aspect/ExtraTrees_model.pkl')]
    # define blending model
    blending_model = LogisticRegression(random_state=1)

    # initialize multi-stage model
    sg = StackedGeneralizer(base_models, blending_model, n_folds=10, verbose=False).fit(train_data, train_class)

    joblib.dump(sg, 'model_dumps/data_1/wt_aspect/Stacked_model.pkl')

    """PERFORMANCE EVALUATION"""
    accuracy, clf_report = model_utils.k_fold_cv(sg, train_data, train_class, k=10, over_sample_class=True,
                                                 shuffle=True)
    # print("asp_wt: {}".format(0.1 * i))
    print("Accuracy: ", accuracy)  # Accuracy:  0.7685515376742712
    print(clf_report)


# def train_VotingClassifier(filePath):
#     """TRAINING"""
#     train_df = pandas.read_csv(filePath, sep='\t')
#     train_df = model_utils.oversample_neutral_class(train_df)
#     train_class = train_df[' class'].as_matrix()

#     train_data = model_utils.apply_aspdep_weight(train_df, 0.3)
#     clf1 = MultinomialNB(alpha=0.6, fit_prior=True, class_prior=None)
#     clf2 = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=607,
#                                               max_iter=1000000, tol=1e-2)
#     clf3 = BernoulliNB(alpha=1.2, fit_prior=True, class_prior=None)
#     eclf1 = VotingClassifier(estimators=[('mNB', clf1), ('sgd', clf2), ('bNB', clf3)], voting='hard')
#     eclf2 = VotingClassifier(estimators=[('mNB', clf1), ('sgd', clf2), ('bNB', clf3)], voting='soft')
#     eclf3 = VotingClassifier(estimators=[('mNB', clf1), ('sgd', clf2), ('bNB', clf3)], voting='soft', weights=[2,2,1], flatten_transform=True)
#     eclf1.fit(train_data, train_class)
#     eclf2.fit(train_data, train_class)
#     eclf3.fit(train_data, train_class)
# #     joblib.dump(sg, 'Stacked_model.pkl')
#     """PERFORMANCE EVALUATION"""
#     accuracy, clf_report = model_utils.get_cv_metrics(eclf1, train_data, train_class, k_split=10)
#     print("Accuracy: ", accuracy) #Accuracy: 73.06
#     print(clf_report)
#     accuracy, clf_report = model_utils.get_cv_metrics(eclf2, train_data, train_class, k_split=10)
#     print("Accuracy: ", accuracy) #Accuracy: 71.58
#     print(clf_report)
#     accuracy, clf_report = model_utils.k_fold_cv(text_clf, train_data, train_class, k=10, over_sample_class=True, shuffle=True)
#     print("Accuracy: ", accuracy) #Accuracy: 72.16
#     print(clf_report)

def hyperparam_tuning_MultinomialNB():
    """HYPER-PARAMETER TUNING"""
    clf = joblib.load('Multinomial_nb_model.pkl')

    train_df = pandas.read_csv('out_data_1/data_1_sw.csv', sep='\t')
    train_data = model_utils.apply_aspdep_weight(train_df, 0.9)
    train_class = train_df[' class'].as_matrix()
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__fit_prior': (True, False),
                  'clf__alpha': np.arange(0.0, 1.1, 0.1)}
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_data, train_class)
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def hyperparam_tuning_XGB():
    """HYPER-PARAMETER TUNING"""
    clf = XGBClassifier()

    train_df = pandas.read_csv('out_data_1/data_1_sw.csv', sep='\t')
    train_data = model_utils.apply_aspdep_weight(train_df, 0.8)
    train_class = train_df[' class'].as_matrix()

    parameters = {
        'learning_rate': np.arange(1.1, 1.2, 0.1).tolist(),
        'gamma': np.arange(0.5, 0.6, 0.1).tolist(),
        'objective': ['binary:logistic'],
        'n_estimators': np.arange(20, 80, 10).tolist(),
        'max_depth': np.arange(60, 70, 10).tolist(),
    }
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1, verbose=1)
    gs_clf = gs_clf.fit(train_data, train_class)
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def hyperparam_tuning_ETC():
    """HYPER-PARAMETER TUNING"""
    clf = ExtraTreesClassifier()

    train_df = pandas.read_csv('out_data_1/data_1_sw.csv', sep='\t')
    train_data = model_utils.apply_aspdep_weight(train_df, 0.8)
    train_class = train_df[' class'].as_matrix()
    parameters = {
        'n_estimators': np.arange(150, 160, 1).tolist(),
        'max_depth': np.arange(120, 130, 1).tolist(),
        'random_state': np.arange(0, 1, 1).tolist()
    }
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1, verbose=1)
    gs_clf = gs_clf.fit(train_data, train_class)
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def hyperparam_tuning_SVC():
    """HYPER-PARAMETER TUNING"""
    clf = SVC()

    train_df = pandas.read_csv('out_data_1/data_1_sw.csv', sep='\t')
    train_data = model_utils.apply_aspdep_weight(train_df, 0.8)
    train_class = train_df[' class'].as_matrix()
    parameters = {
        'C': np.arange(1, 5, 1).tolist(),
        'kernel': ['rbf', 'poly'],  # precomputed,'poly', 'sigmoid'
        'degree': np.arange(0, 3, 1).tolist(),
        'gamma': np.arange(0.0, 1.0, 0.1).tolist(),
        'coef0': np.arange(0.0, 1.0, 0.1).tolist(),
        'shrinking': [True],
        'probability': [False],
        'tol': np.arange(0.001, 0.01, 0.001).tolist(),
        'cache_size': [2000],
        'class_weight': [None],
        'verbose': [False],
        'max_iter': [-1],
        'random_state': [None],
    }
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_data, train_class)
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def hyperparam_tuning_SGD():
    """HYPER-PARAMETER TUNING"""
    clf = linear_model.SGDClassifier()

    train_df = pandas.read_csv('out_data_1/data_1_sw.csv', sep='\t')
    train_data = model_utils.apply_aspdep_weight(train_df, 0.8)
    train_class = train_df[' class'].as_matrix()
    parameters = {'loss': ['hinge', 'huber', 'squared_loss'],
                  'max_iter': np.arange(20, 100, 50),
                  'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                  'alpha': np.arange(0.001, 1, 0.5)}
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_data, train_class)
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def final_testing(text_clf, test_input_file, test_preproc_file, test_predict_file,
                  asp_wt):  # TODO test, oversample and save models
    """TESTING"""
    preproc_args = ["-i", test_input_file,
                    "-o", test_preproc_file,
                    "-sw", "y",
                    "-pu", "y",
                    "-lo", "y",
                    "-ad", "y"]
    PreProcessor.main(preproc_args)

    test_df = pandas.read_csv(test_preproc_file, sep='\t')
    test_data = model_utils.apply_aspdep_weight(test_df, asp_wt)

    predicted = text_clf.predict(test_data)
    print(predicted)
    with open(test_predict_file, 'w') as res_file:
        for doc, y_pred in zip(test_df['example_id'].as_matrix(), predicted):
            print("%r ;; %s" % (str(doc), y_pred))
            res_file.write("%r ;; %s\n" % (str(doc), y_pred))


if __name__ == '__main__':
    """===============================TESTING==========================================="""
    # parser = argparse.ArgumentParser(description='Aspect-based sentiment classifier')
    # optional = parser._action_groups.pop()
    # required = parser.add_argument_group('required arguments')
    # required.add_argument('-d', '--data', help='data set (1/2)', choices=['1', '2'], required=True)
    # required.add_argument('-i', '--input', help='path to input test data file', required=True)
    # required.add_argument('-o', '--output', help='path to output processed test data file', required=True)
    # required.add_argument('-r', '--result', help='path to predictions file', required=True)
    #
    # parser._action_groups.append(optional)
    # args = vars(parser.parse_args())
    #
    # if args['data'] == 1:
    #     clf = joblib.load('model_dumps/data_1/wt_aspect/Multinomial_nb_model.pkl')
    #     asp_wt = 0.7
    # else:
    #     clf = joblib.load('model_dumps/data_2/wt_aspect/Bernoulli_nb_model.pkl')
    #     asp_wt = 0.5
    # final_testing(clf, args['input'], args['output'], args['result'])

    """===============================TRAINING==========================================="""
    # fileLists = ['out_data_1/test_data_1_sw.csv']
    # for fileno, filePath in enumerate(fileLists):
    filePath = 'out_data_2/data_2_sw.csv'
    # filePath = 'embedding/data_set_1/improvedvec.txt'
    # print("Multinomial NB")
    # train_MultinomialNB(filePath)
    # print("Bernoulli NB ")
    # train_BernoulliNB(filePath)
    # print("SGD ")
    # train_SGD(filePath)
    print("SVC ")
    train_SVC(filePath)
    # print("XGBClassifier ")
    # train_XGBClassifier(filePath)
    # print("Random Forest")
    # train_RF(filePath)
    # print("Extra Tree ")
    # train_ET(filePath)
    # print("Stacked Generalizer")
    # train_StackedGeneralizer(filePath)