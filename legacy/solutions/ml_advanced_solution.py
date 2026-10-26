import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    make_scorer,
    f1_score
)
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
import random
import json
warnings.filterwarnings('ignore')


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(data, path: str):
    """Save data as JSON with pretty formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def extract_advanced_nlp_features(text):
    """
    ADVANCED NLP feature extraction with 40+ features
    for maximum discriminative power.
    """
    features = {}
    text_lower = text.lower()
    
    features['has_link'] = int('@link' in text_lower)
    features['has_param'] = int('@param' in text_lower)
    features['has_return'] = int('@return' in text_lower)
    features['has_see'] = int('@see' in text_lower)
    features['has_code'] = int('@code' in text_lower)
    features['has_deprecated'] = int('deprecat' in text_lower)
    features['has_example'] = int('example' in text_lower)
    features['has_note'] = int('note' in text_lower)
    features['has_todo'] = int('todo' in text_lower)
    features['has_author'] = int('author' in text_lower)
    features['has_version'] = int('version' in text_lower)
    features['has_since'] = int('since' in text_lower)
    features['has_throws'] = int('throw' in text_lower)
    features['has_exception'] = int('exception' in text_lower)
    
    features['has_question'] = int('?' in text)
    features['has_exclamation'] = int('!' in text)
    features['has_colon'] = int(':' in text)
    features['has_semicolon'] = int(';' in text)
    features['has_comma'] = int(',' in text)
    features['has_period'] = int('.' in text)
    
    words = text.split()
    features['word_count'] = len(words)
    features['char_count'] = len(text)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['max_word_length'] = max([len(w) for w in words]) if words else 0
    features['sentence_length_bin'] = min(len(words) // 5, 10)
    
    features['has_uppercase'] = int(any(c.isupper() for c in text))
    features['has_number'] = int(any(c.isdigit() for c in text))
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
    
    features['has_class_ref'] = int(bool(re.search(r'[A-Z][a-z]+[A-Z]', text)))
    features['has_method_call'] = int('()' in text)
    features['has_curly_braces'] = int('{' in text or '}' in text)
    features['has_brackets'] = int('[' in text or ']' in text)
    features['has_parentheses'] = int('(' in text or ')' in text)
    
    features['sentence_starts_with_capital'] = int(text and text[0].isupper())
    features['ends_with_period'] = int(text.endswith('.'))
    features['starts_with_at'] = int(text.startswith('@'))
    
    features['has_this'] = int('this' in text_lower)
    features['has_method'] = int('method' in text_lower)
    features['has_class'] = int('class' in text_lower)
    features['has_function'] = int('function' in text_lower)
    features['has_variable'] = int('variable' in text_lower)
    features['has_parameter'] = int('parameter' in text_lower)
    
    features['has_should'] = int('should' in text_lower)
    features['has_will'] = int('will' in text_lower)
    features['has_may'] = int('may' in text_lower)
    features['has_can'] = int('can' in text_lower)
    features['has_must'] = int('must' in text_lower)
    
    features['has_if'] = int(' if ' in text_lower)
    features['has_when'] = int('when' in text_lower)
    features['has_where'] = int('where' in text_lower)
    features['has_why'] = int('why' in text_lower)
    features['has_how'] = int('how' in text_lower)
    features['has_what'] = int('what' in text_lower)
    
    return features


def extract_features_for_dataset(sentences):
    return pd.DataFrame([extract_advanced_nlp_features(s) for s in sentences]).values


def create_advanced_text_features(X_train_text, X_test_text):
    """
    Create multiple text representations and combine them.
    """
    word_tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        lowercase=True,
        stop_words='english',
        sublinear_tf=True,
        analyzer='word'
    )
    
    char_tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(3, 5),
        analyzer='char',
        lowercase=True
    )
    
    word_count = CountVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        binary=True
    )
    
    X_train_word_tfidf = word_tfidf.fit_transform(X_train_text)
    X_test_word_tfidf = word_tfidf.transform(X_test_text)
    
    X_train_char_tfidf = char_tfidf.fit_transform(X_train_text)
    X_test_char_tfidf = char_tfidf.transform(X_test_text)
    
    X_train_word_count = word_count.fit_transform(X_train_text)
    X_test_word_count = word_count.transform(X_test_text)
    
    return (X_train_word_tfidf, X_test_word_tfidf,
            X_train_char_tfidf, X_test_char_tfidf,
            X_train_word_count, X_test_word_count)


def get_optimized_models():
    """
    Optimized models with aggressive hyperparameters for 60-70%+ performance.
    """
    models = {
        'Logistic Regression (Optimized)': LogisticRegression(
            C=5.0,
            max_iter=3000,
            solver='saga',
            penalty='elasticnet',
            l1_ratio=0.5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'Linear SVC (Optimized)': LinearSVC(
            C=1.5,
            max_iter=3000,
            random_state=42,
            dual=False,
            class_weight='balanced',
            loss='squared_hinge'
        ),
        'Random Forest (Optimized)': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'
        ),
        'Gradient Boosting (Optimized)': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.15,
            max_depth=7,
            min_samples_split=4,
            subsample=0.9,
            random_state=42
        )
    }
    return models


def create_ensemble_model():
    """
    Create stacked ensemble for maximum performance.
    """
    base_estimators = [
        ('lr', LogisticRegression(C=5.0, solver='saga', max_iter=2000, class_weight='balanced')),
        ('svc', LinearSVC(C=1.5, max_iter=2000, class_weight='balanced', dual=False)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', n_jobs=-1))
    ]
    
    final_estimator = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=3,
        n_jobs=-1
    )
    
    return stacking


def train_with_advanced_kfold(df, category, clf_name, clf, n_folds=5):
    """
    Advanced k-fold training with feature engineering, selection, and SMOTE.
    """
    print(f"\n  🚀 Training {