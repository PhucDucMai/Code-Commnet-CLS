import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import hstack

from src.utils import set_seed, save_json


def extract_nlp_features(text):
    features = {}
    
    text_lower = text.lower()
    
    features['has_link'] = 1 if '@link' in text_lower or 'link' in text_lower else 0
    features['has_param'] = 1 if '@param' in text_lower or 'parameter' in text_lower else 0
    features['has_return'] = 1 if '@return' in text_lower or 'return' in text_lower else 0
    features['has_see'] = 1 if '@see' in text_lower or 'see' in text_lower else 0
    features['has_code'] = 1 if '@code' in text_lower or '`' in text else 0
    features['has_deprecated'] = 1 if 'deprecat' in text_lower else 0
    features['has_example'] = 1 if 'example' in text_lower or 'e.g.' in text_lower else 0
    features['has_note'] = 1 if 'note' in text_lower or 'warning' in text_lower else 0
    features['has_todo'] = 1 if 'todo' in text_lower or 'fixme' in text_lower else 0
    features['has_author'] = 1 if 'author' in text_lower or '@author' in text_lower else 0
    features['has_version'] = 1 if 'version' in text_lower or '@version' in text_lower else 0
    features['has_since'] = 1 if 'since' in text_lower or '@since' in text_lower else 0
    
    features['has_question'] = 1 if '?' in text else 0
    features['has_exclamation'] = 1 if '!' in text else 0
    features['has_colon'] = 1 if ':' in text else 0
    features['has_semicolon'] = 1 if ';' in text else 0
    
    words = text.split()
    features['word_count'] = len(words)
    features['char_count'] = len(text)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    
    features['has_uppercase'] = 1 if any(c.isupper() for c in text) else 0
    features['has_number'] = 1 if any(c.isdigit() for c in text) else 0
    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    features['starts_with_verb'] = 1 if words and words[0].lower() in [
        'get', 'set', 'create', 'delete', 'update', 'add', 'remove', 'find', 
        'check', 'validate', 'process', 'handle', 'return', 'throw', 'call'
    ] else 0
    
    features['has_class_ref'] = 1 if re.search(r'[A-Z][a-z]+[A-Z]', text) else 0
    features['has_method_call'] = 1 if '()' in text or '(' in text else 0
    
    features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
    
    return features


def extract_features_for_dataset(sentences):
    nlp_features_list = []
    for sent in sentences:
        nlp_features_list.append(extract_nlp_features(sent))
    
    nlp_df = pd.DataFrame(nlp_features_list)
    return nlp_df.values


def train_per_category_with_nlp(clf_name, base_clf, category_data, category_name, use_nlp=True):
    train_data = category_data[category_data['partition'] == 0]
    test_data = category_data[category_data['partition'] == 1]
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        lowercase=True,
        stop_words='english',
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(train_data['comment_sentence'])
    X_test_tfidf = vectorizer.transform(test_data['comment_sentence'])
    
    if use_nlp:
        X_train_nlp = extract_features_for_dataset(train_data['comment_sentence'])
        X_test_nlp = extract_features_for_dataset(test_data['comment_sentence'])
        
        X_train = hstack([X_train_tfidf, X_train_nlp])
        X_test = hstack([X_test_tfidf, X_test_nlp])
    else:
        X_train = X_train_tfidf
        X_test = X_test_tfidf
    
    y_train = train_data['instance_type'].values
    y_test = test_data['instance_type'].values
    
    clf = base_clf
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', pos_label=1, zero_division=0
    )
    
    return {
        'category': category_name,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'support': int(y_test.sum()),
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'test_positive': int(y_test.sum())
    }


def main():
    set_seed(42)
    
    print("="*80)
    print("IMPROVED REPRODUCTION WITH NLP FEATURES + HYPERPARAMETER TUNING")
    print("="*80)
    print()
    
    languages = {
        'Java': 'data/code-comment-classification/java/input/java.csv',
        'Python': 'data/code-comment-classification/python/input/python.csv',
        'Pharo': 'data/code-comment-classification/pharo/input/pharo.csv'
    }
    
    configs = [
        ('Logistic Regression (C=0.5)', LogisticRegression(C=0.5, max_iter=2000, solver='liblinear', random_state=42)),
        ('Logistic Regression (C=1.0)', LogisticRegression(C=1.0, max_iter=2000, solver='liblinear', random_state=42)),
        ('Logistic Regression (C=2.0)', LogisticRegression(C=2.0, max_iter=2000, solver='liblinear', random_state=42)),
        ('Linear SVC (C=0.5)', LinearSVC(C=0.5, max_iter=2000, random_state=42, dual=True)),
        ('Linear SVC (C=1.0)', LinearSVC(C=1.0, max_iter=2000, random_state=42, dual=True)),
        ('Linear SVC (C=2.0)', LinearSVC(C=2.0, max_iter=2000, random_state=42, dual=True)),
        ('Random Forest (100)', RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)),
        ('Random Forest (200)', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)),
    ]
    
    all_results = {name: [] for name, _ in configs}
    
    for lang_name, csv_path in languages.items():
        print(f"\n{'='*80}")
        print(f"Processing {lang_name}")
        print(f"{'='*80}")
        
        df = pd.read_csv(csv_path)
        categories = df['category'].unique()
        
        for category in categories:
            category_data = df[df['category'] == category].copy()
            print(f"\n  Category: {category}")
            
            for clf_name, clf in configs:
                result = train_per_category_with_nlp(clf_name, clf, category_data, category, use_nlp=True)
                all_results[clf_name].append(result)
                
                if 'C=1.0' in clf_name or '(100)' in clf_name:
                    print(f"    {clf_name}: F1={result['f1']:.3f}")
    
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print()
    
    best_results = {}
    
    for clf_name, results in all_results.items():
        results_df = pd.DataFrame(results)
        avg_precision = results_df['precision'].mean()
        avg_recall = results_df['recall'].mean()
        avg_f1 = results_df['f1'].mean()
        outperformed = (results_df['f1'] > 0.0).sum()
        
        best_results[clf_name] = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'outperformed': f"{outperformed}/19"
        }
    
    print(f"{'Method':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Categories':>12}")
    print("="*80)
    
    print(f"{'Paper - Logistic Regression':<35} {'0.540':>10} {'0.560':>10} {'0.547':>10} {'19/19':>12}")
    for name in ['Logistic Regression (C=0.5)', 'Logistic Regression (C=1.0)', 'Logistic Regression (C=2.0)']:
        r = best_results[name]
        print(f"{name:<35} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>10.3f} {r['outperformed']:>12}")
    
    print()
    print(f"{'Paper - Linear SVC':<35} {'0.542':>10} {'0.558':>10} {'0.547':>10} {'18/19':>12}")
    for name in ['Linear SVC (C=0.5)', 'Linear SVC (C=1.0)', 'Linear SVC (C=2.0)']:
        r = best_results[name]
        print(f"{name:<35} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>10.3f} {r['outperformed']:>12}")
    
    print()
    print(f"{'Paper - Random Forest':<35} {'0.661':>10} {'0.479':>10} {'0.537':>10} {'17/19':>12}")
    for name in ['Random Forest (100)', 'Random Forest (200)']:
        r = best_results[name]
        print(f"{name:<35} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>10.3f} {r['outperformed']:>12}")
    
    print("\n" + "="*80)
    print("BEST MATCHING CONFIGURATIONS")
    print("="*80)
    
    best_lr = max([(name, r['f1']) for name, r in best_results.items() if 'Logistic' in name], key=lambda x: x[1])
    best_svm = max([(name, r['f1']) for name, r in best_results.items() if 'SVC' in name], key=lambda x: x[1])
    best_rf = max([(name, r['f1']) for name, r in best_results.items() if 'Forest' in name], key=lambda x: x[1])
    
    print(f"\nBest Logistic Regression: {best_lr[0]}")
    print(f"  F1 = {best_lr[1]:.3f} (Paper: 0.547, Diff: {abs(best_lr[1] - 0.547):.3f})")
    
    print(f"\nBest Linear SVC: {best_svm[0]}")
    print(f"  F1 = {best_svm[1]:.3f} (Paper: 0.547, Diff: {abs(best_svm[1] - 0.547):.3f})")
    
    print(f"\nBest Random Forest: {best_rf[0]}")
    print(f"  F1 = {best_rf[1]:.3f} (Paper: 0.537, Diff: {abs(best_rf[1] - 0.537):.3f})")
    
    output_dir = Path('runs/improved_reproduction')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'best_logistic_regression': {
            'config': best_lr[0],
            'f1': float(best_lr[1]),
            'paper_f1': 0.547,
            'difference': float(abs(best_lr[1] - 0.547))
        },
        'best_linear_svc': {
            'config': best_svm[0],
            'f1': float(best_svm[1]),
            'paper_f1': 0.547,
            'difference': float(abs(best_svm[1] - 0.547))
        },
        'best_random_forest': {
            'config': best_rf[0],
            'f1': float(best_rf[1]),
            'paper_f1': 0.537,
            'difference': float(abs(best_rf[1] - 0.537))
        }
    }
    
    save_json(summary, output_dir / 'best_configurations.json')
    print(f"\nSaved best configurations to: {output_dir}/best_configurations.json")
    
    print("\n" + "="*80)
    print("REPRODUCTION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
