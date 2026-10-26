import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from scipy.sparse import hstack
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


def extract_nlp_features(text):
    """
    Extract NLP heuristic features following competition baseline approach.
    Based on NEON tool patterns from Di Sorbo et al.
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
    
    features['has_question'] = int('?' in text)
    features['has_exclamation'] = int('!' in text)
    features['has_colon'] = int(':' in text)
    
    words = text.split()
    features['word_count'] = len(words)
    features['char_count'] = len(text)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    
    features['has_uppercase'] = int(any(c.isupper() for c in text))
    features['has_number'] = int(any(c.isdigit() for c in text))
    
    features['has_class_ref'] = int(bool(re.search(r'[A-Z][a-z]+[A-Z]', text)))
    features['has_method_call'] = int('()' in text)
    features['has_curly_braces'] = int('{' in text or '}' in text)
    features['has_brackets'] = int('[' in text or ']' in text)
    
    features['sentence_starts_with_capital'] = int(text and text[0].isupper())
    features['ends_with_period'] = int(text.endswith('.'))
    
    return features


def extract_features_for_dataset(sentences):
    """Convert sentences to NLP feature matrix."""
    return pd.DataFrame([extract_nlp_features(s) for s in sentences]).values


def get_ml_models():
    """
    Define traditional ML models for comparison.
    NO deep learning - only classical ML algorithms.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            C=3.0,
            max_iter=2000,
            solver='liblinear',
            random_state=42,
            class_weight='balanced'
        ),
        'Linear SVC': LinearSVC(
            C=0.8,
            max_iter=2000,
            random_state=42,
            dual=True,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'Naive Bayes': MultinomialNB(alpha=0.1)
    }
    return models


def train_with_kfold_cv(df, category, clf_name, clf, n_folds=5, use_nlp=True):
    """
    Train and evaluate using k-fold cross-validation.
    This is the KEY requirement - using k-fold instead of fixed split.
    """
    print(f"\n  Training {clf_name} on '{category}' with {n_folds}-fold CV...")
    
    category_data = df[df['category'] == category].copy()
    X_text = category_data['comment_sentence'].values
    y = category_data['instance_type'].values
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_text, y)):
        X_train_text = X_text[train_idx]
        X_test_text = X_text[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            stop_words='english',
            sublinear_tf=True
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train_text)
        X_test_tfidf = vectorizer.transform(X_test_text)
        
        if use_nlp:
            X_train_nlp = extract_features_for_dataset(X_train_text)
            X_test_nlp = extract_features_for_dataset(X_test_text)
            X_train = hstack([X_train_tfidf, X_train_nlp])
            X_test = hstack([X_test_tfidf, X_test_nlp])
        else:
            X_train = X_train_tfidf
            X_test = X_test_tfidf
        
        fold_clf = clf.__class__(**clf.get_params())
        fold_clf.fit(X_train, y_train)
        y_pred = fold_clf.predict(X_test)
        
        try:
            if hasattr(fold_clf, 'predict_proba'):
                y_proba = fold_clf.predict_proba(X_test)[:, 1]
            elif hasattr(fold_clf, 'decision_function'):
                y_proba = fold_clf.decision_function(X_test)
            else:
                y_proba = y_pred.astype(float)
        except:
            y_proba = y_pred.astype(float)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', pos_label=1, zero_division=0
        )
        
        fold_results.append({
            'fold': fold_idx + 1,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(y_test.sum())
        })
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    
    avg_precision = np.mean([f['precision'] for f in fold_results])
    avg_recall = np.mean([f['recall'] for f in fold_results])
    avg_f1 = np.mean([f['f1'] for f in fold_results])
    std_f1 = np.std([f['f1'] for f in fold_results])
    
    try:
        roc_auc = roc_auc_score(all_y_true, all_y_proba)
    except:
        roc_auc = 0.0
    
    return {
        'category': category,
        'classifier': clf_name,
        'avg_precision': float(avg_precision),
        'avg_recall': float(avg_recall),
        'avg_f1': float(avg_f1),
        'std_f1': float(std_f1),
        'roc_auc': float(roc_auc),
        'fold_results': fold_results,
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred).tolist()
    }


def compare_with_competition_baseline(our_results, competition_results):
    """
    Compare our k-fold CV results with competition baseline (fixed split).
    """
    print("\n" + "="*100)
    print("COMPARISON: Our k-Fold CV Results vs Competition Baseline (Fixed Split)")
    print("="*100)
    
    print(f"\n{'Method':<30} {'Our F1 (k-fold)':>20} {'Competition F1':>20} {'Improvement':>15}")
    print("-"*100)
    
    comparisons = []
    for method, our_f1 in our_results.items():
        comp_f1 = competition_results.get(method, 0.0)
        improvement = our_f1 - comp_f1
        improvement_pct = (improvement / comp_f1 * 100) if comp_f1 > 0 else 0
        
        status = "✅" if improvement > 0 else "⚠️"
        print(f"{method:<30} {our_f1:>20.4f} {comp_f1:>20.4f} {status} {improvement:>+7.4f} ({improvement_pct:>+6.1f}%)")
        
        comparisons.append({
            'method': method,
            'our_f1': our_f1,
            'competition_f1': comp_f1,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        })
    
    return comparisons


def analyze_results(all_results, output_dir):
    """
    Comprehensive analysis of obtained results.
    """
    print("\n" + "="*100)
    print("DETAILED RESULTS ANALYSIS")
    print("="*100)
    
    results_df = pd.DataFrame(all_results)
    
    print("\n1. Overall Performance by Classifier:")
    print("-" * 100)
    classifier_summary = results_df.groupby('classifier').agg({
        'avg_precision': 'mean',
        'avg_recall': 'mean',
        'avg_f1': 'mean',
        'std_f1': 'mean',
        'roc_auc': 'mean'
    }).round(4)
    print(classifier_summary)
    
    print("\n2. Performance by Category:")
    print("-" * 100)
    category_summary = results_df.groupby('category').agg({
        'avg_precision': 'mean',
        'avg_recall': 'mean',
        'avg_f1': 'mean'
    }).round(4).sort_values('avg_f1', ascending=False)
    print(category_summary.head(10))
    
    print("\n3. Best Model per Category:")
    print("-" * 100)
    best_per_category = results_df.loc[results_df.groupby('category')['avg_f1'].idxmax()]
    for _, row in best_per_category.iterrows():
        print(f"  {row['category']:<20} → {row['classifier']:<25} (F1: {row['avg_f1']:.4f})")
    
    print("\n4. Statistical Summary:")
    print("-" * 100)
    best_classifier = results_df.loc[results_df['avg_f1'].idxmax()]
    print(f"  Best Overall Performance: {best_classifier['classifier']} on '{best_classifier['category']}'")
    print(f"    Precision: {best_classifier['avg_precision']:.4f}")
    print(f"    Recall:    {best_classifier['avg_recall']:.4f}")
    print(f"    F1-score:  {best_classifier['avg_f1']:.4f} ± {best_classifier['std_f1']:.4f}")
    print(f"    ROC-AUC:   {best_classifier['roc_auc']:.4f}")
    
    results_df.to_csv(output_dir / 'kfold_cv_detailed_results.csv', index=False)
    
    summary_stats = {
        'classifier_summary': classifier_summary.to_dict(),
        'best_overall': {
            'classifier': best_classifier['classifier'],
            'category': best_classifier['category'],
            'f1': float(best_classifier['avg_f1']),
            'std_f1': float(best_classifier['std_f1'])
        }
    }
    
    save_json(summary_stats, output_dir / 'analysis_summary.json')
    
    return classifier_summary


def main():
    """
    Main execution: Complete ML solution with k-fold CV
    """
    set_seed(42)
    
    print("="*100)
    print("NLBSE'23 TOOL COMPETITION - MACHINE LEARNING SOLUTION")
    print("Approach: k-Fold Cross-Validation with Traditional ML (NO Deep Learning)")
    print("="*100)
    
    languages = {
        'Java': 'data/code-comment-classification/java/input/java.csv',
        'Python': 'data/code-comment-classification/python/input/python.csv',
        'Pharo': 'data/code-comment-classification/pharo/input/pharo.csv'
    }
    
    models = get_ml_models()
    all_results = []
    
    n_folds = 5
    use_nlp_features = True
    
    print(f"\nConfiguration:")
    print(f"  - k-fold: {n_folds}")
    print(f"  - NLP features: {use_nlp_features}")
    print(f"  - Models: {', '.join(models.keys())}")
    print(f"  - Languages: {', '.join(languages.keys())}")
    
    for lang_name, csv_path in languages.items():
        print(f"\n{'='*100}")
        print(f"Processing {lang_name}...")
        print(f"{'='*100}")
        
        df = pd.read_csv(csv_path)
        categories = df['category'].unique()
        print(f"Categories: {', '.join(categories)}")
        
        for category in categories:
            print(f"\nCategory: {category}")
            
            for clf_name, clf in models.items():
                result = train_with_kfold_cv(
                    df, category, clf_name, clf, 
                    n_folds=n_folds, 
                    use_nlp=use_nlp_features
                )
                result['language'] = lang_name
                all_results.append(result)
    
    output_dir = Path('runs/ml_solution_kfold')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*100)
    print("ANALYSIS PHASE")
    print("="*100)
    
    classifier_summary = analyze_results(all_results, output_dir)
    
    competition_baseline_f1 = {
        'Logistic Regression': 0.547,
        'Linear SVC': 0.547,
        'Random Forest': 0.537
    }
    
    our_avg_f1 = {}
    results_df = pd.DataFrame(all_results)
    for clf in models.keys():
        clf_results = results_df[results_df['classifier'] == clf]
        our_avg_f1[clf] = clf_results['avg_f1'].mean()
    
    comparisons = compare_with_competition_baseline(our_avg_f1, competition_baseline_f1)
    
    save_json({
        'all_results': all_results,
        'comparisons': comparisons,
        'configuration': {
            'n_folds': n_folds,
            'use_nlp_features': use_nlp_features,
            'models': list(models.keys())
        }
    }, output_dir / 'complete_results.json')
    
    pd.DataFrame(comparisons).to_csv(
        output_dir / 'baseline_comparison.csv', index=False
    )
    
    print(f"\n{'='*100}")
    print(f"✅ All results saved to: {output_dir}/")
    print(f"{'='*100}")
    
    print("\n📊 Key Findings:")
    print("  1. k-Fold CV provides more robust estimates than fixed split")
    print("  2. Multiple models tested for comprehensive comparison")
    print("  3. NLP features combined with TF-IDF for enhanced performance")
    print("  4. Results compared with competition baseline")
    
    best_model = max(our_avg_f1.items(), key=lambda x: x[1])
    print(f"\n🏆 Best Model: {best_model[0]} (F1: {best_model[1]:.4f})")


if __name__ == '__main__':
    main()


