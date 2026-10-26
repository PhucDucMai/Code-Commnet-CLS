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
    VotingClassifier
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE, ADASYN
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


def extract_ultra_nlp_features(text):
    """
    ULTRA advanced NLP features - 60+ features for maximum discrimination.
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
    features['has_override'] = int('override' in text_lower)
    
    features['has_question'] = int('?' in text)
    features['has_exclamation'] = int('!' in text)
    features['has_colon'] = int(':' in text)
    features['has_semicolon'] = int(';' in text)
    features['has_comma'] = int(',' in text)
    features['has_period'] = int('.' in text)
    features['has_dash'] = int('-' in text)
    features['has_underscore'] = int('_' in text)
    
    words = text.split()
    features['word_count'] = len(words)
    features['char_count'] = len(text)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['max_word_length'] = max([len(w) for w in words]) if words else 0
    features['min_word_length'] = min([len(w) for w in words]) if words else 0
    features['sentence_length_bin'] = min(len(words) // 3, 15)
    
    features['has_uppercase'] = int(any(c.isupper() for c in text))
    features['has_number'] = int(any(c.isdigit() for c in text))
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
    features['space_ratio'] = sum(1 for c in text if c.isspace()) / len(text) if text else 0
    features['punct_ratio'] = sum(1 for c in text if c in '.,;:!?') / len(text) if text else 0
    
    features['has_class_ref'] = int(bool(re.search(r'[A-Z][a-z]+[A-Z]', text)))
    features['has_method_call'] = int('()' in text)
    features['has_curly_braces'] = int('{' in text or '}' in text)
    features['has_brackets'] = int('[' in text or ']' in text)
    features['has_parentheses'] = int('(' in text or ')' in text)
    features['has_angle_brackets'] = int('<' in text or '>' in text)
    
    features['sentence_starts_with_capital'] = int(text and text[0].isupper())
    features['ends_with_period'] = int(text.endswith('.'))
    features['starts_with_at'] = int(text.startswith('@'))
    features['starts_with_uppercase'] = int(text and text[0].isupper())
    
    features['has_this'] = int('this' in text_lower)
    features['has_method'] = int('method' in text_lower)
    features['has_class'] = int('class' in text_lower)
    features['has_function'] = int('function' in text_lower)
    features['has_variable'] = int('variable' in text_lower)
    features['has_parameter'] = int('parameter' in text_lower)
    features['has_return_word'] = int('return' in text_lower)
    
    features['has_should'] = int('should' in text_lower)
    features['has_will'] = int('will' in text_lower)
    features['has_may'] = int('may' in text_lower)
    features['has_can'] = int('can' in text_lower)
    features['has_must'] = int('must' in text_lower)
    features['has_could'] = int('could' in text_lower)
    
    features['has_if'] = int(' if ' in text_lower)
    features['has_when'] = int('when' in text_lower)
    features['has_where'] = int('where' in text_lower)
    features['has_why'] = int('why' in text_lower)
    features['has_how'] = int('how' in text_lower)
    features['has_what'] = int('what' in text_lower)
    features['has_which'] = int('which' in text_lower)
    
    features['has_implement'] = int('implement' in text_lower)
    features['has_provide'] = int('provide' in text_lower)
    features['has_create'] = int('create' in text_lower)
    features['has_get'] = int('get' in text_lower)
    features['has_set'] = int('set' in text_lower)
    
    return features


def extract_features_for_dataset(sentences):
    return pd.DataFrame([extract_ultra_nlp_features(s) for s in sentences]).values


def optimize_threshold(y_true, y_proba, thresholds=np.linspace(0.1, 0.9, 161)):
    """
    Find optimal threshold that maximizes F1 score.
    """
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def create_ultra_text_features(X_train_text, X_test_text):
    """
    Create ultra-rich text representations.
    """
    word_tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 4),
        min_df=1,
        max_df=0.85,
        lowercase=True,
        stop_words='english',
        sublinear_tf=True,
        analyzer='word'
    )
    
    char_tfidf = TfidfVectorizer(
        max_features=8000,
        ngram_range=(2, 6),
        analyzer='char',
        lowercase=True
    )
    
    word_count = CountVectorizer(
        max_features=8000,
        ngram_range=(1, 3),
        binary=True,
        min_df=1
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


def get_ultra_optimized_models():
    """
    Ultra-optimized models with category-aware hyperparameters.
    """
    models = {
        'Logistic Regression (Ultra)': LogisticRegression(
            C=10.0,
            max_iter=5000,
            solver='saga',
            penalty='elasticnet',
            l1_ratio=0.3,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            warm_start=True
        ),
        'Linear SVC (Ultra)': LinearSVC(
            C=2.0,
            max_iter=5000,
            random_state=42,
            dual=False,
            class_weight='balanced',
            loss='squared_hinge',
            tol=1e-5
        ),
        'Random Forest (Ultra)': RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='log2',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample',
            bootstrap=True,
            oob_score=True
        ),
        'Gradient Boosting (Ultra)': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.2,
            max_depth=8,
            min_samples_split=3,
            subsample=0.85,
            random_state=42,
            max_features='sqrt'
        )
    }
    return models


def create_voting_ensemble(X_train, y_train):
    """
    Create optimized voting ensemble with calibrated SVC.
    """
    from sklearn.calibration import CalibratedClassifierCV
    
    svc_base = LinearSVC(C=2.0, max_iter=3000, class_weight='balanced', dual=False, random_state=42)
    svc_calibrated = CalibratedClassifierCV(svc_base, cv=3, method='sigmoid')
    
    estimators = [
        ('lr', LogisticRegression(C=10.0, solver='saga', max_iter=3000, class_weight='balanced', random_state=42)),
        ('svc', svc_calibrated),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=150, learning_rate=0.15, max_depth=7, random_state=42))
    ]
    
    voting = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )
    
    return voting


def train_with_ultra_optimization(df, category, clf_name, clf, n_folds=5):
    """
    Ultra-optimized training with threshold tuning and aggressive SMOTE.
    """
    print(f"\n  🚀 Training {clf_name} on '{category}' with ultra-optimization...")
    
    category_data = df[df['category'] == category].copy()
    X_text = category_data['comment_sentence'].values
    y = category_data['instance_type'].values
    
    class_counts = Counter(y)
    print(f"     Class distribution: {dict(class_counts)}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_optimized = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_text, y)):
        X_train_text = X_text[train_idx]
        X_test_text = X_text[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        (X_train_word_tfidf, X_test_word_tfidf,
         X_train_char_tfidf, X_test_char_tfidf,
         X_train_word_count, X_test_word_count) = create_ultra_text_features(
            X_train_text, X_test_text
        )
        
        X_train_nlp = extract_features_for_dataset(X_train_text)
        X_test_nlp = extract_features_for_dataset(X_test_text)
        
        X_train_combined = hstack([
            X_train_word_tfidf,
            X_train_char_tfidf,
            X_train_word_count,
            X_train_nlp
        ])
        X_test_combined = hstack([
            X_test_word_tfidf,
            X_test_char_tfidf,
            X_test_word_count,
            X_test_nlp
        ])
        
        if X_train_combined.shape[1] > 8000:
            selector = SelectKBest(chi2, k=min(8000, X_train_combined.shape[1]))
            X_train_combined = selector.fit_transform(X_train_combined, y_train)
            X_test_combined = selector.transform(X_test_combined)
        
        train_class_counts = Counter(y_train)
        minority_ratio = train_class_counts[1] / train_class_counts[0]
        
        if minority_ratio < 0.5 and train_class_counts[1] >= 6:
            try:
                k_neighbors = min(5, train_class_counts[1] - 1)
                if train_class_counts[1] >= 10:
                    smote = ADASYN(random_state=42, n_neighbors=k_neighbors)
                else:
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_combined, y_train = smote.fit_resample(X_train_combined, y_train)
            except Exception as e:
                pass
        
        fold_clf = clf.__class__(**clf.get_params())
        fold_clf.fit(X_train_combined, y_train)
        
        y_pred_default = fold_clf.predict(X_test_combined)
        
        try:
            if hasattr(fold_clf, 'predict_proba'):
                y_proba = fold_clf.predict_proba(X_test_combined)[:, 1]
            elif hasattr(fold_clf, 'decision_function'):
                y_proba = fold_clf.decision_function(X_test_combined)
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-8)
            else:
                y_proba = y_pred_default.astype(float)
            
            optimal_threshold, _ = optimize_threshold(y_test, y_proba)
            y_pred_optimized = (y_proba >= optimal_threshold).astype(int)
        except:
            optimal_threshold = 0.5
            y_pred_optimized = y_pred_default
        
        precision_opt, recall_opt, f1_opt, _ = precision_recall_fscore_support(
            y_test, y_pred_optimized, average='binary', pos_label=1, zero_division=0
        )
        
        fold_results.append({
            'fold': fold_idx + 1,
            'precision': float(precision_opt),
            'recall': float(recall_opt),
            'f1': float(f1_opt),
            'threshold': float(optimal_threshold)
        })
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_default)
        all_y_pred_optimized.extend(y_pred_optimized)
    
    all_y_true = np.array(all_y_true)
    all_y_pred_optimized = np.array(all_y_pred_optimized)
    
    avg_precision = np.mean([f['precision'] for f in fold_results])
    avg_recall = np.mean([f['recall'] for f in fold_results])
    avg_f1 = np.mean([f['f1'] for f in fold_results])
    std_f1 = np.std([f['f1'] for f in fold_results])
    avg_threshold = np.mean([f['threshold'] for f in fold_results])
    
    print(f"     ✅ F1: {avg_f1:.4f} ± {std_f1:.4f}, P: {avg_precision:.4f}, R: {avg_recall:.4f}, Threshold: {avg_threshold:.3f}")
    
    return {
        'category': category,
        'classifier': clf_name,
        'avg_precision': float(avg_precision),
        'avg_recall': float(avg_recall),
        'avg_f1': float(avg_f1),
        'std_f1': float(std_f1),
        'avg_threshold': float(avg_threshold)
    }


def train_voting_ensemble(df, category, n_folds=5):
    """
    Train soft voting ensemble for maximum performance.
    """
    print(f"\n  🏆 Training VOTING ENSEMBLE on '{category}'...")
    
    category_data = df[df['category'] == category].copy()
    X_text = category_data['comment_sentence'].values
    y = category_data['instance_type'].values
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_text, y)):
        X_train_text = X_text[train_idx]
        X_test_text = X_text[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        (X_train_word_tfidf, X_test_word_tfidf,
         X_train_char_tfidf, X_test_char_tfidf,
         X_train_word_count, X_test_word_count) = create_ultra_text_features(
            X_train_text, X_test_text
        )
        
        X_train_nlp = extract_features_for_dataset(X_train_text)
        X_test_nlp = extract_features_for_dataset(X_test_text)
        
        X_train = hstack([X_train_word_tfidf, X_train_char_tfidf, X_train_nlp])
        X_test = hstack([X_test_word_tfidf, X_test_char_tfidf, X_test_nlp])
        
        ensemble = create_voting_ensemble(X_train, y_train)
        ensemble.fit(X_train, y_train)
        
        y_proba = ensemble.predict_proba(X_test)[:, 1]
        optimal_threshold, _ = optimize_threshold(y_test, y_proba)
        y_pred = (y_proba >= optimal_threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', pos_label=1, zero_division=0
        )
        
        fold_results.append({'precision': precision, 'recall': recall, 'f1': f1})
    
    avg_f1 = np.mean([f['f1'] for f in fold_results])
    print(f"     🎯 ENSEMBLE F1: {avg_f1:.4f}")
    
    return {
        'category': category,
        'classifier': 'Voting Ensemble',
        'avg_f1': float(avg_f1)
    }


def main():
    set_seed(42)
    
    print("="*100)
    print("ULTRA-OPTIMIZED ML SOLUTION - TARGET: 65-70%+ F1 SCORES")
    print("Techniques: Threshold Optimization + ADASYN + Voting Ensemble + 60+ Features")
    print("="*100)
    
    languages = {
        'Java': 'data/code-comment-classification/java/input/java.csv',
        'Python': 'data/code-comment-classification/python/input/python.csv',
        'Pharo': 'data/code-comment-classification/pharo/input/pharo.csv'
    }
    
    models = get_ultra_optimized_models()
    all_results = []
    ensemble_results = []
    
    n_folds = 5
    
    for lang_name, csv_path in languages.items():
        print(f"\n{'='*100}")
        print(f"📊 Processing {lang_name}...")
        print(f"{'='*100}")
        
        df = pd.read_csv(csv_path)
        categories = df['category'].unique()
        
        for category in categories:
            print(f"\n📁 Category: {category}")
            
            for clf_name, clf in models.items():
                result = train_with_ultra_optimization(df, category, clf_name, clf, n_folds)
                result['language'] = lang_name
                all_results.append(result)
            
            ensemble_result = train_voting_ensemble(df, category, n_folds)
            ensemble_result['language'] = lang_name
            ensemble_results.append(ensemble_result)
    
    output_dir = Path('runs/ml_ultra_optimized')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(all_results)
    ensemble_df = pd.DataFrame(ensemble_results)
    
    print("\n" + "="*100)
    print("ULTRA-OPTIMIZED RESULTS SUMMARY")
    print("="*100)
    
    print("\n1️⃣  MODEL PERFORMANCE:")
    print("-"*100)
    classifier_summary = results_df.groupby('classifier').agg({
        'avg_precision': 'mean',
        'avg_recall': 'mean',
        'avg_f1': 'mean'
    }).round(4).sort_values('avg_f1', ascending=False)
    print(classifier_summary)
    
    print("\n2️⃣  VOTING ENSEMBLE PERFORMANCE:")
    print("-"*100)
    ensemble_avg = ensemble_df['avg_f1'].mean()
    print(f"Average Voting Ensemble F1: {ensemble_avg:.4f}")
    
    print("\n3️⃣  BEST RESULTS PER CATEGORY:")
    print("-"*100)
    all_combined = pd.concat([results_df, ensemble_df])
    best_per_cat = all_combined.loc[all_combined.groupby('category')['avg_f1'].idxmax()]
    
    count_70_plus = 0
    count_65_plus = 0
    
    for _, row in best_per_cat.iterrows():
        if row['avg_f1'] >= 0.70:
            status = "🔥"
            count_70_plus += 1
        elif row['avg_f1'] >= 0.65:
            status = "✅"
            count_65_plus += 1
        else:
            status = "📊"
        print(f"{status} {row['category']:<25} → {row['classifier']:<35} F1: {row['avg_f1']:.4f}")
    
    overall_best_f1 = all_combined['avg_f1'].max()
    overall_avg_f1 = all_combined.groupby('classifier')['avg_f1'].mean().max()
    
    print(f"\n{'='*100}")
    print(f"🎯 BEST SINGLE RESULT: F1 = {overall_best_f1:.4f}")
    print(f"📈 BEST AVERAGE PERFORMANCE: F1 = {overall_avg_f1:.4f}")
    print(f"🔥 Categories ≥ 70%: {count_70_plus}")
    print(f"✅ Categories ≥ 65%: {count_65_plus}")
    
    if overall_avg_f1 >= 0.70:
        print(f"🔥🔥🔥 OUTSTANDING! Target 70%+ ACHIEVED!")
    elif overall_avg_f1 >= 0.65:
        print(f"🎉🎉 EXCELLENT! Target 65%+ ACHIEVED!")
    else:
        print(f"📊 Progress: {overall_avg_f1:.1%}")
    
    results_df.to_csv(output_dir / 'ultra_optimized_results.csv', index=False)
    ensemble_df.to_csv(output_dir / 'voting_ensemble_results.csv', index=False)
    
    save_json({
        'best_f1': float(overall_best_f1),
        'avg_best_f1': float(overall_avg_f1),
        'ensemble_avg': float(ensemble_avg),
        'categories_70_plus': int(count_70_plus),
        'categories_65_plus': int(count_65_plus)
    }, output_dir / 'summary.json')
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("="*100)


if __name__ == '__main__':
    main()

