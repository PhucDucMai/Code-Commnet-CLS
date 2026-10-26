import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support

from src.data import load_raw_data
from src.labels import LabelManager
from src.utils import set_seed, save_json


def train_and_evaluate(clf_name, base_clf, X_train, y_train, X_test, y_test, label_names):
    print(f"\n{'='*60}")
    print(f"Training {clf_name}...")
    print(f"{'='*60}")
    
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    print("Predicting...")
    try:
        y_pred_proba = clf.predict_proba(X_test)
        if isinstance(y_pred_proba, list):
            y_pred_proba = np.column_stack(y_pred_proba)
    except AttributeError:
        y_pred_proba = clf.decision_function(X_test)
        from scipy.special import expit
        y_pred_proba = expit(y_pred_proba)
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='micro', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    
    per_label_metrics = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    print(f"\n{clf_name} Results (Competition Partition):")
    print(f"  Average Precision (Micro): {precision_micro:.3f}")
    print(f"  Average Recall (Micro):    {recall_micro:.3f}")
    print(f"  Average F1-score (Micro):  {f1_micro:.3f}")
    print(f"\n  Average Precision (Macro): {precision_macro:.3f}")
    print(f"  Average Recall (Macro):    {recall_macro:.3f}")
    print(f"  Average F1-score (Macro):  {f1_macro:.3f}")
    
    per_label_df = pd.DataFrame({
        'label': label_names,
        'precision': per_label_metrics[0],
        'recall': per_label_metrics[1],
        'f1': per_label_metrics[2],
        'support': per_label_metrics[3].astype(int)
    })
    
    outperformed = (per_label_df['f1'] > 0.0).sum()
    print(f"\n  Outperformed Categories: {outperformed}/{len(label_names)}")
    
    return {
        'classifier': clf_name,
        'precision_micro': float(precision_micro),
        'recall_micro': float(recall_micro),
        'f1_micro': float(f1_micro),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'outperformed_categories': f"{outperformed}/{len(label_names)}",
        'per_label': per_label_df
    }


def main():
    set_seed(123)
    
    print("="*80)
    print("REPRODUCING NLBSE'23 COMPETITION BASELINE RESULTS")
    print("="*80)
    print()
    
    print("Loading data with original competition partition...")
    df = load_raw_data('data/raw/sentences.csv')
    
    train_df = df[df['partition'] == 0].reset_index(drop=True)
    test_df = df[df['partition'] == 1].reset_index(drop=True)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print()
    
    label_manager = LabelManager([])
    all_labels = set()
    for label_str in df['labels']:
        labels = label_manager.parse_label_string(label_str)
        all_labels.update(labels)
    
    label_manager = LabelManager(list(all_labels))
    print(f"Number of labels: {label_manager.get_num_labels()}")
    print(f"Labels: {label_manager.label_names}")
    print()
    
    train_labels = label_manager.encode([
        label_manager.parse_label_string(s) for s in train_df['labels']
    ])
    test_labels = label_manager.encode([
        label_manager.parse_label_string(s) for s in test_df['labels']
    ])
    
    print("Extracting TF-IDF features...")
    print("  Parameters: max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.95")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        lowercase=True,
        stop_words='english'
    )
    
    X_train = vectorizer.fit_transform(train_df['sentence'])
    X_test = vectorizer.transform(test_df['sentence'])
    
    print(f"  Feature dimension: {X_train.shape[1]}")
    print()
    
    results = []
    
    logreg = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        random_state=123
    )
    result = train_and_evaluate(
        "Logistic Regression", logreg, X_train, train_labels, X_test, test_labels,
        label_manager.label_names
    )
    results.append(result)
    
    svm = LinearSVC(
        C=1.0,
        max_iter=1000,
        random_state=123
    )
    result = train_and_evaluate(
        "Linear SVC", svm, X_train, train_labels, X_test, test_labels,
        label_manager.label_names
    )
    results.append(result)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=123,
        n_jobs=-1
    )
    result = train_and_evaluate(
        "Random Forest", rf, X_train, train_labels, X_test, test_labels,
        label_manager.label_names
    )
    results.append(result)
    
    print("\n" + "="*80)
    print("COMPARISON WITH COMPETITION RESULTS (TABLE III)")
    print("="*80)
    print()
    
    comparison_df = pd.DataFrame([
        {
            'Method': 'Logistic Regression (Paper)',
            'Precision': 0.540,
            'Recall': 0.560,
            'F1-score': 0.547,
            'Categories': '19/19'
        },
        {
            'Method': 'Logistic Regression (Ours)',
            'Precision': results[0]['precision_macro'],
            'Recall': results[0]['recall_macro'],
            'F1-score': results[0]['f1_macro'],
            'Categories': results[0]['outperformed_categories']
        },
        {
            'Method': '',
            'Precision': None,
            'Recall': None,
            'F1-score': None,
            'Categories': None
        },
        {
            'Method': 'Linear SVC (Paper)',
            'Precision': 0.542,
            'Recall': 0.558,
            'F1-score': 0.547,
            'Categories': '18/19'
        },
        {
            'Method': 'Linear SVC (Ours)',
            'Precision': results[1]['precision_macro'],
            'Recall': results[1]['recall_macro'],
            'F1-score': results[1]['f1_macro'],
            'Categories': results[1]['outperformed_categories']
        },
        {
            'Method': '',
            'Precision': None,
            'Recall': None,
            'F1-score': None,
            'Categories': None
        },
        {
            'Method': 'Random Forest (Paper)',
            'Precision': 0.661,
            'Recall': 0.479,
            'F1-score': 0.537,
            'Categories': '17/19'
        },
        {
            'Method': 'Random Forest (Ours)',
            'Precision': results[2]['precision_macro'],
            'Recall': results[2]['recall_macro'],
            'F1-score': results[2]['f1_macro'],
            'Categories': results[2]['outperformed_categories']
        },
    ])
    
    print(comparison_df.to_string(index=False))
    print()
    
    Path('runs/competition_reproduction').mkdir(parents=True, exist_ok=True)
    
    for result in results:
        clf_name = result['classifier'].lower().replace(' ', '_')
        output_dir = f"runs/competition_reproduction/{clf_name}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        result['per_label'].to_csv(f"{output_dir}/metrics_per_label.csv", index=False)
        
        summary = {k: v for k, v in result.items() if k != 'per_label'}
        save_json(summary, f"{output_dir}/metrics_summary.json")
        
        print(f"Saved results to: {output_dir}/")
    
    comparison_df.to_csv('runs/competition_reproduction/comparison_table.csv', index=False)
    print(f"\nSaved comparison table to: runs/competition_reproduction/comparison_table.csv")
    
    print("\n" + "="*80)
    print("REPRODUCTION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
