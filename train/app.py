import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
import joblib
from src.data_utils import load_data, get_features, get_target, numerical_features, categorical_features
from src.utils import generate_confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# MLflow setup
mlflow.set_experiment("F1_Tire_Deg_Experiment")  # Generic name

df = load_data()

# Add lap_count if missing (simulate stint: 1-70 laps/race)
if 'lap_count' not in df.columns:
    df['lap_count'] = np.random.randint(1, 71, len(df))  # Fix: Real data would have this from telemetry
    mlflow.log_param("added_lap_count", True)  # Log for repro

# Add interaction feature for cumulative deg (lap * force amps wear)
df['lap_force'] = df['lap_count'] * df['force_on_tire']  # New eng; boosts rigor
numerical_features.append('lap_force')  # Update list

# Split
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df['Track'])  # Stratify by track for generalization
train_df = df.loc[train_idx]
test_df = df.loc[test_idx]

# Preprocess
X_train_num, X_train_cat = get_features(train_df)
X_test_num, X_test_cat = get_features(test_df)
y_train = get_target(train_df)
y_test = get_target(test_df)

encoder = OneHotEncoder(sparse_output=False, drop='first')
X_train_cat_encoded = encoder.fit_transform(X_train_cat)
X_test_cat_encoded = encoder.transform(X_test_cat)

X_train = np.hstack((X_train_num, X_train_cat_encoded))
X_test = np.hstack((X_test_num, X_test_cat_encoded))

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_encoded)

params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'seed': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 10,
    'tree_method': 'hist'
}

with mlflow.start_run(run_name="XGBoost_Classifier_POC"):
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(X_train_scaled, y_train_encoded, sample_weight=sample_weights,
                  eval_set=[(X_test_scaled, y_test_encoded)], verbose=False)
    
    y_pred_encoded = xgb_model.predict(X_test_scaled)
    y_probs = xgb_model.predict_proba(X_test_scaled)
    
    class_report = classification_report(y_test_encoded, y_pred_encoded, target_names=le.classes_, output_dict=True, zero_division=0)
    auc_roc = roc_auc_score(y_test_encoded, y_probs, multi_class='ovr', average='weighted')
    
    metrics = {
        'accuracy': class_report['accuracy'],
        'precision_safe': class_report['safe']['precision'],
        'recall_safe': class_report['safe']['recall'],
        'precision_medium': class_report['medium']['precision'],
        'recall_medium': class_report['medium']['recall'],
        'precision_critical': class_report['critical']['precision'],
        'recall_critical': class_report['critical']['recall'],
        'f1_critical': class_report['critical']['f1-score'],
        'auc_roc_weighted': auc_roc
    }
    mlflow.log_metrics(metrics)
    mlflow.log_params(params)
    
    signature = infer_signature(X_train_scaled, y_pred_encoded)
    mlflow.xgboost.log_model(xgb_model, "xgb_model", signature=signature)
    
    # Confusion matrix
    generate_confusion_matrix(y_test_encoded, y_pred_encoded, le.classes_, run_id=mlflow.active_run().info.run_id)
    print("Report:\n", classification_report(y_test_encoded, y_pred_encoded, target_names=le.classes_, zero_division=0))
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    # Feature importances
    feature_names = numerical_features + list(encoder.get_feature_names_out(categorical_features))
    importances = xgb_model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    importance_dict = {f'importance_{feature_names[i]}': importances[i] for i in sorted_idx[:5]}
    mlflow.log_metrics(importance_dict)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in sorted_idx[:10]], importances[sorted_idx[:10]])
    plt.xlabel('Importance')
    plt.title('Top Feature Importances')
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)
    
    # Cross-track generalization
    test_df['true'] = le.inverse_transform(y_test_encoded)
    test_df['predicted'] = le.inverse_transform(y_pred_encoded)
    test_df['true_encoded'] = y_test_encoded
    test_df['probs'] = [prob for prob in y_probs]
    
    for track in test_df['Track'].unique():
        df_track = test_df[test_df['Track'] == track]
        if len(df_track) > 0:
            report_track = classification_report(df_track['true'], df_track['predicted'], output_dict=True, zero_division=0)
            probs_track = np.vstack(df_track['probs'])
            auc_track = roc_auc_score(df_track['true_encoded'], probs_track, multi_class='ovr', average='weighted')
            
            mlflow.log_metric(f'recall_critical_{track}', report_track['critical']['recall'])
            mlflow.log_metric(f'f1_critical_{track}', report_track['critical']['f1-score'])
            mlflow.log_metric(f'auc_roc_{track}', auc_track)
            print(f"Track {track} - Recall Critical: {report_track['critical']['recall']:.4f}, F1 Critical: {report_track['critical']['f1-score']:.4f}, AUC: {auc_track:.4f}")

    # Export artifacts
    joblib.dump(xgb_model, 'models/xgb_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoder, 'models/encoder.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')