import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.utils import class_weight

# Set the matplotlib backend to Agg to prevent TclError
import matplotlib
matplotlib.use('Agg')

# -------------------------------
# STEP 1: Load and Clean Dataset
# -------------------------------
df = pd.read_csv("flood_prediction_dataset.csv")

# Correct the column name with a leading space
if ' Population' in df.columns:
    df.rename(columns={' Population': 'Population'}, inplace=True)
if 'Discharge (m³/s)' in df.columns:
    df.rename(columns={'Discharge (m³/s)': 'Discharge_m3s'}, inplace=True)

# Replace '--' with NaN across the dataset
df = df.replace("--", np.nan)

# -------------------------------
# STEP 2: Drop Unnecessary Columns
# -------------------------------
df = df.drop(columns=["Areas", "Nearest Station", "Drainage_properties", "Drainage_line_id"], errors='ignore')

# -------------------------------
# STEP 3: Handle Categorical Variables Safely
# -------------------------------
cat_cols = ["Ward Code", "Land Use Classes", "Flood_occured", "Road Density_m", "Monitoring_required", "Soil Type"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    else:
        print(f"Warning: Column '{col}' not found in DataFrame.")

# -------------------------------
# STEP 4: Handle Numeric Columns (Fill Missing)
# -------------------------------
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

# -------------------------------
# STEP 5: Feature/Target Separation and Encoding
# -------------------------------
# IMPORTANT: Drop all features that cause data leakage.
features_to_drop = [
    "Flood-risk_level",      # The target variable itself
    "DATE",                  # Irrelevant for prediction
    "true_conditions_count", # Highly correlated (leaky)
    "Soil Wetness Index",    # Highly correlated (leaky)
    "Runoff equivalent",     # Highly correlated (leaky)
    "Discharge_m3s",         # Highly correlated (leaky)
    "Flood_occured",         # Likely a direct cause of flood-risk
    "Monitoring_required",   # Likely decided based on flood-risk
    "Drainage_properties",
    "Drainage_line_id"
]

X = df.drop(columns=features_to_drop, errors='ignore')
y = df["Flood-risk_level"]

# Encode the target variable `y` into numerical values
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
print(df)
# -------------------------------
# STEP 6: Train-Test Split with a fixed size
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=4500, random_state=42, stratify=y
)

print("✅ Preprocessing complete")
print("New training set shape:", X_train.shape)
print("New test set shape:", X_test.shape)

# -------------------------------
# STEP 7: Scale Numeric Columns
# -------------------------------
num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# -------------------------------
# STEP 8: Final Model Training (Voting Ensemble)
# -------------------------------
print("\nTraining Voting Ensemble Model...")
# Define base models with their best-tuned hyperparameters from previous runs.
base_model_rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2, class_weight='balanced', random_state=42)
base_model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.2, max_depth=7, subsample=1.0, use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Pass the correct sample_weights for XGBoost
sample_weights_xgb = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

# Train the base models individually to get their trained state
base_model_rf.fit(X_train, y_train)
base_model_xgb.fit(X_train, y_train, sample_weight=sample_weights_xgb)

# Create the Voting Ensemble model
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', base_model_rf),
        ('xgb', base_model_xgb)
    ],
    voting='soft',
    n_jobs=-1
)

ensemble_model.fit(X_train, y_train)
print("✅ Voting Ensemble Model trained successfully.")


# -------------------------------
# STEP 9: Evaluate the Final Model
# -------------------------------
def evaluate_model(X_train, y_train, X_test, y_test, model, name):
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    # This line prints the Training Accuracy
    print(f"\nTraining Accuracy for {name}: {train_acc:.4f}")
    
    # Calculate validation accuracy using cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy')
    validation_acc = np.mean(cv_scores)
    # This line prints the Validation Accuracy
    print(f"Validation Accuracy for {name} (Cross-Validation): {validation_acc:.4f}")
    
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    # This line prints the Test Accuracy
    print(f"✅ Test Accuracy for {name}: {test_acc:.4f}")
    
    print("Classification Report on Test Data:\n", classification_report(y_test, y_pred, zero_division=0, target_names=target_encoder.classes_))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

evaluate_model(X_train, y_train, X_test, y_test, ensemble_model, "Voting Ensemble")

# -------------------------------
# STEP 10: Save Final Model
# -------------------------------
joblib.dump(ensemble_model, 'ensemble_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(target_encoder, 'target_encoder.joblib')

print("\n✅ Final model and encoders saved successfully.")