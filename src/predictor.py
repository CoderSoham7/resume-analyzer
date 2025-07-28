import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
try:
    from scipy.sparse import issparse
except ImportError:
    def issparse(x) -> bool:
        return False

def train_model(career_data_file, model_output_path):
    df = pd.read_csv(career_data_file)
    df['Skills'] = df['Skills'].apply(lambda x: x.split(','))

    mlb = MultiLabelBinarizer()
    skill_features = mlb.fit_transform(df['Skills'])

    current_roles = pd.get_dummies(df['Current Role'])

    # Convert skill_features to DataFrame with string column names
    # Ensure skill_features is a dense numpy array if it is a sparse matrix
    import numpy as np
    try:
        from scipy.sparse import issparse
        is_sparse = issparse(skill_features)
    except ImportError:
        is_sparse = False
    if is_sparse:
        # Convert to dense numpy array using np.asarray
        skill_features = np.asarray(skill_features)
    elif not isinstance(skill_features, np.ndarray):
        skill_features = np.array(skill_features)
    skill_df = pd.DataFrame(skill_features, columns=[str(i) for i in range(skill_features.shape[1])])
    X = pd.concat([skill_df, current_roles], axis=1)
    X.columns = X.columns.astype(str)
    y = df['Next Role']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print("Model Accuracy:", accuracy_score(y_test, clf.predict(X_test)))

    joblib.dump(clf, model_output_path)
    joblib.dump(mlb, "skill_encoder.pkl")
    joblib.dump(current_roles.columns.tolist(), "role_columns.pkl")

def predict_next_role(model_path, skill_encoder_path, role_columns_path, current_role, skills):
    clf = joblib.load(model_path)
    mlb = joblib.load(skill_encoder_path)
    role_columns = joblib.load(role_columns_path)

    skill_vector = mlb.transform([skills])
    role_vector = [1 if role == current_role else 0 for role in role_columns]

    input_vector = pd.DataFrame([list(skill_vector[0]) + role_vector])
    return clf.predict(input_vector)[0]
