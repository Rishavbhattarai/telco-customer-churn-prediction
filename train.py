"""Customer churn: clean, compare LogReg vs Random Forest, SHAP importances."""
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/telco.csv").drop(columns="customerID")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)  # blanks = brand-new customers
df["Churn"] = (df["Churn"] == "Yes").astype(int)
X, y = df.drop(columns="Churn"), df["Churn"]
num = ["tenure", "MonthlyCharges", "TotalCharges"]
cat = [c for c in X.columns if c not in num]
pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced", n_jobs=-1, random_state=42),
}
metrics, fitted = {}, {}
for name, clf in models.items():
    p = Pipeline([("pre", pre), ("clf", clf)]).fit(Xtr, ytr)
    pred, proba = p.predict(Xte), p.predict_proba(Xte)[:, 1]
    metrics[name] = {"precision": precision_score(yte, pred), "recall": recall_score(yte, pred),
                     "f1": f1_score(yte, pred), "roc_auc": roc_auc_score(yte, proba)}
    fitted[name] = p
    print(name, {k: round(v, 3) for k, v in metrics[name].items()})

best = max(metrics, key=lambda n: metrics[n]["f1"])
pipe = fitted[best]
print("best by F1:", best)

# SHAP on the best model (transformed feature space)
Xs = pipe.named_steps["pre"].transform(Xte.iloc[:500])
names = pipe.named_steps["pre"].get_feature_names_out()
clf = pipe.named_steps["clf"]
expl = shap.TreeExplainer(clf) if best == "Random Forest" else shap.LinearExplainer(clf, pipe.named_steps["pre"].transform(Xtr))
sv = expl.shap_values(Xs)
if isinstance(sv, list):
    sv = sv[1]
elif getattr(sv, "ndim", 2) == 3:
    sv = sv[:, :, 1]
shap.summary_plot(sv, Xs, feature_names=names, show=False, max_display=12)
plt.tight_layout(); plt.savefig("shap_summary.png", dpi=130); plt.close()
mean_abs = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)

joblib.dump({"pipe": pipe, "shap_importance": mean_abs.head(15), "columns": list(X.columns),
             "test": pd.concat([Xte, yte], axis=1)}, "model.joblib")
json.dump({"best": best, "metrics": metrics}, open("metrics.json", "w"), indent=2)
