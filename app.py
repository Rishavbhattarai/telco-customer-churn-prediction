import json
import joblib
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Churn Predictor", layout="wide")
art = joblib.load("model.joblib")
pipe, test = art["pipe"], art["test"]
m = json.load(open("metrics.json"))

st.title("Customer Churn Predictor")
tab1, tab2, tab3 = st.tabs(["Model comparison", "Why customers leave", "Score a customer"])

with tab1:
    st.dataframe(pd.DataFrame(m["metrics"]).T.round(3))
    st.caption(f"Best by F1: {m['best']}. Recall is prioritised: missing a leaver costs more than a wasted offer.")

with tab2:
    st.image("shap_summary.png", caption="SHAP: red = high feature value, right = pushes toward churn")
    st.bar_chart(art["shap_importance"])

with tab3:
    idx = st.slider("Pick a test-set customer", 0, len(test) - 1, 0)
    row = test.iloc[[idx]].drop(columns="Churn")
    p = pipe.predict_proba(row)[0, 1]
    c1, c2 = st.columns(2)
    c1.metric("Churn probability", f"{p:.0%}")
    c1.write("Actual outcome: " + ("churned" if test.iloc[idx]["Churn"] else "stayed"))
    c1.dataframe(row.T.astype(str))
    Xt = pipe.named_steps["pre"].transform(row)
    names = pipe.named_steps["pre"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    sv = shap.TreeExplainer(clf).shap_values(Xt) if hasattr(clf, "estimators_") else shap.LinearExplainer(clf, Xt).shap_values(Xt)
    sv = sv[:, :, 1][0] if getattr(sv, "ndim", 0) == 3 else (sv[1][0] if isinstance(sv, list) else sv[0])
    top = pd.Series(sv, index=names).sort_values(ascending=False).head(5)
    c2.subheader("Top churn drivers for this customer")
    c2.bar_chart(top)
    tips = {"Contract_Month-to-month": "Offer a discount to move to a 1-2 year contract.",
            "InternetService_Fiber optic": "Check fibre service quality / price; offer a speed or price review.",
            "OnlineSecurity_No": "Bundle free online security for 3 months.",
            "TechSupport_No": "Offer complimentary tech support.",
            "PaymentMethod_Electronic check": "Incentivise auto-pay by card or bank transfer.",
            "tenure": "New customer: run an onboarding check-in at months 1-3.",
            "MonthlyCharges": "Review the plan for a cheaper fit."}
    c2.subheader("Suggested retention actions")
    shown = False
    for f in top.index:
        key = f.split("__", 1)[1]
        if key in tips and top[f] > 0:
            c2.write("• " + tips[key]); shown = True
    if not shown:
        c2.write("• No strong risk drivers; keep as is.")
