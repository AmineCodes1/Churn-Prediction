from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn.csv"
TARGET_CANDIDATES = {"churn", "exited", "attrition", "left", "is_churn", "target"}


def risk_label(probability: float) -> str:
    if probability >= 0.60:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


@st.cache_data
def load_reference_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def resolve_target_column(columns: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def build_feature_input(reference_df: pd.DataFrame, key_prefix: str = "single") -> pd.DataFrame:
    target_column = resolve_target_column(reference_df.columns.tolist())
    feature_df = reference_df.drop(columns=[target_column]) if target_column else reference_df.copy()

    user_values: dict[str, object] = {}
    for column in feature_df.columns:
        if pd.api.types.is_numeric_dtype(feature_df[column]):
            default_value = float(feature_df[column].median()) if not feature_df[column].dropna().empty else 0.0
            user_values[column] = st.number_input(
                column,
                value=default_value,
                key=f"{key_prefix}_{column}",
            )
        else:
            options = [str(value) for value in feature_df[column].dropna().astype(str).unique().tolist()]
            if not options:
                options = [""]
            user_values[column] = st.selectbox(
                column,
                options=sorted(options),
                key=f"{key_prefix}_{column}",
            )

    return pd.DataFrame([user_values])


def find_default_model_path() -> Path | None:
    candidates = sorted(MODEL_DIR.glob("*_pipeline.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    st.set_page_config(page_title="Churn Prediction", layout="wide")
    st.title("Customer Churn Prediction")
    st.caption("Predict churn risk and estimate financial impact from targeted retention.")

    st.sidebar.header("Business Assumptions")
    retention_cost = st.sidebar.number_input("Retention cost per customer ($)", min_value=0.0, value=50.0)
    clv = st.sidebar.number_input("Average CLV ($)", min_value=0.0, value=800.0)
    decision_threshold = st.sidebar.slider("Decision threshold", min_value=0.01, max_value=0.99, value=0.10)

    default_model = find_default_model_path()
    model_input = st.text_input(
        "Model path",
        value=str(default_model) if default_model else str(MODEL_DIR / "xgboost_pipeline.joblib"),
    )
    model_path = Path(model_input)

    if not model_path.exists():
        st.error(f"Model not found at: {model_path}")
        st.stop()

    model = load_model(model_path)
    st.success(f"Loaded model: {model_path.name}")

    schema_data_input = st.text_input("Reference dataset path (for form schema)", value=str(DEFAULT_DATA_PATH))
    schema_data_path = Path(schema_data_input)

    if not schema_data_path.exists():
        st.warning(f"Reference dataset not found at: {schema_data_path}. Upload a CSV for batch scoring below.")
        reference_df = None
    else:
        reference_df = load_reference_data(schema_data_path)

    tab_single, tab_batch = st.tabs(["Single Customer", "Batch CSV"])

    with tab_single:
        st.subheader("Single Customer Prediction")
        if reference_df is None:
            st.info("Provide a valid reference dataset path to auto-generate the input form.")
        else:
            with st.form("single_prediction_form"):
                input_df = build_feature_input(reference_df)
                submitted = st.form_submit_button("Predict Churn Risk")

            if submitted:
                probability = float(model.predict_proba(input_df)[:, 1][0])
                risk = risk_label(probability)
                action = "Target for retention" if probability >= decision_threshold else "Do not target"
                expected_value = (probability * clv) - retention_cost

                col1, col2, col3 = st.columns(3)
                col1.metric("Churn Probability", f"{probability:.2%}")
                col2.metric("Risk Category", risk)
                col3.metric("Decision", action)

                st.markdown("### Financial View")
                st.write(
                    f"Expected net value for targeting this customer: **${expected_value:,.2f}** "
                    f"(computed as probability × CLV − retention cost)."
                )

    with tab_batch:
        st.subheader("Batch Prediction & Portfolio Impact")
        uploaded = st.file_uploader("Upload CSV for scoring", type=["csv"])

        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            scored = batch_df.copy()
            scored["churn_probability"] = model.predict_proba(batch_df)[:, 1]
            scored["risk_category"] = scored["churn_probability"].apply(risk_label)
            scored["target_flag"] = scored["churn_probability"] >= decision_threshold

            targeted_count = int(scored["target_flag"].sum())
            expected_saved_revenue = float((scored.loc[scored["target_flag"], "churn_probability"].sum()) * clv)
            campaign_cost = float(targeted_count * retention_cost)
            expected_net_benefit = expected_saved_revenue - campaign_cost

            m1, m2, m3 = st.columns(3)
            m1.metric("Customers Scored", f"{len(scored):,}")
            m2.metric("Customers Targeted", f"{targeted_count:,}")
            m3.metric("Expected Net Benefit", f"${expected_net_benefit:,.0f}")

            st.dataframe(scored.head(100), use_container_width=True)


if __name__ == "__main__":
    main()
