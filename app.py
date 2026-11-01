
import streamlit as st
import pandas as pd
from pathlib import Path

from src.prediction import ProductClassifier
from src.analytics import (
    load_model_metadata,
    load_model_comparison,
    load_cross_validation,
    load_misclassified
)


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="ShopSense AI",
    page_icon="🛍️",
    layout="wide"
)


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"


# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_classifier():
    return ProductClassifier()


try:
    classifier = load_classifier()
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()


# -----------------------------
# Load Reports
# -----------------------------

metadata = load_model_metadata()
comparison_df = load_model_comparison()
cv_df = load_cross_validation()
misclassified_df = load_misclassified()


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("ShopSense AI")

st.sidebar.write(
    "E-commerce Product Classification using NLP"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Single Prediction",
        "Batch Prediction",
        "Model Analysis",
        "About"
    ]
)


# -----------------------------
# Dashboard
# -----------------------------

if page == "Dashboard":

    st.title("ShopSense AI Dashboard")

    st.write(
        "An NLP-powered system for automatic "
        "e-commerce product classification."
    )

    selected_model = metadata.get(
        "selected_model",
        "Not available"
    )

    categories = metadata.get("categories", [])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Selected Model",
        str(selected_model)
    )

    col2.metric(
        "Product Categories",
        len(categories)
    )

    col3.metric(
        "Compared Models",
        len(comparison_df)
    )

    st.divider()

    st.subheader("Supported Categories")

    if categories:
        cols = st.columns(len(categories))

        for col, category in zip(cols, categories):
            col.info(str(category))
    else:
        st.info("Category metadata is not available.")

    st.divider()

    st.subheader("Model Comparison")

    if not comparison_df.empty:
        st.dataframe(
            comparison_df,
            use_container_width=True
        )
    else:
        st.info("Model comparison report is not available.")


# -----------------------------
# Single Prediction
# -----------------------------

elif page == "Single Prediction":

    st.title("Single Product Classification")

    st.write(
        "Enter a product description to predict its category."
    )

    product_text = st.text_area(
        "Product Description",
        placeholder=(
            "Example: Samsung Galaxy smartphone "
            "with 128GB storage and high-resolution camera"
        ),
        height=180
    )

    if st.button("Classify Product", type="primary"):

        if not product_text.strip():

            st.warning("Please enter a product description.")

        else:

            try:
                result = classifier.predict(product_text)

                st.success("Prediction completed!")

                st.subheader("Predicted Category")

                st.title(result["prediction"])

                with st.expander("View Cleaned Text"):
                    st.write(result["cleaned_text"])

                scores = result["scores"]

                if scores:

                    st.subheader("Model Scores")

                    score_df = pd.DataFrame({
                        "Category": list(scores.keys()),
                        "Score": list(scores.values())
                    })

                    st.bar_chart(
                        score_df.set_index("Category")
                    )

                    if hasattr(
                        classifier.model,
                        "predict_proba"
                    ):
                        st.caption(
                            "Scores are predicted probabilities."
                        )
                    else:
                        st.caption(
                            "Scores are decision-function values, "
                            "not calibrated probabilities."
                        )

            except Exception as error:
                st.error(f"Prediction error: {error}")


# -----------------------------
# Batch Prediction
# -----------------------------

elif page == "Batch Prediction":

    st.title("Batch Product Classification")

    st.write(
        "Upload a CSV file containing product descriptions."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.warning("The uploaded CSV file is empty.")
                st.stop()

            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            text_column = st.selectbox(
                "Select Product Description Column",
                df.columns
            )

            if st.button(
                "Classify All Products",
                type="primary"
            ):

                texts = df[text_column].fillna("").astype(str)

                valid_mask = texts.str.strip().ne("")

                results = [None] * len(df)

                if valid_mask.any():

                    valid_predictions = classifier.predict_batch(
                        texts[valid_mask].tolist()
                    )

                    for position, prediction in zip(
                        df.index[valid_mask],
                        valid_predictions
                    ):
                        results[df.index.get_loc(position)] = prediction

                output_df = df.copy()

                output_df["Predicted_Category"] = results

                output_df["Prediction_Status"] = [
                    "Predicted" if prediction is not None
                    else "Empty Description"
                    for prediction in results
                ]

                st.success("Batch classification completed!")

                st.dataframe(
                    output_df,
                    use_container_width=True
                )

                csv_data = output_df.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_data,
                    file_name="shopsense_predictions.csv",
                    mime="text/csv"
                )

        except Exception as error:
            st.error(f"Could not process CSV: {error}")


# -----------------------------
# Model Analysis
# -----------------------------

elif page == "Model Analysis":

    st.title("Model Analysis")

    st.subheader("Model Comparison")

    if not comparison_df.empty:

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        numeric_columns = comparison_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:

            metric = st.selectbox(
                "Select Evaluation Metric",
                numeric_columns
            )

            chart_df = comparison_df.copy()

            st.bar_chart(
                chart_df.set_index(
                    chart_df.columns[0]
                )[metric]
            )

    else:
        st.info("No model comparison report found.")

    st.divider()

    st.subheader("Cross Validation Results")

    if not cv_df.empty:

        st.dataframe(
            cv_df,
            use_container_width=True
        )

    else:
        st.info("No cross-validation report found.")

    st.divider()

    st.subheader("Misclassified Examples")

    if not misclassified_df.empty:

        st.dataframe(
            misclassified_df,
            use_container_width=True
        )

    else:
        st.info("No misclassified examples report found.")


# -----------------------------
# About
# -----------------------------

elif page == "About":

    st.title("About ShopSense AI")

    st.write("""
    ShopSense AI is an NLP-based e-commerce product
    classification system.

    It automatically classifies product descriptions
    into predefined product categories.

    **Machine Learning Workflow:**

    - Data Exploration
    - Text Preprocessing
    - TF-IDF Feature Extraction
    - Model Training
    - Cross Validation
    - Model Evaluation
    - Model Selection
    - Product Classification

    **Technologies:**

    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF
    - NLP
    - Streamlit
    """)

    st.subheader("Selected Model")

    st.write(
        metadata.get("selected_model", "Not available")
    )

    st.subheader("Categories")

    st.write(
        metadata.get("categories", [])
    )


# -----------------------------
# Footer
# -----------------------------

st.sidebar.divider()

st.sidebar.caption(
    "ShopSense AI | NLP Product Classification"
)
