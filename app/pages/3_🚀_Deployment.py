import streamlit as st
import pandas as pd
import pickle
import os
import json

# Define paths for pipeline storage and registry
PIPELINES_DIR = os.path.join('assets', 'pipelines')
REGISTRY_PATH = os.path.join(PIPELINES_DIR, 'registry.json')


# Load the registry to access saved pipelines
def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r') as f:
            return json.load(f)
    return []


# Load a specific pipeline from its path
def load_pipeline(pipeline_path):
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


# Set up Streamlit page configuration
st.set_page_config(page_title="Deployment", page_icon="🚀")
st.title("🚀 Model Deployment")

# Step 1: Load the pipeline registry and list available pipelines
registry = load_registry()

if registry:
    st.subheader("Select a Saved Pipeline")
    pipeline_options = [
        f"{p['pipeline_name']} (v{p['pipeline_version']})" for p in registry
        ]
    selected_pipeline = st.selectbox("Choose a pipeline for deployment",
                                     pipeline_options)

    # Step 2: Display pipeline summary
    if selected_pipeline:
        selected_pipeline_info = next(
            (p for p in registry
             if f"{p['pipeline_name']} (v{p['pipeline_version']})" ==
             selected_pipeline),
            None
        )

        if selected_pipeline_info:
            # Show summary of the selected pipeline
            st.write("### Pipeline Summary")
            st.write(
                f"**Pipeline Name**: {selected_pipeline_info['pipeline_name']}"
                )
            st.write(
                f"**Version**: {selected_pipeline_info['pipeline_version']}"
                )
            st.write(f"**Model Type**: {selected_pipeline_info['model_type']}")
            st.write(f"**Model Name**: {selected_pipeline_info['model_name']}")
            st.write(
                "**Input Features**: "
                f"{', '.join(selected_pipeline_info['input_features'])}"
                )
            st.write(
                "**Target Feature**: "
                f"{selected_pipeline_info['target_feature']}"
                )
            st.write("**Metrics**:")
            for metric_name, scores in selected_pipeline_info["metrics"]\
                    .items():
                st.write(f"  - **{metric_name}**: Train = "
                         f"{scores['train']:.4f}, Test = {scores['test']:.4f}")

            # Load the actual pipeline for prediction
            pipeline_path = selected_pipeline_info["path"]
            pipeline = load_pipeline(pipeline_path)

            # Step 3: Allow the user to upload a CSV file for predictions
            st.subheader("Upload Data for Prediction")
            uploaded_file = st.file_uploader(
                "Upload CSV file for prediction", type=["csv"]
                )

            if uploaded_file:
                # Read and display the uploaded data
                new_data = pd.read_csv(uploaded_file)
                st.write("### Uploaded Data Preview")
                st.dataframe(new_data.head())

                try:
                    # Select the required input features from the uploaded data
                    input_features = [
                        feature.name for feature in pipeline._input_features
                        ]
                    missing_features = [
                        feat for feat in input_features
                        if feat not in new_data.columns
                        ]

                    if missing_features:
                        st.error(
                            "Uploaded data missing required input features: "
                            f"{', '.join(missing_features)}"
                            )
                    else:
                        # Keep only required input features for predictions
                        new_data = new_data[input_features]

                        # Perform prediction
                        predictions = pipeline.model.predict(new_data)

                        # Display the predictions
                        st.write("### Predictions")
                        prediction_df = pd.DataFrame(
                            predictions, columns=["Predicted Output"]
                            )
                        st.dataframe(prediction_df)

                        # Provide option to download predictions
                        csv_data = prediction_df.to_csv(
                            index=False).encode('utf-8'
                                                )
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv_data,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Error in processing or predicting: {e}")
else:
    st.write("No saved pipelines available. "
             "Please create and save a pipeline in the Modelling page.")
