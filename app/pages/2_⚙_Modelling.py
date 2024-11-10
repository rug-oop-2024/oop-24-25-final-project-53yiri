import streamlit as st
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import get_metric
from autoop.core.ml.model import (
    KNearestNeighbors, Logistic, SupportVectorMachine,
    LassoRegression, MultipleLinearRegression, RandomForest
)
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.feature import Feature
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os
import json

# Path to the pipelines directory
PIPELINES_DIR = os.path.join('assets', 'pipelines')
os.makedirs(PIPELINES_DIR, exist_ok=True)

# Path to the registry file
REGISTRY_PATH = os.path.join(PIPELINES_DIR, 'registry.json')

# Load registry if it exists
if not os.path.exists(REGISTRY_PATH):
    with open(REGISTRY_PATH, 'w') as registry_file:
        json.dump([], registry_file)

if "trained" not in st.session_state:
    st.session_state["trained"] = False
    st.session_state["train_Y"] = None
    st.session_state["test_Y"] = None
    st.session_state["train_predictions"] = None
    st.session_state["test_predictions"] = None


def get_model(model_name: str):
    """
    Factory function to instantiate the model based on the selected model name.
    """
    if model_name == "K-Nearest Neighbors":
        return KNearestNeighbors()
    elif model_name == "Logistic Regression":
        return Logistic()
    elif model_name == "SVM":
        return SupportVectorMachine()
    elif model_name == "Lasso Regression":
        return LassoRegression()
    elif model_name == "Multiple Linear Regression":
        return MultipleLinearRegression()
    elif model_name == "Random Forest Regressor":
        return RandomForest()
    else:
        raise ValueError(f"Model '{model_name}' is not implemented.")


if "trained" not in st.session_state:
    st.session_state["trained"] = False

# Page setup
st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


# Page title and description
st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine "
                  "learning pipeline to train a model on a dataset.")

# Initialize AutoML system instance
automl = AutoMLSystem.get_instance()

# Fetch existing datasets from the registry
artifacts = automl.registry.list(type="dataset")

# Convert artifacts to Dataset instances for the dropdown
datasets = []
for artifact in artifacts:
    if artifact.type == "dataset":
        dataset = Dataset(
            name=artifact.name,
            asset_path=artifact.asset_path,
            version=artifact.version,
            data=artifact.data,
        )
        datasets.append(dataset)

# Display the dataset selection dropdown
dataset_names = [d.name for d in datasets]
selected_dataset_name = st.selectbox("Select a Dataset", dataset_names)
selected_dataset = next((d for d in datasets
                         if d.name == selected_dataset_name), None)


# Preview the selected dataset and detect features
if selected_dataset:
    st.write("### Dataset Preview")
    df = selected_dataset.read()
    st.dataframe(df.head())

    # Detect features in the selected dataset
    features = detect_feature_types(selected_dataset)

    # Create options for selecting input and target features
    feature_names = [f.name for f in features]
    selected_input_features = st.multiselect("Select Input Features",
                                             feature_names)
    selected_target_feature = st.selectbox("Select Target Feature",
                                           feature_names)

    # Determine the task type based on the selected target feature
    target_feature = next(
        (f for f in features if f.name == selected_target_feature), None
        )
    task_type = None
    model_options = []
    metric_options = []

    if target_feature:
        if target_feature.feature_type == "categorical":
            task_type = "Classification"
            model_options = ["K-Nearest Neighbors",
                             "Logistic Regression",
                             "SVM"]
            metric_options = ["accuracy", "precision", "recall"]
        elif target_feature.feature_type == "numerical":
            task_type = "Regression"
            model_options = ["Lasso Regression",
                             "Multiple Linear Regression",
                             "Random Forest Regressor"]
            metric_options = ["mean_squared_error",
                              "mean_absolute_error",
                              "r2_score"]

        # Display task type
        st.write("### Detected Task Type")
        st.write(f"Task Type: **{task_type}**")

        # Display model selection dropdown based on task type
        selected_model_name = st.selectbox("Select a Model", model_options)

        # Display metric selection for the task type
        st.write("### Select Evaluation Metrics")
        selected_metrics = st.multiselect("Choose metrics to evaluate the "
                                          "model performance", metric_options)

        # Dataset split slider
        st.write("### Dataset Split")
        train_split = st.slider(
            "Select Training Data Split (%)",
            min_value=50,
            max_value=90,
            value=80,
            step=1,
            help="Choose the percentage of data to be used for training."
        )

        test_split = 100 - train_split

        # Pipeline Summary
        st.write("### Pipeline Summary")
        st.markdown("---")
        st.write("#### Selected Dataset")
        st.write(f"- **Name**: {selected_dataset_name}")
        st.write(f"- **Version**: {selected_dataset.version}")

        st.write("#### Features")
        st.write(f"- **Input Features**: {', '.join(selected_input_features)}")
        st.write(f"- **Target Feature**: {selected_target_feature}")

        st.write("#### Model")
        st.write(f"- **Model**: {selected_model_name}")

        st.write("#### Task Type")
        st.write(f"- **Task**: {task_type}")

        st.write("#### Evaluation Metrics")
        st.write(", ".join(selected_metrics))

        st.write("#### Data Split")
        st.write(f"- **Training Split**: {train_split}%")
        st.write(f"- **Testing Split**: {test_split}%")

        st.markdown("---")
        st.write("Review your configurations and "
                 "proceed with training when ready.")

        # Instantiate the model and metrics
        model = get_model(selected_model_name)
        metrics = [get_metric(metric_name) for metric_name in selected_metrics]
        input_features = [
            Feature(
                name=feat.name,
                feature_type=feat.feature_type,
                unique_values=feat.unique_values
            ) for feat in features if feat.name in selected_input_features
            ]
        target_feature_obj = Feature(
            name=target_feature.name,
            feature_type=target_feature.feature_type,
            unique_values=target_feature.unique_values
        )

        # Create the pipeline
        pipeline = Pipeline(
            metrics=metrics,
            dataset=selected_dataset,
            model=model,
            input_features=input_features,
            target_feature=target_feature_obj,
            split=train_split / 100
        )

        # Button to start training
        if st.button("Train Model"):
            st.session_state["trained"] = True
            st.write("### Training Results")
            results = pipeline.execute()
            trained_model = results["model"]

            with open("trained_model.pkl", "wb") as model_file:
                pickle.dump(trained_model, model_file)

            st.session_state["train_Y"] = pipeline._train_y
            st.session_state["test_Y"] = pipeline._test_y
            st.session_state["train_predictions"] = results[
                "predictions"
                ]["train"]
            st.session_state["test_predictions"] = results[
                "predictions"
                ]["test"]

            # Display evaluation results
            st.write("### Evaluation Metrics")
            for metric_type, metric_dict in results["metrics"].items():
                st.write(f"#### {metric_type.capitalize()} Metrics")
                metric_cols = st.columns(len(metric_dict))
                for idx, (metric_name, score) in enumerate(
                        metric_dict.items()):
                    metric_cols[idx].metric(label=metric_name.capitalize(),
                                            value=f"{score:.4f}")

            # Prepare data for predictions display
            train_X = pipeline._compact_vectors(pipeline._train_X)
            train_Y = pipeline._train_y
            train_predictions = results["predictions"]["train"]
            test_X = pipeline._compact_vectors(pipeline._test_X)
            test_Y = pipeline._test_y
            test_predictions = results["predictions"]["test"]

            # Display predictions in a table
            st.write("### Predictions")

            st.write("#### Training Set Predictions")
            train_predictions_df = pd.DataFrame({
                'Actual': train_Y,
                'Predicted': train_predictions
            })
            st.dataframe(train_predictions_df.head(10))

            st.write("#### Test Set Predictions")
            test_predictions_df = pd.DataFrame({
                'Actual': test_Y,
                'Predicted': test_predictions
            })
            st.dataframe(test_predictions_df.head(10))

            # Visualize predictions
            if task_type == "Regression":
                # Plot actual vs. predicted
                st.write("#### Actual vs. Predicted Values (Test Set)")
                plt.figure(figsize=(10, 6))
                plt.scatter(test_Y, test_predictions, alpha=0.7)
                plt.plot([test_Y.min(), test_Y.max()],
                         [test_Y.min(), test_Y.max()], 'r--')
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title('Actual vs. Predicted Values')
                st.pyplot(plt)
            elif task_type == "Classification":
                # Confusion matrix
                st.write("#### Confusion Matrix (Test Set)")
                cm = confusion_matrix(test_Y, test_predictions)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted Labels')
                plt.ylabel('True Labels')
                plt.title('Confusion Matrix')
                st.pyplot(plt)

                # Classification report
                st.write("#### Classification Report (Test Set)")
                report = classification_report(test_Y, test_predictions,
                                               output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

            st.success("Model training and evaluation completed!")

        st.write("### Save Pipeline")
        pipeline_name = st.text_input("Enter a name for your pipeline")
        pipeline_version = st.text_input("Enter a version for your pipeline",
                                         value="1.0.0")

        if st.button("Save Pipeline"):
            if not st.session_state["trained"]:
                st.error("Please train the model before saving the pipeline.")
            elif not pipeline_name or not pipeline_version:
                st.error("Please enter both a name "
                         "and version for the pipeline.")
            else:
                try:
                    # Serialize the pipeline object
                    pipeline_data = pickle.dumps(pipeline)

                    # Define the pipeline file path
                    pipeline_filename = (
                        f"{pipeline_name}_{pipeline_version}.pkl"
                    )
                    pipeline_path = os.path.join(PIPELINES_DIR,
                                                 pipeline_filename)

                    # Save the serialized pipeline to the file
                    with open(pipeline_path, 'wb') as f:
                        f.write(pipeline_data)

                    metric_results = {}
                    for metric in metrics:
                        # Retrieve a readable name for the metric
                        metric_name = metric.__class__.__name__

                        # Calculate metric values for train and test sets
                        train_metric_value = metric(
                            st.session_state["train_Y"],
                            st.session_state["train_predictions"]
                            )
                        test_metric_value = metric(
                            st.session_state["test_Y"],
                            st.session_state["test_predictions"]
                            )

                        # Store the results in a dictionary with metric names
                        metric_results[metric_name] = {
                            "train": float(train_metric_value),
                            "test": float(test_metric_value)
                        }

                    # Load the current registry data
                    with open(REGISTRY_PATH, 'r+') as registry_file:
                        registry = json.load(registry_file)
                        pipeline_metadata = {
                            "pipeline_name": pipeline_name,
                            "pipeline_version": pipeline_version,
                            "path": pipeline_path,
                            "model_type": model.__class__.__name__,
                            "model_name": selected_model_name,
                            "input_features": [f.name for f in input_features],
                            "target_feature": target_feature_obj.name,
                            "split_ratio": train_split / 100,
                            "metrics": metric_results
                        }
                        registry.append(pipeline_metadata)
                        registry_file.seek(0)
                        json.dump(registry, registry_file, indent=4)

                    st.success(f"Pipeline '{pipeline_name}' version "
                               f"'{pipeline_version}' saved successfully!")
                except Exception as e:
                    st.error(f"Failed to save pipeline: {e}")
