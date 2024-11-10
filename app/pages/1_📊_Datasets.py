import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


def display_existing_datasets(automl: AutoMLSystem) -> None:
    """
    Display a list of existing datasets available in the AutoMLSystem registry.
    """
    st.title("Dataset Management")
    st.header("Existing Datasets")

    # Retrieve datasets of type "dataset" from the registry
    datasets = automl.registry.list(type="dataset")

    if datasets:
        for dataset in datasets:
            dataset_name = dataset.metadata.get("name", "Unnamed Dataset")
            st.write(f"Dataset Name: {dataset_name}")
            st.write(f"Version: {dataset.version}")
            st.write("---")
    else:
        st.write("No datasets available. Please upload a new dataset.")


def upload_new_dataset(automl: AutoMLSystem) -> None:
    """
    Provides an interface for uploading a new dataset, saving it as an
    artifact in the AutoMLSystem.

    Args:
        automl (AutoMLSystem): The AutoML system instance managing artifacts.
    """
    st.header("Upload a New Dataset")

    # File uploader for CSV files
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        # Load CSV into DataFrame
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded dataset:", df.head())
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return

        # Get metadata for the dataset
        dataset_name = st.text_input("Dataset Name", "my_dataset")
        dataset_version = st.text_input("Dataset Version", "1.0.0")

        # Save the dataset as an artifact
        if st.button("Save Dataset"):
            try:
                # Convert DataFrame to Dataset object
                dataset = Dataset.from_dataframe(
                    data=df,
                    name=dataset_name,
                    asset_path="path/to/store",  # Ensure this path is valid
                    version=dataset_version
                )

                # Register the dataset in AutoMLSystem
                automl.registry.register(dataset)
                st.success(f"Dataset '{dataset_name}' version "
                           f"'{dataset_version}' saved successfully!")
            except Exception as e:
                st.error(f"Failed to save dataset: {e}")


def main() -> None:
    """
    Main function to run the Streamlit app for dataset management.
    Displays existing datasets and provides interface to upload new datasets.
    """
    # Get the singleton instance of AutoMLSystem
    automl = AutoMLSystem.get_instance()

    # Display existing datasets and provide an interface to upload new datasets
    display_existing_datasets(automl)
    upload_new_dataset(automl)


if __name__ == "__main__":
    main()
