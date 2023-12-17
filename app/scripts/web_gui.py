import streamlit as st
from single_model_run import SkinDiseaseModel
from _config import MODEL_DIR, TRAIN_DIR, TEST_DIR, TEST_IMG_PATH, RESULT_DIR
import datetime
from PIL import Image
from pathlib import Path
import pandas as pd

# Set theme
st.set_page_config(
    page_title="Skin disease prediction application",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "Skin disease analysis. Made to detect skin anomalies and diseases using Machine learning.",
    },
)

# Create an instance of the SkinDiseaseModel
model_instance = SkinDiseaseModel(model_type="mobilenetv2")


def display_directory_status_table():
    st.sidebar.title("Directory and file health check")
    st.sidebar.write(
        "Displays status of various directories and files required for the project to function."
    )
    st.sidebar.divider()

    # Data for the table
    directory_data = {
        "Model Directory": MODEL_DIR,
        "Model File": model_instance.model_weights_file,
        "Training Directory": TRAIN_DIR,
        "Test Directory": TEST_DIR,
        "Result Directory": RESULT_DIR,
        "Image upload directory": TEST_IMG_PATH,
    }

    # Check if the directories exist
    status_data = {
        key: ("Available ✅" if Path(value).exists() else "Not Available ❌")
        for key, value in directory_data.items()
    }

    # Display the table without column names
    for key, value in status_data.items():
        st.sidebar.write(f"**{key}**: {value}")


def predict_tab():
    col1, col2 = st.columns(2)

    with col1:
        st.title("Predict by uploading an image")
        st.write("Perform prediction by uploading an image.")
        # File upload handler
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    with col2:
        # Number of predictions text box
        num_predictions = st.slider(
            "Number of results to return",
            value=3,
            key="num_predictions",
            min_value=1,
            step=1,
            max_value=10,
        )

        if uploaded_file is not None:
            # Display the "Predict" button between file upload and image preview
            if st.button("Predict"):
                timestamp = datetime.datetime.now().strftime("%d%m%y_%H%M")
                original_filename = uploaded_file.name
                image_path = f"{TEST_IMG_PATH}/{original_filename[:-4]}_{timestamp}.jpg"
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                predict_result = model_instance.predict_class(
                    image_path, num_predictions=num_predictions
                )
                st.json(predict_result)

    with col1:
        # Display the uploaded image
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(
                uploaded_image,
                caption=f"{uploaded_file.name}",
                use_column_width=True,
            )


def train_tab():
    st.title("Training the model")
    st.write(
        "Train the model again after replacing the dataset or after changing the underlying code."
    )

    col1, col2 = st.columns(2)

    with col1:  # Train
        # Override toggle

        override_toggle = st.toggle(
            label="Override check for existing model file", key="override", value=False
        )

        if override_toggle:
            override = True
        else:
            override = False

        if st.button("Train Model"):
            model_instance.train_model(TRAIN_DIR, override=override)
            st.success(f"Model {'re-trained' if override else 'trained'} successfully!")

    with col2:
        # Display the most recent training accuracy image
        train_accuracy_image_path = RESULT_DIR / "training_accuracy_mobilenetv2.png"
        if train_accuracy_image_path.exists():
            train_accuracy_image = Image.open(train_accuracy_image_path.as_posix())
            st.image(
                train_accuracy_image,
                caption="Most Recent Training Accuracy",
                width=600,
                use_column_width="auto",
            )


def evaluate_tab():
    st.title("Evaluating the model")
    st.write(
        "Evaluating the model by testing it on test data for gauging model performance."
    )

    col1, col2 = st.columns(2)

    with col1:
        # Evaluate
        if st.button("Evaluate Model"):
            evaluate_result = model_instance.evaluate_model(TEST_DIR)
            st.json(evaluate_result)

    with col2:
        # Display the confusion matrix image
        confusion_matrix_image_path = RESULT_DIR / "confusion_matrix_mobilenetv2.png"
        if confusion_matrix_image_path.exists():
            confusion_matrix_image_path.as_posix()
            confusion_matrix_image = Image.open(confusion_matrix_image_path.as_posix())

            st.image(
                confusion_matrix_image,
                caption="Evaluate Function Confusion Matrix",
                width=600,
                use_column_width="auto",
            )


def main():
    st.markdown(
        r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.title("Skin disease prediction 🔬")
    # Display directory status table in the sidebar
    display_directory_status_table()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Predict", "Train", "Evaluate"])

    # Main content based on selected tab
    with tab1:
        predict_tab()
    with tab2:
        train_tab()
    with tab3:
        evaluate_tab()


if __name__ == "__main__":
    main()
