import pickle

import numpy as np
import pandas as pd
import streamlit as st



st.set_page_config(
    page_title="Visual Response Predictor",
    page_icon="👁️",
    layout="centered",
)


OH_ENCODER_PATH = "onehot_encoder.pkl"
CLASSIFICATION_MODEL = "hgb_normal_model"



@st.cache_resource
def load_model(model_path):
    """Load and cache a serialized model or transformer."""
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


CLASS_LABELS = {
    0: "Not a major visual responder",
    1: "Major visual responder",
}

CATEGORICAL_SECTIONS = {
    "Patient and treatment": {
        "treatment_group": ["Aflibercept", "Bevacizumab", "Ranibizumab"],
        "study_eye": ["OD", "OS"],
        "NSEyeAntiVEGFInj": ["No", "Yes"],
        "Gender": ["F", "M"],
        "Ethnicity": [
            "Hispanic or Latino",
            "Not Hispanic or Latino",
            "Unknown/not reported",
        ],
        "Race": [
            "American Indian/Alaskan Native",
            "Asian",
            "Black/African American",
            "More than one race",
            "Native Hawaiian/Other Pacific Islander",
            "Unknown/not reported",
            "White",
        ],
        "DiabetesType": ["Type 1", "Type 2", "Uncertain"],
        "InsulinUsed": ["No", "Yes"],
        "PtCurrMed": ["No", "Yes"],
        "PreExistMedCond": ["No", "Yes"],
    },
    "Eye history and examination": {
        "study_eye_dmetrthx": ["No", "Yes"],
        "IOPMeasured": ["Goldmann", "Other", "Tonopen"],
        "IOPDilation": ["Post-Dilation", "Pre-Dilation"],
        "study_eye_lensstatus": ["PC IOL", "Phakic"],
        "study_eye_lensopac": [
            "Minimal, no effect on visual acuity",
            "Visually significant",
        ],
        "study_eye_vitreoushemcond": ["No", "Yes"],
        "study_eye_vitreoushaze": ["Clear", "Mild blur", "Trace haze"],
        "study_eye_drseverity": [
            "Microaneurysms only",
            "Mild/moderate NPDR",
            "PDR and/or prior scatter",
            "Severe NPDR",
        ],
        "study_eye_dmeclinexam": ["Present, center involved"],
        "study_eye_dmechar": [
            "Intermediate",
            "Predominately diffuse",
            "Predominately focal",
            "Typical diffuse (textbook example)",
            "Typical focal (textbook example)",
        ],
        "study_eye_postsegfundabno": ["No", "Yes"],
    },
    "OCT findings": {
        "OCTMachine": [
            "Heidelberg Spectralis",
            "Zeiss Cirrus",
            "Zeiss Stratus",
        ],
        "baseline_intraretinal_edema": ["No", "Yes"],
        "baseline_subretinal_fluid": ["Absent", "No", "Ungradable", "Yes"],
        "baseline_epiretinal_membrane": ["No", "Ungradable", "Yes"],
        "baseline_vitreomacular_traction": ["No", "Ungradable", "Yes"],
        "baseline_rc_oct_machine": [
            "Heidelberg Spectralis",
            "Zeiss Cirrus",
            "Zeiss Stratus",
        ],
        "baseline_rc_oct_scan_quality": [
            "Acceptable (after correction)",
            "Acceptable (as obtained)",
            "Unacceptable",
        ],
    },
    "Fundus photography": {
        "baseline_photo_dr_severity": [
            "10",
            "12",
            "20",
            "35B",
            "35C",
            "35D",
            "35E",
            "35F",
            "43A",
            "43B",
            "47A",
            "47B",
            "47C",
            "53A",
            "53B",
            "53C",
            "53E",
            "60",
            "61A",
            "61B",
            "65A",
            "65B",
            "65C",
            "71A",
            "71C",
            "71D",
            "75",
            "90",
        ],
        "baseline_photo_csme": [
            "Cannot grade",
            "Questionable",
            "RT or adjacent HE <=500 microns from center",
            "Zone of RT >= 1 DA, part <= 1 DD from center",
        ],
        "baseline_photo_cyst": ["Cannot grade", "Definite", "Questionable"],
        "baseline_photo_traction": [
            "Cannot grade",
            "Questionable",
            "Tension lines only",
        ],
        "baseline_photo_he_plaque_scar": [
            "Cannot grade",
            "Fibrous scar",
            "Organized plaque",
            "Questionable",
        ],
        "baseline_photo_macular_abnormality": [
            "Cannot grade",
            "No evidence",
            "Questionable or definite",
        ],
        "baseline_photo_quality": [
            "Borderline Unexplained",
            "Good",
            "Ungradable Unexplained",
        ],
    },
}


DEFAULT_VALUES = {
    "treatment_group": "Aflibercept",
    "study_eye": "OD",
    "NSEyeAntiVEGFInj": "No",
    "Gender": "M",
    "Ethnicity": "Not Hispanic or Latino",
    "Race": "White",
    "DiabetesType": "Type 2",
    "InsulinUsed": "Yes",
    "PtCurrMed": "Yes",
    "PreExistMedCond": "Yes",
    "study_eye_dmetrthx": "No",
    "IOPMeasured": "Goldmann",
    "IOPDilation": "Pre-Dilation",
    "study_eye_lensstatus": "Phakic",
    "study_eye_lensopac": "Minimal, no effect on visual acuity",
    "study_eye_vitreoushemcond": "No",
    "study_eye_vitreoushaze": "Clear",
    "study_eye_drseverity": "Mild/moderate NPDR",
    "study_eye_dmeclinexam": "Present, center involved",
    "study_eye_dmechar": "Predominately diffuse",
    "study_eye_postsegfundabno": "No",
    "OCTMachine": "Zeiss Cirrus",
    "baseline_intraretinal_edema": "Yes",
    "baseline_subretinal_fluid": "No",
    "baseline_epiretinal_membrane": "Yes",
    "baseline_vitreomacular_traction": "No",
    "baseline_rc_oct_machine": "Zeiss Cirrus",
    "baseline_rc_oct_scan_quality": "Acceptable (as obtained)",
    "baseline_photo_dr_severity": "47A",
    "baseline_photo_csme": "RT or adjacent HE <=500 microns from center",
    "baseline_photo_cyst": "Definite",
    "baseline_photo_traction": "Cannot grade",
    "baseline_photo_he_plaque_scar": "Cannot grade",
    "baseline_photo_macular_abnormality": "No evidence",
    "baseline_photo_quality": "Good",
}


st.title("Visual Response Predictor")
st.caption("Enter the patient's categorical clinical information.")

with st.form("patient_categorical_inputs"):
    selected_values = {}

    for section_name, section_options in CATEGORICAL_SECTIONS.items():
        st.subheader(section_name)
        header_column, header_category = st.columns([1.25, 2])
        header_column.markdown("**Column**")
        header_category.markdown("**Category**")

        for column_name, categories in section_options.items():
            column_cell, category_cell = st.columns([1.25, 2])
            column_cell.code(column_name, language=None)
            selected_values[column_name] = category_cell.selectbox(
                label=f"Select {column_name}",
                options=categories,
                index=categories.index(DEFAULT_VALUES[column_name]),
                key=column_name,
                label_visibility="collapsed",
            )

    submitted = st.form_submit_button(
        "Save patient inputs",
        type="primary",
        use_container_width=True,
    )

if submitted:
    st.success("Patient inputs saved successfully.")

    try:
       
       ohe_model = load_model(OH_ENCODER_PATH)

       categorical_input = np.array(
            [list(selected_values.values())],
            dtype=object
        )

       transformed = ohe_model.transform(categorical_input)

       if hasattr(transformed, "toarray"):
           transformed = transformed.toarray()


       final_input_array = np.hstack(
            (
                st.session_state.patient_records,
                transformed
            )
        )




       classification_model = load_model(CLASSIFICATION_MODEL)

       y_pred = classification_model.predict(final_input_array)

       predicted_class = y_pred[0]

       st.toast("Processing Completed ✅")

       if predicted_class == 0:
           st.success(f"{CLASS_LABELS.get(predicted_class)}")

       else:
           st.error(f"{CLASS_LABELS.get(predicted_class)}")

       

    except FileNotFoundError:
        st.error(f"Encoder file not found: {OH_ENCODER_PATH}")

    except Exception as error:
        st.error(f"One-hot encoding failed: {error}")