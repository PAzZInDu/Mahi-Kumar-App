import pickle

import numpy as np
import pandas as pd
import streamlit as st

# Require user authentication
if not st.user.is_logged_in:
    st.error("Please log in to access the app.")
    st.stop()


st.set_page_config(
    page_title="Visual Response Predictor",
    page_icon="👁️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main .block-container {max-width: 900px; padding-top: 2rem;}
    [data-testid="stAppViewContainer"] {background: #050505;}
    div[data-testid="stForm"] {background: #0f0f0f; border: 1px solid #333333; border-radius: 18px; padding: 1.4rem;}
    div[data-testid="stSelectbox"] label {font-weight: 600;}
    .info-card {padding: 1rem 1.2rem; border-radius: 14px; background: #171717;
        border: 1px solid #14532d; border-left: 5px solid #34d399;
        color: #d1fae5; margin: .5rem 0 1.5rem;}
    div[data-testid="stSelectbox"]:focus-within {border-color: #34d399; border-radius: 10px;}
    button[kind="primary"] {
        background: linear-gradient(90deg, #ef4444 0%, #b91c1c 55%, #7f1d1d 100%);
        border: 1px solid #ef4444; color: #ffffff;
    }
    button[kind="primary"]:hover {border-color: #fca5a5; filter: brightness(1.08);}
    </style>
    """,
    unsafe_allow_html=True,
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

SECTION_HELP = {
    "Patient and treatment": "Basic patient information, the medicine planned for the study eye, and current health history.",
    "Eye history and examination": "What the clinician observed during the baseline eye examination.",
    "OCT findings": "Details from the retinal OCT scan, including fluid, traction, and scan quality.",
    "Fundus photography": "Grading results from photographs of the back of the study eye.",
}

SECTION_ICONS = {
    "Patient and treatment": "💊",
    "Eye history and examination": "🩺",
    "OCT findings": "👁️",
    "Fundus photography": "📸",
}

FIELD_DETAILS = {
    "treatment_group": ("Planned anti-VEGF medicine", "Select the injection medicine assigned for treatment."),
    "study_eye": ("Study eye", "OD means right eye; OS means left eye."),
    "NSEyeAntiVEGFInj": ("Other-eye anti-VEGF injections", "Has the eye not being assessed received an anti-VEGF injection?"),
    "Gender": ("Sex", "Select the sex recorded in the patient's clinical record."),
    "Ethnicity": ("Ethnicity", "Choose the category documented in the patient record."),
    "Race": ("Race", "Choose the category documented in the patient record."),
    "DiabetesType": ("Type of diabetes", "Select Type 1, Type 2, or Uncertain according to the diagnosis."),
    "InsulinUsed": ("Currently uses insulin", "Select Yes if insulin is part of the patient's current treatment."),
    "PtCurrMed": ("Currently taking medication", "Select Yes if current medications are documented."),
    "PreExistMedCond": ("Other existing medical conditions", "Select Yes if the patient had another medical condition before treatment."),
    "study_eye_dmetrthx": ("Previous DME treatment", "Has the study eye previously been treated for diabetic macular edema (DME)?"),
    "IOPMeasured": ("Eye-pressure measurement method", "Select the instrument used to measure pressure inside the study eye."),
    "IOPDilation": ("When eye pressure was measured", "Was pressure measured before or after the pupil was dilated?"),
    "study_eye_lensstatus": ("Natural or artificial lens", "Phakic means the natural lens is present; PC IOL means an artificial lens was implanted after cataract surgery."),
    "study_eye_lensopac": ("Lens cloudiness", "Choose whether lens cloudiness affected vision at the baseline examination."),
    "study_eye_vitreoushemcond": ("Bleeding in the vitreous", "Select Yes if vitreous hemorrhage was observed in the study eye."),
    "study_eye_vitreoushaze": ("Vitreous clarity", "Choose how clearly the clinician could see through the gel inside the eye."),
    "study_eye_drseverity": ("Diabetic retinopathy severity", "Select the clinician's severity grade. NPDR and PDR are stages of diabetic retinopathy."),
    "study_eye_dmeclinexam": ("DME on clinical examination", "Choose the finding recorded for diabetic macular edema."),
    "study_eye_dmechar": ("Pattern of macular swelling", "Select whether retinal swelling was mainly focal, diffuse, or intermediate."),
    "study_eye_postsegfundabno": ("Other back-of-eye abnormality", "Select Yes if another abnormality was found in the posterior segment."),
    "OCTMachine": ("Clinic OCT machine", "Select the machine used for the clinic's retinal scan."),
    "baseline_intraretinal_edema": ("Fluid within the retina", "Select Yes if the OCT showed swelling or fluid inside the retinal layers."),
    "baseline_subretinal_fluid": ("Fluid beneath the retina", "Choose the exact finding shown in the OCT report."),
    "baseline_epiretinal_membrane": ("Membrane on the retina", "Select Yes if an epiretinal membrane was visible on the OCT scan."),
    "baseline_vitreomacular_traction": ("Vitreomacular traction", "Select Yes if the vitreous was pulling on the macula."),
    "baseline_rc_oct_machine": ("Reading-centre OCT machine", "Select the machine named in the independent reading-centre report."),
    "baseline_rc_oct_scan_quality": ("OCT scan quality", "Choose the reading centre's assessment of whether the scan could be graded."),
    "baseline_photo_dr_severity": ("Photo retinopathy grade", "Copy the diabetic-retinopathy level from the fundus-photo grading report."),
    "baseline_photo_csme": ("Clinically significant swelling", "Select the report description for retinal thickening (RT) or hard exudates (HE) near the centre."),
    "baseline_photo_cyst": ("Cyst-like changes", "Select the cyst grading reported from the baseline photographs."),
    "baseline_photo_traction": ("Retinal traction", "Select the traction finding in the photography report."),
    "baseline_photo_he_plaque_scar": ("Plaque or scar", "Select the hard-exudate plaque or fibrous-scar finding."),
    "baseline_photo_macular_abnormality": ("Other macular abnormality", "Select whether another abnormality was visible in the central retina."),
    "baseline_photo_quality": ("Fundus photo quality", "Choose the grader's assessment of the baseline photograph quality."),
}


st.title("👁️ Treatment Response Check")
st.markdown(
    """
    <div class="info-card"><strong>Step 2 of 2</strong><br>
    Add the patient's treatment, examination, scan, and eye-photography details. The app
    will combine them with the measurements from Step 1 to estimate likely visual response.</div>
    """,
    unsafe_allow_html=True,
)
st.caption("Use the ⓘ beside each question for a plain-language explanation. Choose only values supported by the clinical record.")

if not len(st.session_state.get("patient_records", [])):
    st.error(
        "You cannot continue to Step 2 yet. Complete and save all patient "
        "measurements in Step 1 first."
    )
    st.page_link(
        "pages/00_Patient_Measurements.py",
        label="← Go to Patient Measurements",
        icon="📋",
    )
    st.stop()

with st.form("patient_categorical_inputs"):
    selected_values = {}

    for section_name, section_options in CATEGORICAL_SECTIONS.items():
        st.subheader(f"{SECTION_ICONS[section_name]} {section_name}")
        st.caption(SECTION_HELP[section_name])

        for column_name, categories in section_options.items():
            friendly_label, help_text = FIELD_DETAILS[column_name]
            selected_values[column_name] = st.selectbox(
                label=friendly_label,
                options=categories,
                index=categories.index(DEFAULT_VALUES[column_name]),
                key=column_name,
                help=help_text,
            )

    submitted = st.form_submit_button(
        "Check likely treatment response",
        type="primary",
        use_container_width=True,
    )

if submitted:
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
           st.warning(
               f"Prediction: {CLASS_LABELS.get(predicted_class)}. "
               "This estimate should support—not replace—a clinician's judgement."
           )

       else:
           st.success(
               f"Prediction: {CLASS_LABELS.get(predicted_class)}. "
               "This estimate should support—not replace—a clinician's judgement."
           )

       

    except FileNotFoundError:
        st.error("A required prediction file could not be loaded. Please contact the app administrator.")

    except Exception as error:
        st.error("The prediction could not be completed. Please check the saved measurements and try again.")
        with st.expander("Technical details"):
            st.code(str(error))
