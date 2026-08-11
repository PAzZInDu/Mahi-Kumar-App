import streamlit as st


st.set_page_config(
    page_title="Visual Response Predictor",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {max-width: 900px; padding-top: 3rem; padding-bottom: 3rem;}
    [data-testid="stAppViewContainer"] {background: #050505;}
    [data-testid="stSidebar"] {background: #0a0a0a; border-right: 1px solid #333333;}
    .hero {
        padding: 2.2rem; border-radius: 24px;
        background: linear-gradient(135deg, #171717 0%, #0d0d0d 100%);
        border: 1px solid #333333; margin-bottom: 1.5rem;
    }
    .hero-badge {
        display: inline-block; padding: .35rem .75rem; border-radius: 999px;
        background: linear-gradient(90deg, #7f1d1d, #450a0a); color: #fecaca;
        border: 1px solid #991b1b; font-size: .82rem;
        font-weight: 700; letter-spacing: .03em; margin-bottom: .8rem;
    }
    .hero h1 {font-size: 2.45rem; line-height: 1.12; margin: 0 0 .8rem; color: #f1f5f9;}
    .hero p {font-size: 1.05rem; line-height: 1.65; color: #c7c7c7; margin: 0;}
    .step-card {
        min-height: 170px; padding: 1.25rem; border-radius: 18px;
        background: #121212; border: 1px solid #333333;
        box-shadow: 0 8px 24px rgba(0, 0, 0, .22);
    }
    .step-number {
        width: 34px; height: 34px; display: inline-flex; align-items: center;
        justify-content: center; border-radius: 10px;
        background: linear-gradient(135deg, #ef4444, #991b1b);
        color: #fff1f2; font-weight: 700; margin-bottom: .7rem;
    }
    .step-card h3 {font-size: 1.05rem; margin: 0 0 .45rem; color: #f1f5f9;}
    .step-card p {font-size: .92rem; line-height: 1.5; color: #b8b8b8; margin: 0;}
    .privacy-note {
        margin-top: 1.5rem; padding: 1rem 1.15rem; border-radius: 14px;
        background: #121212; border: 1px solid #333333;
        border-left: 4px solid #f59e0b; color: #c7c7c7;
    }
    button[kind="primary"] {
        background: linear-gradient(90deg, #ef4444 0%, #b91c1c 55%, #7f1d1d 100%);
        border: 1px solid #ef4444; color: #ffffff;
    }
    button[kind="primary"]:hover {border-color: #fca5a5; filter: brightness(1.08);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">CLINICAL DECISION SUPPORT</div>
        <h1>Visual Response Predictor</h1>
        <p>
            Organize baseline patient information and estimate whether a study eye is
            likely to show a major visual response following treatment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("How it works")
step_one, step_two, step_three = st.columns(3)

with step_one:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">1</div>
            <h3>Enter measurements</h3>
            <p>Add baseline health, eye-pressure, OCT, and vision-test measurements.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_two:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">2</div>
            <h3>Add clinical findings</h3>
            <p>Select treatment, examination, OCT, and fundus-photography findings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with step_three:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">3</div>
            <h3>Review the estimate</h3>
            <p>Use the model result as supporting information alongside clinical judgement.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="privacy-note">
        <strong>Important:</strong> This tool provides an estimate from a trained model.
        It does not provide a diagnosis or replace advice from a qualified clinician.
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Welcome")

if not st.user.is_logged_in:
    st.sidebar.write("Sign in securely to enter patient information and use the predictor.")
    if st.sidebar.button(
        "Log in with Google",
        type="primary",
        icon=":material/login:",
        use_container_width=True,
    ):
        st.login()
    st.info("Please log in with Google to begin the two-step assessment.", icon="🔐")
else:
    user_name = getattr(st.user, "name", None) or "there"
    st.sidebar.success(f"Signed in as {user_name}")

    if st.sidebar.button(
        "Log out",
        type="secondary",
        icon=":material/logout:",
        use_container_width=True,
    ):
        st.logout()
        st.stop()

    st.success(f"Welcome, {user_name}. You can begin with Step 1.", icon="✅")
    if st.button(
        "Start Patient Measurements →",
        type="primary",
        use_container_width=True,
    ):
        st.switch_page("pages/00_Patient_Measurements.py")
