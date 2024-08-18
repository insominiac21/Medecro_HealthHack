import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set page configuration

st.set_page_config(page_title="Disease Predictor",
                   layout="wide",
                   page_icon="🩺")


# Initialize session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Toggle dark mode on button click
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Create a toggle switch for dark mode
st.sidebar.button("Dark Mode/Light Mode", on_click=toggle_dark_mode)

# Apply the appropriate styles based on the dark mode state
if st.session_state.dark_mode:
    # Dark Mode Styles
    st.markdown(
        """
        <style>

          .option-item:hover {
        transform: scale(1.05); /* Blow-out effect */
        transition: transform 0.2s ease-in-out;
        background-color:#4caf50;
        color: #ffffff;
    }
    
/* General page styling */
body {
    background-color: #1c1c1c; /* Dark background color */
    color: #ffffff; /* Light text color */
}

/* Main content area styling */
[data-testid="stAppViewContainer"] {
    background-color: #5f6368;
    padding: 20px;
}

/* Header line styling */
[data-testid="stHeader"] {
    background-color: #263238; /* Dark blue-gray for a cohesive look */
    color: #ffffff;
    padding: 10px;
    border-bottom: 3px solid #4a90e2; /* Lighter blue border for contrast */
}

/* Title styling */
h1,h2,h3,h4,h5,h6 {
    color: #ffffff; /* Light blue color for titles */
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #263238;
    color: #ffffff;
    padding: 20px;
    border-radius:40px;
}

/* Sidebar menu items */
.stOptionMenu > div {
    background-color: #2e2e2e;
    color: #ffffff;
    border-radius: 50px;
    padding: 10px;
    margin-bottom: 10px;
}

/* Input block styling */
.stTextInput > div {
    background-color: #2e2e2e;
    color: #ffffff;
    border-radius: 8px;
    padding: 5px;
    margin-bottom: 20px;
}

[data-testid="sttextinput"]label {
    color: #ffffff;
}

.css-1c5e6hr lable{
color:#ffffff!important;
}

.stTextInput >div >lable {
color:#ffffff;
}

.stNumberInput >div >lable {
color:#ffffff;
}

.stSelectbox >div >lable {
color:#ffffff;
}

.stTextInput > div > input {
    background-color: #444444;
    color: #ffffff;
    border: 1px solid #5a5a5a;
    border-radius: 8px;
    padding: 10px;
}

/* Button styling */
.stButton > button {
    background-color: #2e2e2e;
    color: #ffffff;
    font-size: 16px;
    padding: 10px 20px;
    border-radius: 10px;
    border: none;
    transition: background-color 0.3s ease, font-weight 0.3s ease;
}
.stButton > button:hover {
    background-color: #259ba7;
    color: #000000;
}

/* Success message styling */
.stAlert {
    background-color: #f9f9f9;
    color: #2d7a2d;
    border-left: 5px solid #2d7a2d;
    padding: 15px;
    border-radius: 8px;
}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    # Light Mode Styles
    st.markdown(
        """
        <style>
   .option-item:hover {
        transform: scale(1.05); /* Blow-out effect */
        transition: transform 0.2s ease-in-out;
        background-color:#4caf50;
        color: #ffffff;
    }

          /* General page styling */
        body {
            background-color: #eaf5ff;
            color: #000000;
        }
        /* Main content area styling */
        [data-testid="stAppViewContainer"] {
            background-color: #eaf5ff;
            padding: 20px;
        }
        /* Header line styling */
        [data-testid="stHeader"] {
            background-color: #254d69;
            color: #ffffff;
            padding: 10px;
            border-bottom: 3px solid #0066cc;
        }
        /* Title styling */
        .stTitle {
            color: #4a90e2;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #254d69;
            color: #ffffff;
            padding: 20px;
            border-radius:40px;
        }
        /* Sidebar menu items */
        .stOptionMenu > div {
            background-color: #ffffff;
            color: #4a4a4a;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }
        /* Input block styling */
        .stTextInput > div {
            background-color: #cfe0e8;
            color: #000000;
            border-radius: 8px;
            padding: 5px;
            margin-bottom: 20px;
        }
        .stTextInput > div > label {
            color: #003366;
        }
        .stTextInput > div > input {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #004d99;
            border-radius: 8px;
            padding: 10px;
        }
        /* Button styling */
        .stButton > button {
            background-color: #cfe0e8;
            color: #000000;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            transition: background-color 0.3s ease,font-weight 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #259ba7;
            color:black;
        }

        
        /* Success message styling */
        .stAlert {
            background-color: #dff6f9;
            color: #2d7a2d;
            border-left: 5px solid #2d7a2d;
            padding: 15px;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )



# Load the skin disease prediction model
model_path=r'saved_models/model.h5'
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Preprocess the image with aspect ratio preservation and error handling
def preprocess_image(image):
    # Target size for the model
    target_size = (224, 224)
    # Preserve aspect ratio and pad the rest; fill color is set to black (0)
    image = ImageOps.pad(image, target_size, method=Image.LANCZOS, color=0)
    # Convert image to array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = image_array.astype(np.float32) / 255.0
    # Reshape for the model input
    reshaped_image = normalized_image_array.reshape(1, *target_size, 3)
    return reshaped_image

# Getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))


# Sidebar for navigation
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = 'Diabetes Prediction'

with st.sidebar:
    selected = option_menu('Health Assistant',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Skin Disease Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'sun'],
                           default_index=0,
                           styles={
                               "container": {"padding": "5px","Border-radius":"50px"},
                               "icon": {"color": "#ffa500", "font-size": "25px"},
                               "nav-link": {"font-size": "20px", "text-align": "left", "margin": "5px", "--hover-color": "#cfe0e8", "color": "black"},
                               "nav-link-selected": {"background-color": "#333333","color":"white"},
                               "nav-item": {"margin": "10px", "padding": "5px", "border-radius": "5px", "transition": "all 0.3s ease-in-out"},
                               "option-item": {"margin": "10px", "padding": "10px", "transition": "all 0.3s ease-in-out"},
                           })
    

    st.session_state.selected_page = selected


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # Page title
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1, format="%d", help='e.g., 0 - 20')

    with col2:
        glucose = st.number_input('Glucose Level', min_value=0, max_value=200, step=1, format="%d", help='e.g., 80 - 200 mg/dl')

    with col3:
        bloodpressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=122, step=1, format="%d", help='e.g., 50 - 122 mm Hg')

    with col1:
        skinthickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=99, step=1, format="%d", help='e.g., 10 - 99 mm')

    with col2:
        insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=846, step=1, format="%d", help='e.g., 0 - 846 mu U/ml')

    with col3:
        bmi = st.number_input('BMI (Body Mass Index)', min_value=0.0, max_value=67.1, step=0.1, format="%.1f", help='e.g., 15.0 - 67.1')

    with col1:
        dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.42, step=0.01, format="%.2f", help='e.g., 0.0 - 2.42')

    with col2:
        age = st.number_input('Age (years)', min_value=0, max_value=122, step=1, format="%d", help='e.g., 21 - 122')

    # Code for Prediction
    diabetes_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        # Check if any required inputs are missing
        if any(value == 0 for value in [glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]):
            st.error("Please fill in all the input fields before making a prediction.")
        else:
            user_input = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]

            # You might need to convert categorical values to numerical values if required by your model
            user_input = [float(x) if isinstance(x, (int, float)) else x for x in user_input]

            diabetes_prediction = diabetes_model.predict([user_input])

            if diabetes_prediction[0] == 1:
                diabetes_diagnosis = 'The person has diabetes'
            else:
                diabetes_diagnosis = 'The person does not have diabetes'

    st.success(diabetes_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    # Page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age (years)', min_value=0, max_value=120, step=1, format="%d", help='e.g., 18 - 99 years')

    with col2:
        sex = st.selectbox('Sex', options=['Male', 'Female'], help='Select gender')

    with col3:
        cp = st.selectbox('Chest Pain types', options=['0: Typical Angina', '1: Atypical Angina', '2: Non-Anginal Pain', '3: Asymptomatic'], help='Chest Pain type')

    with col1:
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=200, step=1, format="%d", help='e.g., 90 - 200 mm Hg')

    with col2:
        chol = st.number_input('Serum Cholesterol in mg/dl', min_value=0, max_value=600, step=1, format="%d", help='e.g., 100 - 600 mg/dl')

    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=['0: False', '1: True'], help='Fasting Blood Sugar level')

    with col1:
        restecg = st.selectbox('Resting Electrocardiographic results', options=['0: Normal', '1: Having ST-T wave abnormality', '2: Showing probable or definite left ventricular hypertrophy'], help='Resting Electrocardiographic results')

    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=0, max_value=250, step=1, format="%d", help='e.g., 60 - 250')

    with col3:
        exang = st.selectbox('Exercise Induced Angina', options=['0: No', '1: Yes'], help='Exercise Induced Angina')

    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=6.0, step=0.1, format="%.1f", help='e.g., 0.0 - 6.0')

    with col2:
        slope = st.selectbox('Slope of the peak exercise ST segment', options=['0: Upsloping', '1: Flat', '2: Downsloping'], help='Slope of peak exercise ST segment')

    with col3:
        ca = st.number_input('Major vessels colored by fluoroscopy', min_value=0, max_value=4, step=1, format="%d", help='e.g., 0 - 3')

    with col1:
        thal = st.selectbox('Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect', options=['0: Normal', '1: Fixed defect', '2: Reversible defect'], help='Thalassemia')

    # Code for Prediction
    heart_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        # Check if any required inputs are missing
        if any(value == 0 for value in [age, trestbps, chol, thalach, oldpeak, ca]) or any(value == '' for value in [sex, cp, fbs, restecg, exang, slope, thal]):
            st.error("Please fill in all the input fields before making a prediction.")
        else:
            # Convert categorical values to numerical values
            sex_num = 1 if sex == 'Male' else 0
            cp_num = int(cp.split(':')[0])
            fbs_num = int(fbs.split(':')[0])
            restecg_num = int(restecg.split(':')[0])
            exang_num = int(exang.split(':')[0])
            slope_num = int(slope.split(':')[0])
            thal_num = int(thal.split(':')[0])

            # Prepare the user input list
            user_input = [age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num, thalach, exang_num, oldpeak, slope_num, ca, thal_num]

            # Predict
            heart_prediction = heart_disease_model.predict([user_input])

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)
    
# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    # Page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo (Hz)', min_value=0.0, format="%.2f", help='e.g., 50.0 - 200.0 Hz')

    with col2:
        fhi = st.number_input('MDVP:Fhi (Hz)', min_value=0.0, format="%.2f", help='e.g., 100.0 - 400.0 Hz')

    with col3:
        flo = st.number_input('MDVP:Flo (Hz)', min_value=0.0, format="%.2f", help='e.g., 50.0 - 150.0 Hz')

    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter (%)', min_value=0.0, format="%.2f", help='e.g., 0.0 - 10.0 %')

    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter (Abs)', min_value=0.0, format="%.2f", help='e.g., 0.0 - 0.2')

    with col1:
        RAP = st.number_input('MDVP:RAP', min_value=0.0, format="%.2f", help='e.g., 0.0 - 0.2')

    with col2:
        PPQ = st.number_input('MDVP:PPQ', min_value=0.0, format="%.2f", help='e.g., 0.0 - 0.2')

    with col3:
        DDP = st.number_input('Jitter:DDP', min_value=0.0, format="%.2f", help='e.g., 0.0 - 1.0')

    with col4:
        Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, format="%.2f", help='e.g., 0.0 - 0.2')

    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer (dB)', min_value=0.0, format="%.2f", help='e.g., 0.0 - 5.0')

    with col1:
        APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, format="%.2f", help='e.g., 0.0 - 0.2')

    with col2:
        APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, format="%.2f", help='e.g., 0.0 - 0.2')

    with col3:
        APQ = st.number_input('MDVP:APQ', min_value=0.0, format="%.2f", help='e.g., 0.0 - 0.2')

    with col4:
        DDA = st.number_input('Shimmer:DDA', min_value=0.0, format="%.2f", help='e.g., 0.0 - 0.2')

    with col5:
        NHR = st.number_input('NHR', min_value=0.0, format="%.2f", help='e.g., 0.0 - 0.2')

    with col1:
        HNR = st.number_input('HNR', min_value=0.0, format="%.2f", help='e.g., 0.0 - 20.0')

    with col2:
        RPDE = st.number_input('RPDE', min_value=0.0, format="%.2f", help='e.g., 0.0 - 1.0')

    with col3:
        DFA = st.number_input('DFA', min_value=0.0, format="%.2f", help='e.g., 0.0 - 1.0')

    with col4:
        spread1 = st.number_input('spread1', min_value=0.0, format="%.2f", help='e.g., 0.0 - 1.0')

    with col5:
        spread2 = st.number_input('spread2', min_value=0.0, format="%.2f", help='e.g., 0.0 - 1.0')

    with col1:
        D2 = st.number_input('D2', min_value=0.0, format="%.2f", help='e.g., 0.0 - 10.0')

    with col2:
        PPE = st.number_input('PPE', min_value=0.0, format="%.2f", help='e.g., 0.0 - 5.0')

    # Code for Prediction
    parkinsons_diagnosis = ''

    # Creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        # Check if any input is empty
        if any(value == 0 for value in [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]):
            st.error("Please fill in all the input fields before making a prediction.")
        else:
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                          RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                          APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

            user_input = [float(x) for x in user_input]

            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)


# Skin Disease Prediction Page
if selected == "Skin Disease Prediction":

    st.title("Skin Disease Prediction using Deep Learning")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("Classifying...")
            preprocessed_image = preprocess_image(image)
            prediction = model.predict(preprocessed_image)
            predicted_class = np.argmax(prediction)

            # Class labels
            class_labels = [
                'Benign keratosis-like lesions', 
                'Melanocytic nevi', 
                'Dermatofibroma', 
                'Melanoma', 
                'Vascular lesions', 
                'Basal cell carcinoma (BCC)',  
                'Actinic keratoses (AKs)'
            ]

            st.write(f"Prediction: {class_labels[predicted_class]}")

            # Details for each class
            details = {
                'Benign keratosis-like lesions': {
                    'Diagnosis': """Medical Examination: A dermatologist or healthcare provider should visually inspect the lesion and may perform additional tests like a biopsy if necessary.
    Differential Diagnosis: Differentiating between various benign conditions (like seborrheic keratosis, actinic keratosis, verruca vulgaris, etc.) is crucial for appropriate treatment.""",
                    'Remedies and Treatments': """1. Observation: In some cases, especially if the lesion is small and stable, observation without treatment might be recommended.
    2. Topical Treatments:
    - Topical Retinoids: Prescription retinoid creams can help in some cases, such as for actinic keratosis.
    - Cryotherapy: Freezing the lesion with liquid nitrogen can be effective for certain types of growths.
    - Topical Chemotherapy: For more persistent lesions, topical chemotherapy creams may be prescribed.
    3. Surgical Procedures:
    - Curettage: Scraping off the lesion with a curette.
    - Electrosurgery: Using electricity to burn off or destroy the lesion.
    - Laser Therapy: Laser treatment to remove the lesion.
    4. Home Remedies: While not substitutes for medical treatment, certain home remedies might alleviate symptoms temporarily, such as moisturizing the affected area or using over-the-counter hydrocortisone cream (under medical guidance)."""
                },
                'Melanocytic nevi': {
                    'Diagnosis': """1. Visual Examination: A healthcare provider typically diagnoses melanocytic nevi through visual inspection. They look for features such as asymmetry, irregular borders, uneven coloration, and changes in size or shape.
    2. Dermoscopy: In cases where the diagnosis is unclear, dermoscopy—a non-invasive technique using a handheld device with magnification and light—can aid in examining the mole's structure and pigment patterns.
    3. Biopsy: If a mole shows suspicious features or changes, a biopsy may be recommended. A small sample of tissue is removed and examined under a microscope to confirm the diagnosis and rule out melanoma.""",
                    'Remedies and Management': """1. Observation: Most melanocytic nevi are benign and require no treatment other than periodic observation. Your healthcare provider may recommend regular skin checks to monitor any changes.
    2. Surgical Removal: This may be recommended if the mole shows concerning features (e.g., asymmetry, irregular borders, changes in color or size). Surgical removal can be done through excision (cutting out the mole and stitching the skin) or shave biopsy (shaving off the mole with a scalpel).
    3. Cosmetic Removal: If the mole is bothersome cosmetically (e.g., it's in a prominent location or frequently irritated), it can be removed for aesthetic reasons.
    4. Sun Protection: Since sun exposure can contribute to the development of new moles and potentially melanoma, practicing sun safety (using sunscreen, wearing protective clothing, avoiding peak sun hours) is essential."""
                },
                'Dermatofibroma': {
                    'Diagnosis': """1. Clinical Examination: A dermatologist usually diagnoses dermatofibromas based on their typical appearance and texture. They may use a dermatoscope (a handheld instrument with light and magnification) to examine the lesion closely.
    2. Biopsy: If the diagnosis is uncertain or if there are atypical features, a biopsy may be performed. A small sample of tissue is taken and examined under a microscope to confirm the diagnosis and rule out other conditions.""",
                    'Remedies and Management': """1. Observation: Dermatofibromas are generally harmless and may not require treatment unless they cause symptoms or cosmetic concerns. Your healthcare provider may recommend periodic monitoring to check for changes.
    2. Symptomatic Relief: If the dermatofibroma is causing itching or irritation, topical treatments such as corticosteroid creams or moisturizers may provide relief.
    3. Surgical Excision: If the dermatofibroma is bothersome cosmetically or if there is uncertainty about the diagnosis, surgical removal (excision) may be recommended. This involves cutting out the lesion and closing the skin with stitches.
    4. Cryotherapy: Freezing the dermatofibroma with liquid nitrogen may be an option for removal, particularly for smaller lesions."""
                },
                'Melanoma': {
                    'Diagnosis': """1. Clinical Examination: A dermatologist or healthcare provider will conduct a thorough examination of your skin, including any suspicious moles or lesions.
    2. Dermoscopy: Dermoscopy is a non-invasive technique that allows dermatologists to examine skin lesions with a magnifying tool equipped with light, aiding in the detection of suspicious features.
    3. Biopsy: If a lesion is suspected to be melanoma based on its appearance or changes, a biopsy is performed. This involves removing a sample of tissue from the lesion and examining it under a microscope to confirm the diagnosis.""",
                    'Remedies and Management': """1. Surgical Excision: The primary treatment for melanoma is surgical removal. Depending on the size and depth of the melanoma, the surgical procedure may involve removing the lesion along with a margin of normal skin (wide local excision).
    2. Sentinel Lymph Node Biopsy: In cases where melanoma is thicker or has spread deeper into the skin, a sentinel lymph node biopsy may be recommended to determine if the cancer has spread to nearby lymph nodes.
    3. Adjuvant Therapy: In certain cases of melanoma, especially those with a higher risk of recurrence or spread, adjuvant therapies such as targeted therapy or immunotherapy may be recommended after surgery to reduce the risk of recurrence.
    4. Radiation Therapy: Radiation therapy may be used in some cases to treat melanoma that cannot be completely removed with surgery or to target areas where melanoma has spread."""
                },
                'Vascular lesions': {
                    'Diagnosis': """1. Clinical Examination: A dermatologist or vascular specialist will visually inspect the lesion and may use a dermatoscope for a closer examination.
    2. Imaging Studies: For deeper lesions or those affecting internal organs, imaging techniques such as ultrasound, MRI, or CT scans may be used to assess the extent and characteristics of the vascular lesion.
    3. Biopsy: In some cases, a biopsy may be performed to confirm the diagnosis, especially if the lesion appears atypical or is suspected to be a vascular tumor.""",
                    'Remedies and Management': """1. Observation: Many vascular lesions are harmless and may not require treatment unless they cause symptoms or cosmetic concerns. Regular monitoring may be recommended.
    2. Topical Treatments: Some superficial vascular lesions, such as small hemangiomas or telangiectasias, may respond to topical treatments like laser therapy or topical medications (e.g., timolol gel for infantile hemangiomas).
    3. Laser Therapy: Laser treatments, such as pulsed dye laser (PDL) or Nd:YAG laser, are often effective for treating vascular lesions like port-wine stains, spider veins, and certain hemangiomas.
    4. Sclerotherapy: This involves injecting a sclerosing agent into the blood vessel to shrink and eventually eliminate small to medium-sized vascular malformations or spider veins.
    5. Surgical Excision: Surgical removal may be considered for larger or deeper vascular lesions, particularly if they are causing symptoms or cosmetic concerns.
    6. Embolization: For complex vascular malformations or lesions that bleed excessively, embolization may be performed to block off the abnormal blood vessels."""
                },
                'Basal cell carcinoma (BCC)': {
                    'Diagnosis': """1. Clinical Examination: A dermatologist or healthcare provider will examine suspicious lesions on the skin. BCC often appears as a pinkish or pearly bump with a rolled edge, or as a flat, scaly, reddish patch.
    2. Biopsy: If a lesion is suspected to be basal cell carcinoma based on its appearance or changes, a biopsy is performed. A small sample of tissue is removed from the lesion and examined under a microscope by a pathologist to confirm the diagnosis.""",
                    'Remedies and Management': """1. Surgical Excision: The primary treatment for basal cell carcinoma is surgical removal. The procedure involves cutting out the cancerous tissue along with a surrounding margin of healthy skin. This is typically done under local anesthesia.
    2. Mohs Surgery: Mohs micrographic surgery is a specialized technique used for BCC and other skin cancers, especially on the face or areas where tissue conservation is crucial. It involves removing thin layers of tissue and examining them under a microscope immediately, layer by layer, until no cancer cells remain.
    3. Electrodessication and Curettage (ED&C): For small, superficial BCCs, ED&C may be used. The lesion is scraped with a curette (a sharp, spoon-shaped instrument) to remove cancerous tissue, followed by electrodessication to destroy any remaining cancer cells.
    4. Topical Treatments: In certain cases of superficial BCCs or for patients who are not suitable candidates for surgery, topical medications like imiquimod or 5-fluorouracil (5-FU) may be prescribed to apply directly to the lesion over several weeks.
    5. Radiation Therapy: Radiation therapy may be considered for elderly patients or those with medical conditions that make surgery difficult. It can also be used for areas where surgical removal is challenging, such as around the eyes or nose."""
                },
                'Actinic keratoses (AKs)': {
                    'Diagnosis': """1. Clinical Examination: A dermatologist or healthcare provider will visually inspect the skin for signs of actinic keratoses. They may use a dermatoscope for a closer examination to differentiate AKs from other skin conditions.
    2. Biopsy: If a lesion appears suspicious or if there is uncertainty about the diagnosis, a biopsy may be performed. A small sample of tissue is removed and examined under a microscope to confirm the presence of AK and rule out skin cancer.""",
                    'Remedies and Management': """1. Topical Treatments:
    - Topical Retinoids: Prescription retinoid creams (e.g., tretinoin) help to normalize cell turnover and reduce the number of AK lesions.
    - Topical Chemotherapy: Creams containing 5-fluorouracil (5-FU) or imiquimod stimulate the immune system to attack AK cells.
    - Diclofenac Gel: This topical treatment has anti-inflammatory properties and is used specifically for AKs.
    2. Cryotherapy: Liquid nitrogen is applied to freeze and destroy AKs. This is a common treatment for isolated lesions.
    3. Photodynamic Therapy (PDT): This involves applying a photosensitizing agent to the skin and then exposing it to a light source that activates the agent, selectively destroying AK cells.
    4. Chemical Peels: For widespread AKs, chemical peels may be used to remove the top layers of skin, promoting new skin growth.
    5. Surgical Removal: In cases where AKs have progressed to become thick or develop into squamous cell carcinoma, surgical removal may be necessary."""
                }
            }

            # Display details for the predicted class
            if class_labels[predicted_class] in details:
                detail = details[class_labels[predicted_class]]
                for key, value in detail.items():
                    with st.expander(key):
                        st.write(value)

        except Exception as e:
            
            st.error(
                """
                An error occurred while processing the image:
                - Please ensure the image is not too small or too large and be closest to being a square.
                - Ideal image sizes range from small (e.g., 100x100 pixels) to moderately large (e.g., 1000x1000 pixels).
                - Extremely large images or those with unusual aspect ratios might cause processing issues.
                
                **Error details:** {}
                """.format(e)
            )

