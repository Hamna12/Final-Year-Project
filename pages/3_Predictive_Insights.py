import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from chat_llm import generate_response 

# Load the model, preprocessor, and label encoder
model = joblib.load('Data/lgb_model.pkl')
preprocessor = joblib.load('Data/preprocessor.pkl')
label_encoder = joblib.load('Data/label_encoder.pkl')

# Load dataset for feature ranges
df_filtered = pd.read_csv("Data/filtered_dataset.csv", encoding="ISO-8859-1")

X = df_filtered.drop('Grade', axis=1)
y = df_filtered['Grade']

categorical_features = ['sex', 'group', 'is_urban']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Define feature categories
feature_categories = {
    'Demographics': ['sex', 'group', 'is_urban'],
    'Socio-Economic': [
        'urban_rural_count(%)', 'urban_fraction_yearly', 'rural_fraction_yearly',
        'government_non_government_count(%)', 'government_fraction_yearly', 'non_government_fraction_yearly',
        'HI', 'PGI', 'SPG', 'SI', 'SST', 'WI', 'HI_tehsil', 'PGI_tehsil', 'SPG_tehsil', 'SI_tehsil', 'SST_tehsil', 'WI_tehsil'
    ],
    'Educational Statistics': [
        'is_government', 'Year', 'district_student_fraction', 'tehsil_student_fraction',
        'district_student_count_yearly', 'tehsil_student_count_yearly', 'fail_district_fraction',
        'pass_district_fraction', 'fail_tehsil_fraction', 'pass_tehsil_fraction'
    ]
}

# Ensure all columns used in training are in the user input dataframe
all_features = set(X.columns)
user_inputs = {feature: 0 for feature in all_features}  # Initialize with default values

# Streamlit app
st.title('Student Grade Prediction and Insights')

# Sidebar for user inputs
st.sidebar.header('User Input Features')

# Function to process user input and make prediction
def make_prediction(user_inputs):
    user_input_df = pd.DataFrame(user_inputs, index=[0])
    user_input_processed = preprocessor.transform(user_input_df)
    prediction_numeric = model.predict(user_input_processed)
    prediction_category = label_encoder.inverse_transform(prediction_numeric)
    return prediction_category[0], user_input_processed

# Function to display SHAP waterfall plot
def display_shap_waterfall_plot(shap_values, processed_input, max_display, selected_features, class_idx):
    expl = shap.Explanation(
        values=shap_values[class_idx][0],
        base_values=explainer.expected_value[class_idx],
        data=processed_input[0],
        feature_names=selected_features
    )
    fig, ax = plt.subplots()
    shap.waterfall_plot(expl, max_display=max_display, show=False)
    st.pyplot(fig)

# Create dropdowns for each category
for category, features in feature_categories.items():
    with st.sidebar.expander(category):
        for feature in features:
            if feature in categorical_features:
                user_inputs[feature] = st.selectbox(f"{feature}", options=X[feature].unique(), key=feature)
            else:
                min_value = float(X[feature].min())
                max_value = float(X[feature].max())
                if min_value == max_value:
                    min_value -= 1
                    max_value += 1
                user_inputs[feature] = st.slider(f"{feature}", min_value=min_value, max_value=max_value, value=float(X[feature].mean()), key=feature)

# Button to predict demographics
if st.sidebar.button("Prediction on Educational Statistics"):
    category_prediction, user_input_processed = make_prediction(user_inputs)
    st.write(f"Predicted Grade for Educational Statistics: {category_prediction}")

    # Display SHAP values for selected features
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_input_processed)

    class_idx = 0  # Assuming class index 0, adjust based on your requirement
    if isinstance(shap_values, list):
        shap_values = shap_values[class_idx]  # Select SHAP values for the specific class
        base_value = explainer.expected_value[class_idx]
    else:
        base_value = explainer.expected_value

    st.subheader('SHAP Value Explanation')
    st.write("SHAP values show the impact of each feature on the prediction. Positive SHAP values push the prediction higher, while negative SHAP values push it lower.")
    display_shap_waterfall_plot(shap_values, user_input_processed, max_display=15, selected_features=features, class_idx=class_idx)


# Single prediction button for all features
if st.sidebar.button("Predict All"):
    overall_prediction, user_input_processed = make_prediction(user_inputs)
    st.write(f"Overall Predicted Grade: {overall_prediction}")
    
    # Load precomputed SHAP values
    shap_values = joblib.load('Data/shap_values.pkl')
    X_test = pd.read_csv('Data/X_test.csv')

    st.subheader('SHAP Value Explanation')
    st.write("SHAP values show the impact of each feature on the prediction. Positive SHAP values push the prediction higher, while negative SHAP values push it lower.")
    selected_index = st.selectbox("Select data point index:", options=list(range(len(shap_values))))
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values[selected_index], show=False)
    st.pyplot(fig)

    with st.expander('SHAP Summary Plot'):
        # Display SHAP summary plot for feature importance
        st.subheader('SHAP Summary Plot')
        fig_summary, ax_summary = plt.subplots()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(fig_summary)

# Main performance metrics
st.subheader('Model Performance')
X_processed = preprocessor.transform(X)
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert predictions and actual values to grades
y_pred_grades = label_encoder.inverse_transform(y_pred)
y_test_grades = label_encoder.inverse_transform(y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Detailed analysis and visualizations
st.subheader("Explore Further")
with st.expander("Confusion Matrix and Classification Report"):
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_grades, y_pred_grades, labels=label_encoder.classes_)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    st.write('Confusion Matrix:')
    st.dataframe(conf_matrix_df)

    # Plot confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix, cmap='coolwarm')
    plt.title('Confusion matrix of the classifier')
    plt.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

    # # Display classification report
    # st.write('Classification Report:')
    # classification_rep = classification_report(y_test_grades, y_pred_grades, target_names=label_encoder.classes_, output_dict=True)
    # st.json(classification_rep)

with st.expander("Class Distribution"):
    # Class distribution plot
    pred_counts = pd.Series(y_pred_grades).value_counts().sort_index()
    actual_counts = pd.Series(y_test_grades).value_counts().sort_index()
    fig, ax = plt.subplots()
    width = 0.35
    ax.bar(pred_counts.index, pred_counts.values, width=width, label='Predicted')
    ax.bar(actual_counts.index, actual_counts.values, width=width, label='Actual', bottom=pred_counts.values)
    ax.set_xlabel('Grade')
    ax.set_ylabel('Count')
    ax.legend()
    plt.title('Predicted vs Actual Class Distribution')
    st.pyplot(fig)

    # Grade Distribution Histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    actual_grades = label_encoder.inverse_transform(y_test)
    predicted_grades = label_encoder.inverse_transform(y_pred)

    ax.hist([actual_grades, predicted_grades], bins=len(label_encoder.classes_), label=['Actual', 'Predicted'], alpha=0.7)
    ax.set_xlabel('Grade')
    ax.set_ylabel('Count')
    ax.set_title('Actual vs Predicted Grade Distribution')
    ax.legend()
    st.pyplot(fig)

with st.expander("One-vs-Rest ROC Curve"):
    # One-vs-Rest ROC Curve
    y_bin = label_binarize(y_encoded, classes=[0, 1, 2, 3, 4, 5])
    n_classes = y_bin.shape[1]

    X_train, X_test, y_train_bin, y_test_bin = train_test_split(X_processed, y_bin, test_size=0.2, random_state=42)
    clf = OneVsRestClassifier(LogisticRegression(random_state=42))
    y_score = clf.fit(X_train, y_train_bin).predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label=f'Grade {label_encoder.classes_[i]} (area = {roc_auc[i]:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('One-vs-Rest ROC Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)

with st.expander("Class Balance in Dataset"):
    # Class Balance Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    y_counts = pd.Series(y).value_counts()
    ax.bar(y_counts.index, y_counts.values, alpha=0.7)
    ax.set_xlabel('Grade')
    ax.set_ylabel('Count')
    ax.set_title('Class Balance in Dataset')
    st.pyplot(fig)

with st.expander("Prediction Errors"):
    # Prediction Errors Plot
    errors = pd.DataFrame({'Actual': label_encoder.inverse_transform(y_test),
                           'Predicted': label_encoder.inverse_transform(y_pred)})
    errors['Error'] = errors['Actual'] != errors['Predicted']

    error_counts = errors[errors['Error']].groupby(['Actual', 'Predicted']).size().reset_index(name='Count')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(error_counts['Actual'] + ' to ' + error_counts['Predicted'], error_counts['Count'], alpha=0.7)
    ax.set_xlabel('Count')
    ax.set_title('Prediction Errors')
    st.pyplot(fig)

# Divider for chatbot section
st.markdown("---")

st.markdown("## 🤖 Chat with EduBot")
st.write("Ask any questions related to the project, and our chatbot will provide insights.")

# Input for the chatbot
user_input = st.text_input("Enter your question here", key="chat_input")

# Button to get response from the chatbot
if st.button("Get Response", key='chat_button'):
    if user_input:
        with st.spinner('Generating response...'):
            response = generate_response(user_input)
        st.write("### Chatbot Response:")
        st.write(response)
    else:
        st.warning("Please enter a question to get a response.")
