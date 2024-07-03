# pages/Project_Overview.py

import streamlit as st
from chat_llm import generate_response 

st.set_page_config(page_title="Project Overview", page_icon="🌍", layout="wide")

st.markdown("# 🌍 Project Overview")
st.sidebar.header("Project Overview")

# Main content
st.write(
    """
    ## Project Objectives
    This project aims to analyze the performance of students in various tehsils based on demographic and socio-economic data. The primary goals are:
    - To identify factors that influence student grades.
    - To highlight areas with low living standards and suggest targeted interventions.
    - To provide actionable insights for stakeholders to improve student performance.
    """
)

# Collapsible section for Data Sources
with st.expander("### Data Sources", expanded=False):
    st.write(
        """
        1. **Demographic Data**: Contains information on demographics for all tehsils.
        2. **Socio-Economic Data**: Contains socio-economic information for three specific tehsils.
        """
    )

# Collapsible section for Methodology
with st.expander("### Methodology", expanded=False):
    st.write(
        """
        - **Data Collection**: Data was collected from multiple sources.
        - **Feature Engineering**: Interaction and polynomial features were generated.
        - **Model Training**: A LightGBM model was trained to predict student grades.
        - **Insights Generation**: Insights were generated using SHAP values and other techniques.
        """
    )

# # Initialize ChatBot with custom data
# chatbot_component = ChatBotComponent(custom_data)

# # Chat interface
# if 'responses' not in st.session_state:
#     st.session_state['responses'] = []

# st.title("Ask Questions")
# user_input = st.text_input("You: ", "")

# if st.button("Send"):
#     if user_input.strip() != "":
#         response = chatbot_component.get_response(user_input)
#         st.session_state['responses'].append(f"You: {user_input}")
#         st.session_state['responses'].append(f"Bot: {response}")
#     else:
#         st.warning("Please enter a message.")

# for response in st.session_state['responses']:
#     st.write(response)

#commenting
# Initialize ChatBotComponent
# chatbot_component = ChatBotComponent()

# # Display chat interface
# st.title("Chat with our Bot")
# if 'responses' not in st.session_state:
#     st.session_state['responses'] = []

# user_input = st.text_input("You: ", "")
# if st.button("Send"):
#     response = chatbot_component.get_response(user_input)
#     st.session_state['responses'].append(f"You: {user_input}")
#     st.session_state['responses'].append(f"Bot: {response}")

# for response in st.session_state['responses']:
#     if response.startswith("You:"):
#         message(response[5:], is_user=True)  # Display user message, remove "You: " prefix
#     else:
#         message(response[5:])  # Display bot message, remove "Bot: " prefix

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