# main.py
import streamlit as st
from PIL import Image
import base64
#from pages import Project_Overview, Descriptive_Insights, Predictive_Insights

# Set page config
st.set_page_config(page_title="EduInsight", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="collapsed")

# Function to load and display background image
def set_background(png_file):
    with open(png_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
# Main page content with custom HTML and CSS for styling
if "page" not in st.session_state:
    st.session_state.page = "Main"
    

# Main Page
if st.session_state.page == "Main":
    set_background('Data/img.png')
    # Hide sidebar initially on the main page
    st.markdown(
        """
        <style>
        [data-testid="collapsedControl"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Title
    #st.markdown('<div class="main-title">Discover Educational Trends and Patterns</div>', unsafe_allow_html=True)
    
    # Enter button container and button
    # Center the "Enter" button using empty containers
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # with st.container():
        # st.container()
        # st.container()
        # st.container()
        # st.container()
        # st.container()
        # st.container()
        enter_button_clicked = st.button('Enter', key='enter', help="Click to access the dashboard")

        if enter_button_clicked:
            st.session_state.page = "Project Overview"
            st.experimental_rerun()

    st.markdown("""
    <style>
    [data-testid="stButton"]{
        display:flex;
        justify-content: center;
        align-items: center;
        margin-top: 60px;
        font-size: 50px;
        height: 50px;
        width: 300px;
        padding: 20px 40px;
    }
    </style>
        """,
        unsafe_allow_html=True
    )



# # Sidebar Navigation
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Project Overview", "Descriptive Insights", "Predictive Insights", "Chatbot"])

# # Page Content
# if page == "Project Overview":
#     from pages import overview as pov
#     pov.show()
# elif page == "Descriptive Insights":
#     from pages import descriptive_insights as di
#     di.show()
# elif page == "Predictive Insights":
#     from pages import predictive_insights as pi
#     pi.show()
# elif page == "Chatbot":
#     from pages import chatbot as cb
#     cb.show()