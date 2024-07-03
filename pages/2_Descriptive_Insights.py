# pages/Descriptive_Insights.py
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import extra_streamlit_components as stx
import plotly.graph_objects as go
from datetime import datetime
import statsmodels.api as sm
from chat_llm import generate_response 


# Set page config
st.set_page_config(page_title="Descriptive Insights", layout="wide")
st.markdown("# Descriptive Detail")
st.sidebar.header("Descriptive Overview")

page1 = st.sidebar.radio('Descriptive Insights sub-menu', options=['Insights'])


#--------------------------------------------Dataset loading--------------------------------------------------------------
if page1 == "Insights":
    # Check if a file has been uploaded by the user
    uploaded_file = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))

    if uploaded_file is not None:
        # If a file is uploaded, read the DataFrame from the uploaded file
        filename = uploaded_file.name
        st.write(filename)
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    else:
        # If no file is uploaded, fall back to using the default file
        df = pd.read_csv("Data/combined_all_data.csv", encoding="ISO-8859-1")
else:
    # Load the default file if the page is not "Insights"
    #os.chdir(r"C:\Users\PMYLS\Documents\FYP Material\FYP Material")
    df = pd.read_csv("Data/combined_all_data.csv", encoding="ISO-8859-1")

#---------------------------------------------Mentioning Key Performance Indicators-----------------------------------------

# Define function to calculate KPIs based on the selected filters
def calculate_kpis(df):
    # Calculate KPIs
    total_students = len(df)
    #pass_rate = (df['pass'].sum() / total_students) * 100
    male_ratio = (df['sex'] == 'M').mean() * 100
    female_ratio = 100 - male_ratio
    passed_students = df['pass'].sum()
    male_pass_rate = (df[df['sex'] == 'M']['pass'].sum() / (df['sex'] == 'M').sum()) * 100
    female_pass_rate = (df[df['sex'] == 'F']['pass'].sum() / (df['sex'] == 'F').sum()) * 100
    government_schools = df['is_government'].sum()
    government_male_pass_rate = (df[(df['is_government'] == 1) & (df['sex'] == 'M')]['pass'].sum() / df[(df['is_government'] == 1) & (df['sex'] == 'M')].shape[0]) * 100
    government_female_pass_rate = (df[(df['is_government'] == 1) & (df['sex'] == 'F')]['pass'].sum() / df[(df['is_government'] == 1) & (df['sex'] == 'F')].shape[0]) * 100
    nongovernment_schools = len(df) - government_schools
    nongovernment_male_pass_rate = (df[(df['is_government'] == 0) & (df['sex'] == 'M')]['pass'].sum() / df[(df['is_government'] == 0) & (df['sex'] == 'M')].shape[0]) * 100
    nongovernment_female_pass_rate = (df[(df['is_government'] == 0) & (df['sex'] == 'F')]['pass'].sum() / df[(df['is_government'] == 0) & (df['sex'] == 'F')].shape[0]) * 100
    urban_schools = df['is_urban'].sum()
    urban_male_pass_rate = (df[(df['is_urban'] == 1) & (df['sex'] == 'M')]['pass'].sum() / df[(df['is_urban'] == 1) & (df['sex'] == 'M')].shape[0]) * 100
    urban_female_pass_rate = (df[(df['is_urban'] == 1) & (df['sex'] == 'F')]['pass'].sum() / df[(df['is_urban'] == 1) & (df['sex'] == 'F')].shape[0]) * 100
    nonurban_schools = len(df) - urban_schools
    nonurban_male_pass_rate = (df[(df['is_urban'] == 0) & (df['sex'] == 'M')]['pass'].sum() / df[(df['is_urban'] == 0) & (df['sex'] == 'M')].shape[0]) * 100
    nonurban_female_pass_rate = (df[(df['is_urban'] == 0) & (df['sex'] == 'F')]['pass'].sum() / df[(df['is_urban'] == 0) & (df['sex'] == 'F')].shape[0]) * 100
   
    
    return total_students, male_ratio, female_ratio,  passed_students, male_pass_rate, female_pass_rate, government_schools, government_male_pass_rate, government_female_pass_rate, nongovernment_schools, nongovernment_male_pass_rate, nongovernment_female_pass_rate, urban_schools, urban_male_pass_rate, urban_female_pass_rate, nonurban_schools, nonurban_male_pass_rate, nonurban_female_pass_rate

# Define function to calculate additional metrics
def calculate_additional_metrics(df):
    df['Year'] = df['Year'].astype(str)
    # Calculate additional metrics
    year_student_count = df.groupby('Year').size()
    year_male_female_ratio = df.groupby('Year')['sex'].value_counts(normalize=True).unstack()
    year_male_female_pass_rate = df.groupby(['Year', 'sex'])['pass'].mean() * 100
    year_pass_rate_group = df.groupby(['Year', 'group'])['pass'].mean() * 100
    
    return year_student_count, year_male_female_ratio, year_male_female_pass_rate, year_pass_rate_group

#--------------------------------------------Dashobard Main Page section------------------------------------------------------

if page1 == "About":
    st.title('SSC School Exit Exam Results Dashboard')
    st.write('Welcome to the dashboard!')
    # Introduction and Project Overview
    st.markdown(
        """
        Welcome to the Education Dashboard! This dashboard provides a comprehensive analysis
        of school exit exam results from 2015 to 2023, offering valuable insights into student 
        performance, trends, and more.

        ## Project Overview

        The goal of this project is to:
        - Evaluate student performance over the years.
        - Explore demographic patterns.
        - Provide stakeholders with actionable insights to improve education.

        ## Key Features

        - **Year-wise Analysis:** Explore detailed performance for each year.
        - **Subject-wise Breakdown:** Analyze subject-wise performance gender-wise.
        - **Interactive Components:** Use the sidebar to customize your analysis.

        ## Insights for Stakeholders

        - **Educators:** Identify subject areas where students may need additional support.
        - **Policy Makers:** Understand trends to inform education policies.
        - **Parents:** Track the performance of schools and make informed decisions.
        - **Researchers:** Explore patterns and contribute to educational research.

        """
    )

#----------------------------------------------Sidebar section----------------------------------------------------------------   

# Insights Submenu
elif page1 == "Insights":
    st.title("Revealing Trends")
    insight_option = st.sidebar.selectbox("Select Insight", ["Multi-Year Insights", "Year-wise Insights"])

#----------------------------------------------Yearwise Insight Page-----------------------------------------------------------
   
    if insight_option == "Year-wise Insights":
        with st.sidebar.container():
          # Define province, district, tehsil options
          #province_options = df["PROVINCE"].unique()
          district_options = df["DISTRICT_NAME"].unique()
          tehsil_options = df["teh_name"].unique()
          
          # Initialize selected_tehsil outside of the if block
          selected_tehsil = None

          # Sidebar options to select Province, District, and Tehsil
          #selected_province = st.multiselect("Select Province", province_options)
          selected_district = st.multiselect("Select District", district_options, default=None)
          if selected_district:
              filtered_tehsil_options = df[df["DISTRICT_NAME"].isin(selected_district)]["teh_name"].unique()
          else:
             filtered_tehsil_options = tehsil_options
          selected_tehsil = st.selectbox("Select Tehsil", filtered_tehsil_options, index=0)

          # Dictionary to map values to labels
          area_labels = {1: "Urban", 0: "Rural"}

          # Set default values for the selected area multi-select box
          #default_selected_area = ["Urban", "Rural"]  # 1 for urban, 0 for rural
          selected_area = st.multiselect("Select Area (urban/rural)", list(area_labels.values())) #default=default_selected_area)

        st.title("Year-wise Insights")
        year_selected = st.selectbox("Select Year", ['All Years'] + df['Year'].unique().tolist())
    
        if year_selected == 'All Years':
           st.write("Please select a specific year to view its insights.")
        else:
           st.subheader(f"Insights for Year {year_selected}")

           # Filter data based on selected year and sidebar filters
           filtered_df = df[df['Year'] == int(year_selected)]
           st.write("---")

           if selected_district and len(selected_district) == 1:
              district_name = selected_district[0]
              filtered_df = filtered_df[filtered_df["DISTRICT_NAME"] == district_name]

              if selected_tehsil:
                  filtered_df = filtered_df[filtered_df["teh_name"] == selected_tehsil]
                  title_text = f"Gender Distribution and Pass Rate in {selected_tehsil}, {district_name} for Year {year_selected}"
              else:
                  title_text = f"Gender Distribution and Pass Rate in {district_name} for Year {year_selected}"

              if "Urban" in selected_area and "Rural" in selected_area:
                      # If both urban and rural are selected, do not apply area filter
                      pass
              elif "Urban" in selected_area:
                      filtered_df = filtered_df[filtered_df['is_urban'] == 1]  # Filter for urban areas
              elif "Rural" in selected_area:
                      filtered_df = filtered_df[filtered_df['is_urban'] == 0]
           else:
              title_text = f"Gender Distribution and Pass Rate for Year {year_selected}"

           # KPI container
           kpi_container = st.container()

           with kpi_container:
              # Calculate KPIs for the selected year and filtered data
              total_students, male_ratio, female_ratio, passed_students, male_pass_rate, female_pass_rate, government_schools, government_male_pass_rate, government_female_pass_rate, nongovernment_schools, nongovernment_male_pass_rate, nongovernment_female_pass_rate, urban_schools, urban_male_pass_rate, urban_female_pass_rate, nonurban_schools, nonurban_male_pass_rate, nonurban_female_pass_rate = calculate_kpis(filtered_df)
    
              # Display KPIs for the selected year and filtered data
              col1, col2, col3 = st.columns(3)
              with col1:
                st.markdown(f'<div style="height: 20px;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background-color: #0A0D2D; color: white; border-radius: 30px; padding: 10px; text-align: center;"><span style="font-size: 20px;"><b>Total Students</b><br>{total_students}</span></div>', unsafe_allow_html=True)
              with col2:
                st.markdown(f'<div style="height: 20px;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background-color: #0A0D2D; color: white; border-radius: 30px; padding: 10px; text-align: center;"><span style="font-size: 20px;"><b>Male Ratio (%)</b><br>{male_ratio:.2f}</span></div>', unsafe_allow_html=True)
              with col3:
                st.markdown(f'<div style="height: 20px;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background-color: #0A0D2D; color: white; border-radius: 30px; padding:10px; text-align: center;"><span style="font-size: 20px;"><b>Female Ratio (%)</b><br>{female_ratio:.2f}</span></div>', unsafe_allow_html=True)

              col4, col5, col6 = st.columns(3)
              with col4:
                st.markdown(f'<div style="height: 20px;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background-color: #0A0D2D; color: white; border-radius: 30px; padding: 10px; text-align: center;"><span style="font-size: 20px;"><b>Passed Students</b><br>{passed_students}</span></div>', unsafe_allow_html=True)
              with col5:
                st.markdown(f'<div style="height: 20px;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background-color: #0A0D2D; color: white; border-radius: 30px; padding: 10px; text-align: center;"><span style="font-size: 20px;"><b>Male Pass Rate (%)</b><br>{male_pass_rate:.2f}</span></div>', unsafe_allow_html=True)
              with col6:
                st.markdown(f'<div style="height: 20px;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background-color: #0A0D2D; color: white; border-radius: 30px; padding: 10px; text-align: center;"><span style="font-size: 20px;"><b>Female Pass Rate (%)</b><br>{female_pass_rate:.2f}</span></div>', unsafe_allow_html=True)

              col7, col8, col9 = st.columns(3)
              with col7:
                st.markdown(f'<div style="height: 20px;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background-color: #0A0D2D; color: white; border-radius: 30px; padding: 10px; text-align: center;"><span style="font-size: 20px;"><b>Government Schools Students</b><br>{government_schools}</span></div>', unsafe_allow_html=True)
              with col8:
                st.markdown(f'<div style="height: 20px;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background-color: #0A0D2D; color: white; border-radius: 30px; padding: 10px; text-align: center;"><span style="font-size: 20px;"><b>Male Pass Rate (%)</b><br>{government_male_pass_rate:.2f}</span></div>', unsafe_allow_html=True)
              with col9:
                st.markdown(f'<div style="height: 20px;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background-color: #0A0D2D; color: white; border-radius: 30px; padding: 10px; text-align: center;"><span style="font-size: 20px;"><b>Female Pass Rate (%)</b><br>{government_female_pass_rate:.2f}</span></div>', unsafe_allow_html=True)

              col10, col11, col12 = st.columns(3)
              with col10:
                st.markdown(f'<div style="height: 20px;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background-color: #0A0D2D; color: white; border-radius: 30px; padding: 10px; text-align: center;"><span style="font-size: 20px;"><b>Non-Government Schools Students</b><br>{nongovernment_schools}</span></div>', unsafe_allow_html=True)
              with col11:
                st.markdown(f'<div style="height: 20px;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background-color: #0A0D2D; color: white; border-radius: 30px; padding: 10px; text-align: center;"><span style="font-size: 20px;"><b>Male Pass Rate (%)</b><br>{nongovernment_male_pass_rate:.2f}</span></div>', unsafe_allow_html=True)
              with col12:
                st.markdown(f'<div style="height: 20px;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background-color: #0A0D2D; color: white; border-radius: 30px; padding: 10px; text-align: center;"><span style="font-size: 20px;"><b>Female Pass Rate (%)</b><br>{nongovernment_female_pass_rate:.2f}</span></div>', unsafe_allow_html=True)

#-----------------------------------------------------Visuals based on Variables--------------------------------------------------------------
           st.write("---")
           # Bar chart of gender distribution in the selected year
           col1, col2 = st.columns(2)
           with col1:
              st.subheader("Gender Distribution")
              fig_gender_distribution_year = px.bar(filtered_df, x='sex', color='sex', labels={'x':'Gender', 'y':'Count'}, 
                                           title=f"Gender Distribution in {title_text}", 
                                           #hover_data={'sex': True, 'y': True}, 
                                           category_orders={'sex': ['M', 'F']})
              st.plotly_chart(fig_gender_distribution_year, use_container_width=True)

              with st.expander("View Gender Distribution Data"):
                  #st.write(filtered_df.groupby('sex').size().reset_index(name='Count').style.background_gradient(cmap="Blues"))
                  csv_gender_year = filtered_df.groupby('sex').size().reset_index(name='Count').to_csv(index=False).encode('utf-8')
                  st.download_button("Download Gender Distribution Data", data=csv_gender_year, file_name="Gender_Distribution_Year.csv", mime="text/csv", 
                                help="Click here to download the data as a CSV file")
                
              # Calculate pass rate percentage for government and non-government schools
              government_pass_rate = (filtered_df[filtered_df['is_government'] == 1]['pass'].mean()) * 100
              non_government_pass_rate = (filtered_df[filtered_df['is_government'] == 0]['pass'].mean()) * 100

              # Pie chart of pass rate percentage for government and non-government schools
              st.subheader("Pass Rate Percentage by School Type")
              fig_pass_rate_school_type = px.pie(values=[government_pass_rate, non_government_pass_rate], names=['Government', 'Non-Government'], title=f"Pass Rate Percentage by School Type in {title_text}")
              st.plotly_chart(fig_pass_rate_school_type, use_container_width=True)

              with st.expander("View Pass Rate Percentage Data by School Type"):
                  pass_rate_school_type_data = pd.DataFrame({
                  'School Type': ['Government', 'Non-Government'],
                'Pass Rate Percentage': [government_pass_rate, non_government_pass_rate]
            })
                  #st.write(pass_rate_school_type_data.style.background_gradient(cmap="Blues"))
                  csv_pass_rate_school_type = pass_rate_school_type_data.to_csv(index=False).encode('utf-8')
                  st.download_button("Download Pass Rate Percentage Data by School Type", data=csv_pass_rate_school_type, file_name="Pass_Rate_Percentage_by_School_Type.csv", mime="text/csv", 
                            help="Click here to download the data as a CSV file")
                
              # Calculate gender distribution count for urban and non-urban areas
              urban_gender_count = filtered_df[filtered_df['is_urban'] == 1].groupby('sex').size().reset_index(name='Count')
              rural_gender_count = filtered_df[filtered_df['is_urban'] == 0].groupby('sex').size().reset_index(name='Count')

              # Bar chart of gender distribution count for government and non-government schools
              st.subheader("Gender Distribution by Area Type")
              fig_gender_distribution_area_type = px.bar(pd.concat([urban_gender_count, rural_gender_count]), x='sex', y='Count', color='sex', barmode='group', 
                                                      labels={'sex': 'Gender', 'Count': 'Count'}, title=f"Gender Distribution by Area Type in {title_text}")
              st.plotly_chart(fig_gender_distribution_area_type, use_container_width=True)

              with st.expander("View Gender Distribution Data by Area Type"):
                 st.write("Urban Areas:")
                 #st.write(urban_gender_count.style.background_gradient(cmap="Blues"))
                 st.write("Rural Areas:")
                 #st.write(rural_gender_count.style.background_gradient(cmap="Blues"))
                 csv_gender_distribution_area_type = pd.concat([urban_gender_count, rural_gender_count]).to_csv(index=False).encode('utf-8')
                 st.download_button("Download Gender Distribution Data by Area Type", data=csv_gender_distribution_area_type, file_name="Gender_Distribution_by_area_Type.csv", mime="text/csv", 
                            help="Click here to download the data as a CSV file")
                
           # Pie chart of pass rate percentage gender-wise in the selected year
           with col2:
              st.subheader("Pass Rate Percentage by Gender")
              fig_pass_rate_gender = px.pie(filtered_df, names='sex', values='pass', hole=0.5, title=f"Pass Rate Percentage by Gender in {title_text}")
              st.plotly_chart(fig_pass_rate_gender, use_container_width=True)

              with st.expander("View Pass Rate Percentage Data by Gender"):
                  pass_rate_gender_data = filtered_df.groupby('sex')['pass'].mean() * 100
                  #st.write(pass_rate_gender_data.reset_index(name='Pass Rate Percentage').style.background_gradient(cmap="Blues"))
                  csv_pass_rate_gender = pass_rate_gender_data.reset_index(name='Pass Rate Percentage').to_csv(index=False).encode('utf-8')
                  st.download_button("Download Pass Rate Percentage Data by Gender", data=csv_pass_rate_gender, file_name="Pass_Rate_Percentage_by_Gender.csv", mime="text/csv", 
                            help="Click here to download the data as a CSV file")
                
              # Calculate gender distribution count for government and non-government schools
              government_gender_count = filtered_df[filtered_df['is_government'] == 1].groupby('sex').size().reset_index(name='Count')
              non_government_gender_count = filtered_df[filtered_df['is_government'] == 0].groupby('sex').size().reset_index(name='Count')

              # Bar chart of gender distribution count for government and non-government schools
              st.subheader("Gender Distribution by School Type")
              fig_gender_distribution_school_type = px.bar(pd.concat([government_gender_count, non_government_gender_count]), x='sex', y='Count', color='sex', barmode='group', 
                                                      labels={'sex': 'Gender', 'Count': 'Count'}, title=f"Gender Distribution by School Type in {title_text}")
              st.plotly_chart(fig_gender_distribution_school_type, use_container_width=True)

              with st.expander("View Gender Distribution Data by School Type"):
                 st.write("Government Schools:")
                 #st.write(government_gender_count.style.background_gradient(cmap="Blues"))
                 st.write("Non-Government Schools:")
                 #st.write(non_government_gender_count.style.background_gradient(cmap="Blues"))
                 csv_gender_distribution_school_type = pd.concat([government_gender_count, non_government_gender_count]).to_csv(index=False).encode('utf-8')
                 st.download_button("Download Gender Distribution Data by School Type", data=csv_gender_distribution_school_type, file_name="Gender_Distribution_by_School_Type.csv", mime="text/csv", 
                            help="Click here to download the data as a CSV file")
               
              with col2:
                  st.subheader('Pass Rate Percentage by Area Type')
                  # Calculate pass rate percentage for government and non-government schools
                  urban_pass_rate = (filtered_df[filtered_df['is_urban'] == 1]['pass'].mean()) * 100
                  rural_pass_rate = (filtered_df[filtered_df['is_urban'] == 0]['pass'].mean()) * 100
              
                  #pull_values = [0.1, 0]
                  fig_pass_rate_area_type = px.pie(values=[urban_pass_rate, rural_pass_rate], 
                                      names=['Urban', 'Rural'], 
                                      title=f"Pass Rate Percentage by Area Type in {title_text}",
                                      hole=0.5,  # Adjust the hole size for the "3D" effect
                  )
                  st.plotly_chart(fig_pass_rate_area_type, use_container_width=True)

              with st.expander("View Pass Rate Percentage Data by Area Type"):
                  pass_rate_area_type_data = pd.DataFrame({
                'School Type': ['Government', 'Non-Government'],
                'Pass Rate Percentage': [urban_pass_rate, rural_pass_rate]
                })
                  #st.write(pass_rate_area_type_data.style.background_gradient(cmap="Blues"))
                  csv_pass_rate_area_type = pass_rate_area_type_data.to_csv(index=False).encode('utf-8')
                  st.download_button("Download Pass Rate Percentage Data by Area Type", data=csv_pass_rate_area_type, file_name="Pass_Rate_Percentage_by_area_Type.csv", mime="text/csv", 
                            help="Click here to download the data as a CSV file")
                  
              with col1:
                 st.subheader('Compulsory Subject Performance over the year')
                 # Create a dropdown select box for selecting the subject
                 selected_subject_1 = st.selectbox("Select Subject", ['Sub-UR', 'Sub-ENG', 'Sub-ISL/Ethics', 'Sub-PS', 'Sub-Maths'])

                 # Extract marks column corresponding to the selected subject
                 marks_column = selected_subject_1.replace('Sub-', '') + '-Marks'
                 # Define the total marks for each subject
                 total_marks = {'PS': 100, 'ISL/Ethics': 100, 'Maths': 150, 'ENG': 150, 'UR': 150}
                 subject_df = filtered_df[['Year', selected_subject_1, 'sex', marks_column]]

                 subject_df[marks_column] = pd.to_numeric(subject_df[marks_column], errors='coerce')
                 subject_df = subject_df.dropna(subset=[marks_column])

                 # Group the DataFrame by Year and gender, calculate the mean marks obtained
                 subject_performance = subject_df.groupby(['Year', 'sex']).agg({marks_column: 'mean'}).reset_index()

                 # Calculate the overall mean marks for the selected subject
                 overall_mean_marks = subject_df[marks_column].mean()
                 # Determine the total marks for the selected subject
                 subject_total_marks = total_marks[selected_subject_1.split('-')[1]]

                 # Create a Gauge chart
                 fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_mean_marks,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f'Average Marks in {selected_subject_1}', 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, subject_total_marks], 'tickformat': '.0f'},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, subject_total_marks / 2], 'color': "lightgray"},
                {'range': [subject_total_marks / 2, subject_total_marks], 'color': "lightblue"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': overall_mean_marks}}))

                # Show the gauge chart
              st.plotly_chart(fig, use_container_width=True)

               # Calculate metrics for each compulsory subject
           def calculate_subject_metrics(filtered_df, compulsory_subjects):
                     """
                     Calculates various metrics for each compulsory subject.

                     Args:
                        filtered_df (pd.DataFrame): The filtered DataFrame based on selected year.
                        compulsory_subjects (list): List of compulsory subject names.

                     Returns:
                        pd.DataFrame: A DataFrame containing subject-wise metrics.
                     """

                     metrics = []
                     for subject in compulsory_subjects:
                        marks_column = subject.replace('Sub-', '') + '-Marks'

                        # Get student counts with missing value handling
                        student_count = len(filtered_df[filtered_df[marks_column].notna()])
                        male_count = filtered_df[(filtered_df[marks_column].notna()) & (filtered_df['sex'] == 'M')]['sex'].count()
                        female_count = filtered_df[(filtered_df[marks_column].notna()) & (filtered_df['sex'] == 'F')]['sex'].count()

                        # Get government/non-government student counts
                        government_count = filtered_df[(filtered_df[marks_column].notna()) & (filtered_df['is_government'] == 1)][marks_column].count()
                        non_government_count = filtered_df[(filtered_df[marks_column].notna()) & (filtered_df['is_government'] == 0)][marks_column].count()

                     # Calculate statistics with missing value handling
                        filtered_df[marks_column] = pd.to_numeric(filtered_df[marks_column], errors='coerce').dropna()
                        average_marks = filtered_df[marks_column].mean()
                        lowest_marks = filtered_df[marks_column].min()

                        # Calculate male/female ratios and average marks for each school type
                        government_male_count = filtered_df[(filtered_df[marks_column].notna()) & (filtered_df['is_government'] == 1) & (filtered_df['sex'] == 'M')][marks_column].count()
                        government_female_count = filtered_df[(filtered_df[marks_column].notna()) & (filtered_df['is_government'] == 1) & (filtered_df['sex'] == 'F')][marks_column].count()
                        #government_male_ratio = government_male_count / government_female_count if government_female_count else government_male_count
                        government_average_marks = filtered_df[(filtered_df[marks_column].notna()) & (filtered_df['is_government'] == 1)][marks_column].mean()

                        non_government_male_count = filtered_df[(filtered_df[marks_column].notna()) & (filtered_df['is_government'] == 0) & (filtered_df['sex'] == 'M')][marks_column].count()
                        non_government_female_count = filtered_df[(filtered_df[marks_column].notna()) & (filtered_df['is_government'] == 0) & (filtered_df['sex'] == 'F')][marks_column].count()
                        non_government_average_marks = filtered_df[(filtered_df[marks_column].notna()) & (filtered_df['is_government'] == 0)][marks_column].mean()
                        
                        average_marks = round(average_marks, 0)
                        metrics.append({
                        'Subject': subject,
                        'Total Students': student_count,
                        'Average Marks': average_marks,
                        'Male Count': male_count,
                        'Female Count': female_count,
                        'Lowest Marks Obtained': lowest_marks,
                        'Government Student Count': government_count,
                        'Govt. Male': government_male_count,
                        'Govt. Female': government_female_count,
                        'Non-Government Student Count': non_government_count,
                        'Non-Govt. Male': non_government_male_count,
                        'Non-Govt. Female': non_government_female_count,
                        'Government Average Marks': government_average_marks,
                        'Non-Government Average Marks': non_government_average_marks,

                     })

                     return pd.DataFrame(metrics)

           compulsory_subjects = ['Sub-UR', 'Sub-ENG', 'Sub-ISL/Ethics', 'Sub-PS', 'Sub-Maths']
           metrics_df = calculate_subject_metrics(filtered_df, compulsory_subjects)

               # Display the DataFrame as a table using Streamlit
           st.subheader('Compulsory Subjects Insights')
           st.markdown("<div style='text-align: center'>" + metrics_df.to_html(index=False) + "</div>", unsafe_allow_html=True)
           
           
           #st.subheader('Performance of Regular and private students')
           def calculate_performance(filtered_df):
              """
              Calculates the performance of private and regular students.

              Args:
              filtered_df (pd.DataFrame): The filtered DataFrame.

              Returns:
              pd.DataFrame: A DataFrame containing the performance metrics.
              """
              # Group by student type (private or regular), group, and gender
              performance_df = filtered_df.groupby(['status', 'group', 'sex']).agg(
                Total_Students=('status', 'count'),
                Passed_Students=('pass', 'sum')
              ).reset_index()

              performance_df['Pass_Percentage'] = (performance_df['Passed_Students'] / performance_df['Total_Students']) * 100

              return performance_df
           performance_metrics = calculate_performance(filtered_df)

           # Display performance metrics in a tabular form
           st.subheader('Performance Metrics by Student Type, Group, and Gender')
           st.dataframe(performance_metrics, use_container_width=True)



#------------------------------------------------Entire Year Insight Page-----------------------------------------------------------------------------------

    elif insight_option == "Multi-Year Insights":

        st.title("Comprehensive Annual Review (2016-2023)")
        st.subheader('Top Performing Year-2021')

        # Group the data by year
        grouped_data = df.groupby('Year')

       # Initialize an empty list to store summary data
        summary_data = []

       # Iterate over each group (year) and calculate summary statistics
        for year, group in grouped_data:
          total_all_students = len(group)
          total_all_male_students = len(group[group['sex'] == 'M'])
          total_all_female_students = len(group[group['sex'] == 'F'])
          total_all_passed_students = len(group[group['pass'] == 1])
          total_all_urban_students = len(group[group['is_urban'] == 1])
          total_all_rural_students = len(group[group['is_urban'] == 0])
          total_all_government_students = len(group[group['is_government'] == 1])
          total_all_non_government_students = len(group[group['is_government'] == 0])

          male_percentage = (total_all_male_students / total_all_students) * 100
          female_percentage = (total_all_female_students / total_all_students) * 100
          passed_percentage = (total_all_passed_students / total_all_students) * 100
          urban_percentage = (total_all_urban_students / total_all_students) * 100
          rural_percentage = (total_all_rural_students / total_all_students) * 100
          government_percentage = (total_all_government_students / total_all_students) * 100
          non_government_percentage = (total_all_non_government_students / total_all_students) * 100

          # Append the summary statistics to the list
          summary_data.append({
              'Year': year,
              'Total Students': total_all_students,
              'Male Students': total_all_male_students,
              'Female Students': total_all_female_students,
              'Passed Students': total_all_passed_students,
              'Urban Students': total_all_urban_students,
              'Rural Students': total_all_rural_students,
              'Government Students': total_all_government_students,
              'Non-Government Students': total_all_non_government_students,
              'Male Percentage': male_percentage,
              'Female Percentage': female_percentage,
              'Passed Percentage': passed_percentage,
              'Urban Percentage': urban_percentage,
              'Rural Percentage': rural_percentage,
              'Government Percentage': government_percentage,
              'Non-Government Percentage': non_government_percentage,
          })

        # Create a DataFrame from the summary data
        summary_df = pd.DataFrame(summary_data)

        # Sort the DataFrame by 'Passed Students'
        summary_df = summary_df.sort_values(by='Passed Students', ascending=False)

        # Interactive widgets for user selections
        year_options = summary_df['Year'].astype(str).unique().tolist()
        selected_year = st.selectbox("Select Year", options=year_options, index=0)
        metric_options = ["Total Students", "Male Students", "Female Students", "Passed Students", 
                          "Urban Students", "Rural Students", "Government Students", "Non-Government Students"]
        selected_metric = st.radio("Select Metric", options=metric_options, index=0)

        # Filter data based on user selection
        filtered_data = summary_df[summary_df['Year'] == int(selected_year)]
        metric_value = filtered_data[selected_metric].values[0]

        # Display selected metric
        st.metric(f"{selected_metric} in {selected_year}", f"{metric_value}")

        # Prepare data for hover text
        summary_df['hover_text'] = summary_df.apply(lambda row: (
            f"Year: {row['Year']}<br>"
            f"Total Students: {row['Total Students']}<br>"
            f"Male Percentage: {row['Male Percentage']:.2f}%<br>"
            f"Female Percentage: {row['Female Percentage']:.2f}%<br>"
            f"Passed Percentage: {row['Passed Percentage']:.2f}%<br>"
            f"Urban Percentage: {row['Urban Percentage']:.2f}%<br>"
            f"Rural Percentage: {row['Rural Percentage']:.2f}%<br>"
            f"Government Percentage: {row['Government Percentage']:.2f}%<br>"
            f"Non-Government Percentage: {row['Non-Government Percentage']:.2f}%<br>"
        ), axis=1)

        # Dynamic chart based on user selection
        fig = px.bar(
            summary_df,
            x='Year',
            y=selected_metric,
            text='hover_text',
            title=f"{selected_metric} Over the Years"
        )

        # Update the hover template to reference correctly
        fig.update_traces(
            hovertemplate="%{text}<extra></extra>"
        )

        st.plotly_chart(fig, use_container_width=True)

        # # Display dynamic data table with tooltips
        # st.write("Hover over the rows for more details.")
        
        # # Create tooltips DataFrame
        # tooltips_df = pd.DataFrame({
        #     'Male Percentage': summary_df['Male Percentage'].apply(lambda x: f"{x:.2f}%"),
        #     'Female Percentage': summary_df['Female Percentage'].apply(lambda x: f"{x:.2f}%"),
        #     'Passed Percentage': summary_df['Passed Percentage'].apply(lambda x: f"{x:.2f}%"),
        #     'Urban Percentage': summary_df['Urban Percentage'].apply(lambda x: f"{x:.2f}%"),
        #     'Rural Percentage': summary_df['Rural Percentage'].apply(lambda x: f"{x:.2f}%"),
        #     'Government Percentage': summary_df['Government Percentage'].apply(lambda x: f"{x:.2f}%"),
        #     'Non-Government Percentage': summary_df['Non-Government Percentage'].apply(lambda x: f"{x:.2f}%")
        # })

        # # Align tooltips with summary_df and apply
        # styled_df = summary_df.style.set_tooltips(tooltips_df)

        # # Adding hover style for rows
        # styled_df = styled_df.set_table_styles([
        #     {'selector': 'tr:hover', 'props': [('background-color', '#ffff99')]}
        # ])

        # st.write(styled_df.to_html(), unsafe_allow_html=True)

        # Expandable sections for additional details
        with st.expander("View Detailed Statistics by Year"):
            for year in year_options:
                st.write(f"### {year}")
                year_data = summary_df[summary_df['Year'] == int(year)]
                st.write(year_data)

        # Add custom CSS for hover effects and more
        st.markdown("""
            <style>
            .stTable tr:hover {
                background-color: #ffff99;
            }
            </style>
        """, unsafe_allow_html=True)

        st.write("---")

       # Rest of code for dynamic graphs and insights for the entire year dataset goes here
       
       # Define tabs
       #tab1, tab2 = st.tabs(["🗃Gender Distribution", "📈 Performance"])
       # Create a dropdown menu for selecting tabs
        selected_tab = st.selectbox("Select Tab", ["🗃 Enrollment Trend", "📈 Performance Trend"])

        st.write("---")
       
       # Concatenate year-wise data
        year_student_count, year_male_female_ratio, year_male_female_pass_rate, year_pass_rate_group = calculate_additional_metrics(df)

        if selected_tab == "🗃 Enrollment Trend":
          
           # Tab 1: Gender Distribution
           #with tab1:
              st.subheader("Gender Distribution")

              # Column layout
              col1, col2 = st.columns(2)
    
              # Insight 1: Year-wise Student Count
              with col1:
                st.subheader("Year-wise Student Count")
                year_total_students = df.groupby('Year').size()
                fig_total_students = px.bar(x=year_total_students.index, y=year_total_students.values, labels={'x':'Year', 'y':'Total Student Count'})
                st.plotly_chart(fig_total_students, use_container_width=True)


                 # Add expander for Year-wise Student Count data
                with st.expander("Total Student Count Across All Years"):
                     st.write(year_total_students.astype(str))
                     csv_total_students = year_total_students.to_csv(index=True).encode('utf-8')
                     st.download_button("Download Year-wise Total Student Count Data", data=csv_total_students, file_name="Year-wise_Total_Student_Count.csv", mime="text/csv", help="Click here to download the data as a CSV file")

              # Insight 2: Gender wise distribution in each year
             
              with col2:
               st.subheader('Gender Distribution Across All Years')

               # Group the DataFrame by Year and sex and count the number of students
               gender_distribution = df.groupby(['Year', 'sex']).size().reset_index(name='count')

               # Pivot the DataFrame to get male and female counts as separate columns
               gender_distribution_pivot = gender_distribution.pivot(index='Year', columns='sex', values='count').reset_index()

               # Rename the columns for better interpretation
               gender_distribution_pivot.rename(columns={'M': 'Male', 'F': 'Female'}, inplace=True)

               # Create the stacked bar chart
               fig = go.Figure()

               # Add male bars
               fig.add_trace(go.Bar(name='Male', x=gender_distribution_pivot['Year'], y=gender_distribution_pivot['Male']))

               # Add female bars
               fig.add_trace(go.Bar(name='Female', x=gender_distribution_pivot['Year'], y=gender_distribution_pivot['Female']))
               fig.update_yaxes(tickformat=".0f")
               # Customize layout
               fig.update_layout(barmode='group',
                  xaxis_title='Year',
                  yaxis_title='Number of Students'
                 )

               # Show the stacked bar chart
               st.plotly_chart(fig, use_container_width=True)
               
               # Add expander for Year-wise Student Count data
               with st.expander("Gender Distribution Across All Years"):
                    st.write(gender_distribution_pivot.astype(str))
                    csv1 = gender_distribution_pivot.to_csv(index=True).encode('utf-8')
                    st.download_button("Download Gender Distribution Count Data", data=csv1, file_name="Year-wise_Gender_Distribution_Count.csv", mime="text/csv", help="Click here to download the data as a CSV file")
     

              with col1:
              # Insight 3: Year-wise Student Count by District, Tehsil
                st.subheader("Year-wise Student Count by District, Tehsil")
                year_district_tehsil_sex_counts = df.groupby(['Year', 'DISTRICT_NAME', 'teh_name', 'sex']).size().reset_index(name='student_count')
                fig_treemap = px.treemap(year_district_tehsil_sex_counts, 
                              path=['Year', 'DISTRICT_NAME', 'teh_name', 'sex'],  
                              values='student_count',  
                              title='Student Count by Year, District, Tehsil',  
                              color='student_count',  
                              color_continuous_scale='Blues',  
                              hover_data={'Year': False, 'DISTRICT_NAME': False, 'teh_name': False, 'sex': True, 'student_count': True})  
                fig_treemap.update_layout(width=800, height=650)
                st.plotly_chart(fig_treemap, use_container_width=True)

                # Add expander for Year-wise Student Count District\Tehsil wise
                with st.expander("Year-wise Student Count District\Tehsil wise"):
                  st.write(year_district_tehsil_sex_counts.astype(str))
                  csv1 = year_district_tehsil_sex_counts.to_csv(index=True).encode('utf-8')
                  st.download_button("Download Year-wise Student Count District\Tehsil wise", data=csv1, file_name="Year-wise_Student_Count_District\Tehsil_wise.csv", mime="text/csv", help="Click here to download the data as a CSV file")
       
              with col2:
                # Insight 4: Year-wise Male-Female Ratio
                st.subheader("Year-wise Male-Female Ratio")
                fig2 = px.line(year_male_female_ratio, x=year_male_female_ratio.index, y=['M', 'F'], labels={'x':'Year', 'y':'Ratio'}, title='Male-Female Ratio Over Years')
                st.plotly_chart(fig2, use_container_width=True)

                # Add expander for Year-wise Male-Female Ratio data
                with st.expander("View Year-wise Male-Female Ratio Data"):
                   st.write(year_male_female_ratio)
                   csv2 = year_male_female_ratio.to_csv().encode('utf-8')
                   st.download_button("Download Year-wise Male-Female Ratio Data", data=csv2, file_name="Year-wise_Male_Female_Ratio.csv", mime="text/csv", help="Click here to download the data as a CSV file")
     

              with col1:
                 st.subheader("Enrollment trend District wise over the years")
                 # Group the DataFrame by year and district to calculate the total enrollment in each district for each year
                 district_enrollment_trend = df.groupby(['Year', 'DISTRICT_NAME']).size().reset_index(name='Enrollment')

                 # Find the top district for each year based on enrollment
                 districts = df['DISTRICT_NAME'].unique()

                 # Initialize the figure
                 fig = go.Figure()

                # Plot enrollment trend for each top district
                 for district in districts:
                      district_data = district_enrollment_trend[district_enrollment_trend['DISTRICT_NAME'] == district]
                      fig.add_trace(go.Scatter(x=district_data['Year'], y=district_data['Enrollment'], mode='lines+markers', name=f"{district}"))
                 fig.update_yaxes(tickformat=".0f")
                 fig.update_layout(title='Enrollment Trend of Students Yearly in Districts',
                  xaxis_title='Year',
                  yaxis_title='Enrollment Count')
                 st.plotly_chart(fig, use_container_width=True)

#----------------------------------- Gender Enrollment trend------------------------------------------------------------------------- 
              gender_enrollment_trend = df.groupby(['Year', 'DISTRICT_NAME', 'sex']).size().reset_index(name='Enrollment')

              # Extract the unique district names
              districts = df['DISTRICT_NAME'].unique()

              fig = go.Figure()

              # Plot enrollment trend for each district and gender
              for district in districts:
                  district_data = gender_enrollment_trend[gender_enrollment_trend['DISTRICT_NAME'] == district]
                  male_data = district_data[district_data['sex'] == 'M']
                  female_data = district_data[district_data['sex'] == 'F']
                  fig.add_trace(go.Bar(x=male_data['Year'], y=male_data['Enrollment'], name=f"{district} Male", marker_color='blue'))
                  fig.add_trace(go.Bar(x=female_data['Year'], y=female_data['Enrollment'], name=f"{district} Female", marker_color='red'))

              # Customize layout
              fig.update_layout(barmode='group',  # Group bars
                  title='Gender-wise Enrollment Trend of Students Yearly in Districts',
                  xaxis_title='Year',
                  yaxis_title='Enrollment Count',
                  legend_title='District')
              st.plotly_chart(fig, use_container_width=True)

              
              today_date = datetime.today()
              df['dob'] = pd.to_datetime(df['dob'])
              df['age'] = today_date.year - df['dob'].dt.year

              # Step 2: Calculate average age of students in each academic year
              average_age_yearly = df.groupby('Year')['age'].mean().reset_index()
              gender_avg_age = df.groupby(['Year', 'sex'])['age'].mean().reset_index()

              # Step 3: Plot the trend in gender distribution along with average age for each academic year
              fig = px.scatter(gender_avg_age, x='Year', y='age', color='sex', title='Average Age of Students by Year and Gender',
                     labels={'age': 'Average Age', 'Year': 'Academic Year', 'sex': 'Gender'},
                     hover_data={'age': True, 'Year': True, 'sex': True},
                     trendline='ols',  # Add Ordinary Least Squares trendline
                     )

              fig.update_yaxes(tickformat=".0f")
              fig.update_layout(
                 hovermode='closest',  # Show hover information for the closest point
                 xaxis=dict(
                 showspikes=True,  # Show spikes on hover for x-axis
                 spikethickness=1,
                 spikedash='solid',
               ),
              yaxis=dict(
                showspikes=True,  # Show spikes on hover for y-axis
                spikethickness=1,
                spikedash='solid',
               ),
               width=30,  # Adjust the width of the graph
               height=300
            )

              st.plotly_chart(fig, use_container_width=True)

              # Districts with coordinates
              districts = {
                'KHANEWAL': [30.3016, 71.9328],
                'MULTAN': [30.1798, 71.5249],
                'LODHRAN': [29.5407, 71.6333],
                'VEHARI': [30.0459, 72.3488],
            }

              with col1:
                st.subheader("Top Enrollment Trend by Tehsil")

                # Group data by year, district, and tehsil, and calculate total count of students
                tehsil_district_student_count = df.groupby(['Year', 'DISTRICT_NAME', 'teh_name', 'sex']).size().reset_index(name='Student Count')

                # Find the tehsil with the highest enrollment trend for each district and year
                top_enrollment_tehsils = tehsil_district_student_count.groupby(['Year', 'DISTRICT_NAME', 'sex']).apply(lambda x: x.nlargest(5, 'Student Count')).reset_index(drop=True)

                # Plot geographical chart
                fig = px.scatter_mapbox(top_enrollment_tehsils, 
                        lat=[districts[district.strip()][0] for district in top_enrollment_tehsils['DISTRICT_NAME']],
                        lon=[districts[district.strip()][1] for district in top_enrollment_tehsils['DISTRICT_NAME']],
                        hover_name="teh_name",
                        hover_data={"DISTRICT_NAME": True, "teh_name": True, "Year": True, "sex": True, "Student Count": True},
                        color="sex",
                        size="Student Count",
                        animation_frame="Year",
                        title="Enrollment Trend by Tehsil and District Over Years",
                        template="plotly",
                        mapbox_style="open-street-map"
                       )

                fig.update_layout(mapbox=dict(center=dict(lat=30.9654, lon=72.4183), zoom=7))
                st.plotly_chart(fig, use_container_width=True)
 
              with col2:
                  st.subheader("Enrollment trend in Government or Non Government Schools")
                  enrollment_trend = df.groupby(['Year', 'is_government']).size().reset_index(name='Enrollment Count')
                  enrollment_trend['Institution Type'] = enrollment_trend['is_government'].map({1: 'Government', 0: 'Non-Government'})
                  
                  fig.update_yaxes(tickformat=".0f")
                  # Plot bar chart
                  fig = px.bar(enrollment_trend, x='Year', y='Enrollment Count', color='Institution Type',
                      labels={'Enrollment Count': 'Enrollment Count', 'Year': 'Year', 'Institution Type': 'Institution Type'},
                      title='Enrollment Trend in Government and Non-Government Schools Over Years',
                      barmode='group')

                 # Show the bar chart
                  st.plotly_chart(fig, use_container_width=True)
                

              with col2:
                  st.subheader("Student Distribution in each tehsil")
                  # Group data by year and tehsil and calculate total count of students
                  tehsil_student_count = df.groupby(['Year', 'teh_name']).size().reset_index(name='Student Count')

                  # Calculate total count of students for each year
                  total_students_yearly = df.groupby('Year').size().reset_index(name='Total Students')

                  # Merge total count of students with tehsil_student_count
                  tehsil_student_count = pd.merge(tehsil_student_count, total_students_yearly, on='Year')

                  # Plot horizontal bar chart
                  fig = go.Figure()

                  for tehsil in tehsil_student_count['teh_name'].unique():
                     tehsil_data = tehsil_student_count[tehsil_student_count['teh_name'] == tehsil]
                     fig.add_trace(go.Bar(
                     y=tehsil_data['Year'],
                     x=tehsil_data['Student Count'],
                     orientation='h',
                     name=tehsil,
                     text=tehsil_data['Student Count'], 
                     textposition='auto',
                     #hoverinfo='x+text',
                 ))

                  fig.update_layout(
                     barmode='stack',
                     title='Student Distribution in Each Tehsil',
                     xaxis_title='Student Count',
                     yaxis_title='Year',
                     legend_title='Tehsil',
               )
                  st.plotly_chart(fig, use_container_width=True)

              with col2:
                  st.subheader('Student Enrollment in science subjects')
                  science_subjects = {'PHY': 'Physics', 'GSC': 'General Science', 'BIO': 'Biology', 'CH': 'Chemistry'}

                  all_science_subjects = df[['sub6', 'sub7', 'sub8']].values.flatten()
                  all_science_subjects = [subject for subject in all_science_subjects if str(subject) != 'nan' and subject in science_subjects.keys()]
                  science_df = df[df['sub6'].isin(all_science_subjects) | df['sub7'].isin(all_science_subjects) | df['sub8'].isin(all_science_subjects)]
                  subject_gender_count = science_df.groupby(['Year', 'sub6', 'sub7', 'sub8', 'sex']).size().reset_index(name='count')

                  # Merge all subject columns into one and map them to their corresponding names
                  subject_gender_count['Subject'] = subject_gender_count[['sub6', 'sub7', 'sub8']].apply(lambda row: science_subjects.get(next((col for col in row if col in science_subjects.keys()), None)), axis=1)

                  # Create a list of available science subjects
                  subject_list = list(science_subjects.values())

                  # Add a select box to choose the subject
                  selected_subject = st.selectbox("Select Subject", subject_list)

                  # Filter the DataFrame based on the selected subject
                  selected_subject_df = subject_gender_count[subject_gender_count['Subject'] == selected_subject]

                  # Plot line chart showing male and female enrollment counts year-wise for the selected subject
                  # Plot bar chart showing male and female enrollment counts year-wise for the selected subject
                  fig_bar = px.bar(selected_subject_df, x='Year', y='count', color='sex',
                     labels={'Year': 'Year', 'count': 'Count', 'sex': 'Gender'},
                     title=f'Male/Female Count Year-wise in {selected_subject}',
                     barmode='group')

                  # Show the line chart
                  st.plotly_chart(fig_bar, use_container_width=True)


        elif selected_tab == "📈 Performance Trend":
       
           #with tab2:
              st.subheader("Performance Trend over Time")
              st.write("---")
        
              # Concatenate year-wise data
              year_student_count, year_male_female_ratio, year_male_female_pass_rate, year_pass_rate_group = calculate_additional_metrics(df)
              # Column layout
              col1, col2 = st.columns(2)

              with col1:
                 st.title("Year-wise Pass Rate by Group")
                 fig4 = px.line(year_pass_rate_group.reset_index(), x='Year', y='pass', color='group', labels={'x':'Year', 'y':'Pass Rate'}, title='Pass Rate by Group Over Years')
                 st.plotly_chart(fig4, use_container_width=True)

                 # Add expander for Year-wise Pass Rate by Group data
                 with st.expander("View Year-wise Pass Rate by Group Data"):
                    st.write(year_pass_rate_group.reset_index())
                    csv4 = year_pass_rate_group.reset_index().to_csv(index=False).encode('utf-8')
                    st.download_button("Download Year-wise Pass Rate by Group Data", data=csv4, file_name="Year-wise_Pass_Rate_by_Group.csv", mime="text/csv", help="Click here to download the data as a CSV file") 
       
              with col2:
                 st.title("Year-wise Male-Female Pass Rate")
                 fig3 = px.line(year_male_female_pass_rate.reset_index(), x='Year', y='pass', color='sex', labels={'x':'Year', 'y':'Pass Rate'}, title='Male-Female Pass Rate Over Years')
                 st.plotly_chart(fig3, use_container_width=True)

                 # Add expander for Year-wise Male-Female Pass Rate data
                 with st.expander("View Year-wise Male-Female Pass Rate Data"):
                    st.write(year_male_female_pass_rate.reset_index())
                    csv3 = year_male_female_pass_rate.reset_index().to_csv(index=False).encode('utf-8')
                    st.download_button("Download Year-wise Male-Female Pass Rate Data", data=csv3, file_name="Year-wise_Male_Female_Pass_Rate.csv", mime="text/csv", help="Click here to download the data as a CSV file")

              
              st.write('---')
              with col2:
                  st.title('Top performing tehsil')
                  # Calculate pass rates year-wise for each tehsil
                  year_tehsil_pass_rate = df.groupby(['Year', 'teh_name']).agg({'pass': 'mean'}).reset_index()
                  year_tehsil_pass_rate['Pass Rate'] = year_tehsil_pass_rate['pass'] * 100

                  # Sort the DataFrame by pass rate in descending order
                  year_tehsil_pass_rate = year_tehsil_pass_rate.sort_values(by=['Year', 'Pass Rate'], ascending=[True, False])

                  # Get the top-performing tehsils year-wise
                  top_performing_tehsils = year_tehsil_pass_rate.groupby('Year').head(1)

                  # Plot top-performing tehsils year-wise based on pass rate
                  fig = px.bar(top_performing_tehsils, x='Year', y='Pass Rate', color='teh_name',
                    title='Top Performing Tehsils Year-wise Based on Pass Rate Percentage',
                    labels={'Pass Rate': 'Pass Rate (%)', 'teh_name': 'Top Performing Tehsil'})
                  st.plotly_chart(fig, use_container_width=True)

                  # Add expander for Top Performing Tehsil data
                  with st.expander("View Top Performing Tehsil Year-wise Data"):
                    st.write(year_tehsil_pass_rate.astype(str))
                    csv2 = year_tehsil_pass_rate.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Top Performing Tehsil Data", data=csv2, file_name="Top_Performing_Tehsil.csv", mime="text/csv", help="Click here to download the data as a CSV file")
                    
              st.write('----')
              
              with col1:
                 st.title("Compulsory Subjects Performance Over the Years")

                 # Create a dropdown select box for selecting the subject
                 selected_subject_1 = st.selectbox("Select Subject", ['Sub-UR', 'Sub-ENG', 'Sub-ISL/Ethics', 'Sub-PS', 'Sub-Maths'])

                 # Extract marks column corresponding to the selected subject
                 marks_column = selected_subject_1.replace('Sub-', '') + '-Marks'

                 # Filter the DataFrame based on the selected subject
                 subject_df = df[['Year', selected_subject_1, 'gender', marks_column]]

                 # Convert marks column to numeric
                 subject_df[marks_column] = pd.to_numeric(subject_df[marks_column], errors='coerce')

                 # Drop rows with NaN values in marks column
                 subject_df = subject_df.dropna(subset=[marks_column])

                 # Group the DataFrame by Year and gender, calculate the mean marks obtained
                 subject_performance = subject_df.groupby(['Year', 'gender']).agg({marks_column: 'mean'}).reset_index()
                 
                 
                 # Plot a line chart showing the year-wise trend of male and female performance in the selected subject
                 fig = px.line(subject_performance, x='Year', y=marks_column, color='gender', 
                   labels={'Year': 'Year', marks_column: 'Mean Marks'},
                   title=f'Year-wise Performance Trend in {selected_subject_1}',
                   line_shape='linear')
                 
                 fig.update_yaxes(tickformat=".0f")
                 # Show the line chart
                 st.plotly_chart(fig, use_container_width=True)
 
              with col1:
                 st.title("Subject Performance Yearly in Urban and Rural Areas")

                 # Add a select box to choose between urban and rural areas
                 selected_area = st.selectbox("Select Area", ["Urban", "Rural"])

                 # Filter the DataFrame based on the selected area
                 filtered_df = df[df['is_urban'] == (1 if selected_area == "Urban" else 0)]

                 # Define the subjects and their corresponding columns
                 subjects = ['UR-Marks', 'ENG-Marks', 'ISL/Ethics-Marks', 'PS-Marks', 'Maths-Marks']
                 subject_names = ['UR', 'ENG', 'ISL/Ethics', 'PS', 'Maths']

                 # Initialize a figure
                 # Initialize a list to store the traces
                 traces = []
                 # Iterate over each subject
                 for subject, subject_name in zip(subjects, subject_names):
                    # Group the DataFrame by Year and calculate the mean marks obtained for the subject
                    subject_performance = filtered_df.groupby('Year')[subject].mean().reset_index()

                    # Add a trace for the subject
                    traces.append(go.Bar(
                      x=subject_performance['Year'],
                      y=subject_performance[subject],
                      name=subject_name
                     ))

                  # Create a stacked bar chart
                 fig = go.Figure(data=traces)
                    # Customize layout
                 fig.update_layout(
                      barmode='group',
                      xaxis_title='Year',
                      yaxis_title='Mean Marks',
                      title=f'Subject Performance Yearly in {selected_area} Areas',
                      legend_title='Subject'
                     )
                 
                 fig.update_yaxes(tickformat=".0f")
                 # Show the grouped bar chart
                 st.plotly_chart(fig, use_container_width=True)

              with col2:
                 st.title('Total Obtained Marks by Year')

                 # Calculate total obtained marks year-wise
                 highest_obtained_marks_yearly = df.loc[df.groupby('Year')['Total_Obtained_Marks'].idxmax()]

                 # Plot total obtained marks year-wise
                 fig = px.bar(highest_obtained_marks_yearly, x='Year', y='Total_Obtained_Marks',
                    title='Highest Obtained Marks by Year',
                    labels={'Total_Obtained_Marks': 'Highest Obtained Marks', 'Year': 'Year'})
                 st.plotly_chart(fig, use_container_width=True)

                 # Add expander for Total Obtained Marks by Year data
                 with st.expander("View Highest Obtained Marks by Year Data"):
                      st.write(highest_obtained_marks_yearly.astype(str))
                      csv2 = highest_obtained_marks_yearly.to_csv(index=False).encode('utf-8')
                      st.download_button("Download Highest Obtained Marks by Year Data", data=csv2, file_name="Highest_Obtained_Marks_by_Year.csv", mime="text/csv", help="Click here to download the data as a CSV file")


                 # Define grade ranges
                 grade_ranges = {
                 'A+': (990, 1100),
                 'A': (880, 989),
                 'B+': (770, 879),
                 'B': (660, 769),
                 'C': (550, 659),
                 'D': (440, 549),
                 'Fail': (0, 439)
               }

                 # Function to calculate grade for each obtained mark
              def calculate_grade(total_obtained_marks):
                  for grade, (lower, upper) in grade_ranges.items():
                        if lower <= total_obtained_marks <= upper:
                           return grade
                  return 'Unknown'

               # Apply the calculate_grade function to create a new 'grade' column
              df['grade'] = df['Total_Obtained_Marks'].apply(calculate_grade)

               # Group by Year and grade, then count occurrences
              grade_distribution = df.groupby(['Year', 'grade']).size().reset_index(name='count')

                 # Plot grade distribution
              with col2:
                  st.title('Grade Distribution by Year')
                  fig = px.bar(grade_distribution, x='Year', y='count', color='grade',
                     title='Grade Distribution by Year',
                     labels={'count': 'Number of Students', 'Year': 'Academic Year', 'grade': 'Grade'},
                     barmode='group')
                  st.plotly_chart(fig, use_container_width=True)

              with col1:
                  st.title('Science Subjects Performance')
                  # Define science subjects
                  science_subjects = {'PHY', 'GSC', 'BIO', 'CH'}

                  # Filter DataFrame to include only science subjects
                  science_df = df[df['sub6'].isin(science_subjects)]

                  selected_subject = st.selectbox("Select Subject", sorted(science_subjects))
                  # Define marks column corresponding to the selected subject
                  marks_column = selected_subject + '-Marks'
                  # Extract marks related to the selected science subject from the marks6 column
                  science_df[marks_column] = science_df.apply(lambda row: row['marks6'] if row['sub6'] == selected_subject else None, axis=1)
 
                  # Drop rows with NaN values in marks column
                  science_df = science_df.dropna(subset=[marks_column])

                  # Group the DataFrame by Year and gender, calculate the mean marks obtained
                  subject_performance = science_df.groupby(['Year', 'sex']).agg({marks_column: 'mean'}).reset_index()
                  # Plot a line chart showing the year-wise trend of male and female performance in the selected subject
                  fig = px.line(subject_performance, x='Year', y=marks_column, color='sex', 
                     labels={'Year': 'Year', marks_column: 'Mean Marks'},
                     title=f'Year-wise Performance Trend in {selected_subject}',
                     line_shape='linear')

                  fig.update_yaxes(tickformat=".0f")

                  # Show the line chart
                  st.plotly_chart(fig, use_container_width=True)
 
              government_df = df[df['is_government'] == 1]
              def calculate_grade(total_obtained_marks):
                  for grade, (lower, upper) in grade_ranges.items():
                     if lower <= total_obtained_marks <= upper:
                        return grade
                  return 'Unknown'
              government_df['grade'] = government_df['Total_Obtained_Marks'].apply(calculate_grade)
              government_grade_distribution = government_df.groupby(['Year', 'grade']).size().reset_index(name='count')
              with col1:
                  st.title('Grade Distribution by Year - Government Schools')
                  fig_government = px.bar(government_grade_distribution, x='Year', y='count', color='grade',
                             title='Grade Distribution by Year - Government Schools',
                             labels={'count': 'Number of Students', 'Year': 'Academic Year', 'grade': 'Grade'},
                             barmode='group')
                  st.plotly_chart(fig_government, use_container_width=True)

               # Group by Year, gender, and grade for government DataFrame, then count occurrences
              government_grade_distribution = government_df.groupby(['Year', 'sex', 'grade']).size().reset_index(name='count')

               # Plot grade distribution for male and female students in government schools
              with col2:
                st.title('Gender-wise Grade Distribution in Government Schools')
                fig_government_gender = px.bar(government_grade_distribution, x='Year', y='count', color='grade', facet_col='sex',
                                   title='Gender-wise Grade Distribution in Government Schools',
                                   labels={'count': 'Number of Students', 'Year': 'Academic Year', 'grade': 'Grade'},
                                   barmode='group')
                st.plotly_chart(fig_government_gender, use_container_width=True)

              nongovernment_df = df[df['is_government'] == 0]
              def calculate_grade(total_obtained_marks):
                  for grade, (lower, upper) in grade_ranges.items():
                     if lower <= total_obtained_marks <= upper:
                        return grade
                  return 'Unknown'
              nongovernment_df['grade'] = nongovernment_df['Total_Obtained_Marks'].apply(calculate_grade)
              nongovernment_grade_distribution = nongovernment_df.groupby(['Year', 'grade']).size().reset_index(name='count')
              with col1:
                  st.title('Grade Distribution by Year - Non Government Schools')
                  fig_nongovernment = px.bar(nongovernment_grade_distribution, x='Year', y='count', color='grade',
                             title='Grade Distribution by Year - Non Government Schools',
                             labels={'count': 'Number of Students', 'Year': 'Academic Year', 'grade': 'Grade'},
                             barmode='group')
                  st.plotly_chart(fig_nongovernment, use_container_width=True)

              nongovernment_grade_distribution = nongovernment_df.groupby(['Year', 'sex', 'grade']).size().reset_index(name='count')
               # Plot grade distribution for male and female students in government schools
              with col2:
                st.title('Gender-wise Grade Distribution in Non Government Schools')
                fig_nongovernment_gender = px.bar(nongovernment_grade_distribution, x='Year', y='count', color='grade', facet_col='sex',
                                   title='Gender-wise Grade Distribution in Non-Government Schools',
                                   labels={'count': 'Number of Students', 'Year': 'Academic Year', 'grade': 'Grade'},
                                   barmode='group')
                st.plotly_chart(fig_nongovernment_gender, use_container_width=True)

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
