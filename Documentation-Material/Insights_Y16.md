# Preliminary Wrangling:
- Handling Missing values
- Convert Data types of required columns
- Created another column 'Total Obtained Marks' iterate through each subject and add obtained marks to total. Because already present column of 'Obtained-Marks contains numeric or non numeric values so lay with this column was causing problem foor further analysis.
- Renaming the columns 

# Exploratory Data Analysis: 
- Distribution of subjects in each subject column
- Calculate Average score of each subject
- Boxplot to identify subjects with high variation in marks
- Distribution of students in each subject
- Countplot for distribution of students in all subjects by Gender
- Identify popular subjects based on frequency

# Merged Tehsil/District data file with Main file
- While merging this file with main data issue that I faced because the size of both file is different
I am trying to merge both files on 'inst_code' column because of variation in sizes it shows null values that rows left. 
After merging 'merged_inst_data' with main 'ssc_16' data so the null values in other columns indicates that there are inst_code values in the main_data ('ssc_16) DataFrame that do not have a corresponding match in the other DataFrame (merged_inst_data). The null values are filled in for columns from the other DataFrame(s) when no match is found.

Applied another merging technique "inner_merge"
So based on my data, I applied another method an 'inner' merge will only include rows with matching 'inst_code' values in both DataFrames. But with this approach I lost some of data values. 
First the size of data was 92700 rows, after applying 'inner_merge' the data size become 71008 rows.

# Research questions that I have covered:
- What was the gender distribution in year 2016?
- How has the overall pass rate in year 2016?
- How does the gender distribution vary across different institutes?
- What was count of Male and Female institute in each District?
- What was count of Male and Female institute in each tehsil?
- Calculate average marks for each subject
- What is the average age of students taking the exit exam?
- How does the average age of students differ across tehsils?
- How does the distribution of students vary across different tehsils?
- What is the distribution of student ages in urban and rural institutes?
- What is the distribution of student ages in government and non-government institutes?
- Passing rate overall?
- fail rate of students overall?
- Pass rate district wise?
- pass rate tehsil wise?
- what is the pass rate gender wise in each district?
- what is the pass rate gender wise in each tehsil?
- Fail rate gender wise in each distrivt/tehsil?
- How do students from different institutions (Inst_name) perform in comparison to each other?
- Are there any significant differences in performance between male and female students?
- Are there any observable differences in performance between urban and rural schools within each tehsil?
- What has been Popular Subjects in Government/ non government Schools Based on Frequency?
- what are the subjects are most commonly chosen by students?
- Explore the distribution of institutes based on their governance type (Government or Non-Government) and their location in urban and non-urban areas.
- Count of government and non-government institutes in urban and non-urban areas?
- Tehsil wise institute count
- bar plot for urban vs. non-urban performance based on pass rate and gender
- What will be the tehsil-wise urban vs. non-urban performance based on pass rate and gender
