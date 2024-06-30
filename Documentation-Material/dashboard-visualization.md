## It is essential to first define Key Performance Indicator (KPIs) that we want to visualize on dashboard.

### Here are some KPIs to consider:

1. What is the total number of students over the year?
2. What is the ratio of passing and failing students in exams?
3. How does this ratio change across different subjects or years?
4. How is the distribution of male and female students in the dataset?
5. How does students performance vary across different subjects?
6. What are performance overall trends across multiple years?
7. How does performance vary across different groups (urban/rural)?
8. What percentage of students continue their education from one year to the next?
9. Are there institutions consistently exhibiting high or low performance?
10. Forecast academic outcomes.

### More features to add in dashboard for user interactivity:

1. Allow users to choose specific subject or group oof subjects for analysis (by adding selectbox of 'Select subject')
2. Allow users to select regions or institutes to compare educational patterns across these categories
3. Allow users to explore educational patterns based on demographics such as age, gender, or group. For example, users could analyze how the performance of specific demographics has evolved over the years
4. Implement a feature to showcase the top-performing students or institutions based on various criteria such as marks, pass rates, or any other relevant measure.
5. Predictive modeling 

## Add an authentication component in streamlit app: 
- Login/logout page
- user authentications
(Options we can add, 1. Create a new account, 2. Login to existing account, 3. login as guest)
(We will create a side bar menu using four options:
1. Home
2. Add Account
3. Update Password
4. Delete Account
(We will use DataBase(Sql or postgres) to store user details)
