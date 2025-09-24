# Workflow: Develop and Deploy Simple Linear Regression App

1.0 **Project Initialization and Logging**   
1.1 Add `devLog.md` for documenting development progress  
1.2 Configure logging in Python (INFO level)  

2.0 **Planning and Design**  
2.1 Write `requirements.txt` to capture dependencies   
2.2 Draft `project_plan.md` to describe HW1 objectives and deliverables  

3.0 **Data and Model Implementation**  
3.1 Implement data generator function (with parameters: slope `a`, noise, number of points)  
3.2 Add data exploration: shape, missing values, descriptive statistics  
3.3 Implement train/test split and linear regression model  
3.4 Add evaluation metrics (MAE, RMSE, R²) and residual plots  

4.0 **Web Application Development**  
4.1 Create `main.py` using Streamlit for the interactive UI  
4.2 Implement sidebar inputs for parameters (`a`, noise, number of points, test size, etc.)  
4.3 Display model results, plots, and diagnostics on the main page  
4.4 Provide download options: OLS summary, `model.joblib`, Markdown report  

5.0 **Version Control and GitHub**  
5.1 Create `.gitignore` to exclude venv, cache, and generated files  
5.2 Commit code and documents with descriptive messages  
5.3 Push repository to GitHub for remote storage and collaboration  

6.0 **Deployment and Testing**  
6.1 Connect GitHub repository to Streamlit Cloud  
6.2 Configure `main.py` as the entry point  
6.3 Deploy and obtain public application URL  
6.4 Test the application in browser with different parameter inputs  
6.5 Verify metrics and plots update correctly and share the link for feedback  

