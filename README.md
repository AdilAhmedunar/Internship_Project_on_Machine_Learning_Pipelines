<div>
       <h1>Startup Acquisition Status Modeling Using Machine Learning Pipelines</h1>
</div>
 Our project aims to analyze the financial circumstances of companies and their fundraising objectives.

# Project Description
<div style="text-align: center;">
  <p align="justify">
    The project aims to predict the acquisition status of startups based on various features such as funding rounds, total funding amount, industry category, and geographic location. The objective is to develop a machine learning model that accurately classifies startups into different acquisition status categories, including Operating, IPO, Acquired, or closed. This problem will be addressed using a Supervised Machine Learning approach by training a model based on the historical data of startups that were either acquired or closed. By leveraging machine learning pipelines, we preprocess the data, select relevant features, and train models to classify startups into different acquisition status categories. The project utilizes Python libraries such as scikit-learn, pandas, matplotlib, seaborn, joblib, and XGBoost for model development and evaluation. The goal is to provide insights into the factors influencing startup acquisition and build a predictive tool that can assist stakeholders in making informed decisions.
</div>

# Software Development Life Cycle (SDLC) Model

## Agile Approach

Our project follows the Agile Software Development Life Cycle (SDLC) model, which is well-suited for iterative and collaborative projects like machine learning development. The Agile approach emphasizes flexibility, adaptability, and customer collaboration throughout the project lifecycle. Here's how we applied Agile principles in our project:

1. **Iterative Development**: We embraced iterative development cycles to continuously refine and improve our machine learning models based on feedback and new insights gained during each iteration.

2. **Collaboration and Communication**: Agile principles encouraged regular collaboration and communication among team members, enabling effective management of the project's complexity and ensuring alignment with stakeholders' expectations.

3. **Adaptability to Change**: Agile's adaptive approach allowed us to respond quickly to changes in project requirements, data characteristics, and model performance, ensuring that our solutions remained relevant and effective.

4. **Instructor Feedback**: We actively sought feedback from our mentor and incorporated it into our development process, ensuring that our machine learning models met their needs and expectations.

5. **Continuous Improvement**: Agile principles fostered a culture of continuous improvement, prompting us to regularly reflect on our processes and outcomes, identify areas for enhancement, and implement changes to deliver higher-quality solutions.

By following the Agile SDLC model, we effectively managed the complexity and uncertainty inherent in machine learning projects, delivering valuable and robust solutions to predict startup acquisition status.

## Implementation of Agile Practices

Throughout the project, we implemented various Agile practices, including:

- **Sprint Planning**: We conducted regular sprint planning sessions to define the scope of work for each iteration and prioritize tasks based on their importance and complexity.
- **Daily Stand-up Meetings**: We held daily stand-up meetings to discuss progress, identify obstacles, and coordinate efforts among team members.
- **Continuous Integration and Deployment**: We employed continuous integration and deployment practices to ensure that changes to our machine learning models were integrated smoothly and deployed efficiently.
- **Iterative Testing**: We performed iterative testing throughout the development process to validate the functionality and performance of our models and identify any issues early on.

Through the effective implementation of Agile practices, we were able to deliver a high-quality machine learning solution that met our project objectives and exceeded stakeholders' expectations.


## Flow Chart
<div style="align: center;">
    <img src="FDP.png ">
</div>

# Dataset
<div style="text-align: center;">
  <p align="justify">
<p>This project utilizes a dataset containing industry trends, investment insights, and company information.</p>
<ul>
  <li><strong>Format:</strong> JSON and Excel</li>
  <li><strong>Link to Raw Data:</strong> <a href="#">Excel file</a></li>
  <li><strong>Columns:</strong> id, entity_type, name, category_code, status, founded_at, closed_at, domain, homepage_url, twitter_username, funding_total_usd, country_code, state_code, city, region, etc.</li>
</ul>
<h3>Data Information:</h3>
<ul>
  <li><strong>Total Records:</strong> 196,553</li>
  <li><strong>Data Columns:</strong> 44</li>
  <li><strong>Data Types:</strong> Object, Integer, Float</li>
  <li><strong>Missing Values:</strong> Present in multiple columns</li>
  <li><strong>Data Size:</strong> Approximately 66.0+ MB</li>
</ul>
<p>This dataset serves as the foundation for building the machine learning model to predict the acquisition status of startups based on various features.</p>
</div>
<div style="align: center;">
    <img src="Dataset.JPG ">
</div>

# Data Preprocessing 
<div style="text-align: center;">
  <p align="justify">
<p>The <strong>data preprocessing</strong> phase involved several steps, including:</p>
<ul>
  <li>Deleted columns providing excessive granularity such as <strong>'region', 'city', 'state_code'</strong></li>
  <li>Removed redundant columns such as <strong>'id', 'Unnamed: 0.1', 'entity_type'</strong></li>
  <li>Eliminated irrelevant features such as <strong>'domain', 'homepage_url', 'twitter_username', 'logo_url'</strong></li>
  <li>Handled duplicate values</li>
  <li>Removed columns with high null values</li>
  <li>Dropped instances with missing values such as <strong>'status', 'country_code', 'category_code', 'founded_at'</strong></li>
  <li>Dropped time-based columns such as <strong>'first_investment_at', 'last_investment_at', 'first_funding_at'</strong></li>
  <li>Imputed missing values using mean() and mode() methods in numerical columns and categorical columns accordingly such as <strong>'milestones', 'relationships', 'lat', 'lng'</strong></li>
</ul>
<p>After preprocessing, the DataFrame has the following information:</p>
<ul>
  <li>Total columns: <strong>11</strong></li>
  <li>Non-Null Count: <strong>63585</strong></li>
  <li>Data types: <strong>float64(7), object(4)</strong></li>
  <li>Memory usage: <strong>7.8 MB</strong></li>
</ul>
</div>
<div style="align: center;">
    <img src="Dataset after preprocessing.JPG ">
</div>

# Exploratory Data Analysis (EDA)
<div style="text-align: center;">
  <p align="justify"> 
    <h3>Univariate & Bivariate Analysis</h3>
<p>The <strong>Univaraite & Bivariate Analysis</strong> phases involved exploring relationships between variables in the dataset. Key visualizations and analyses conducted during this phase include:</p>
<ol>
  <li>Visualization of the distribution of the <strong>Status</strong> column, which is the target variable, using a horizontal bar plot.</li>
  <li>Visualization of the distribution of <strong>Milestones</strong> using a histogram.</li>
  <li>Exploring the relationship between <strong>Status</strong> and <strong>Milestones</strong> using a violin plot.</li>
  <li>Visualization of the average funding amount by <strong>Status</strong> using a bar chart.</li>
  <li>Exploring the relationship between <strong>Status</strong> and <strong>Funding Total (USD)</strong> using a violin plot.</li>
</ol>
<p>These visualizations provide insights into how different variables interact with each other and their potential impact on the target variable.</p>
</div>
<div style="align: center;">
    <img src="EDA.JPG ">
    <img src="EDA 2.JPG ">
</div>

# Feature Engineering (FE) 
<div style="text-align: center;">
  <p align="justify"> 
<ol>
  <li><strong>Feature Selection:</strong> We performed feature selection to choose the most relevant features for our analysis.</li>
  <li><strong>Creation of New Features:</strong> We created new features from the existing dataset to enhance predictive power.</li>
  <li><strong>Normalization and Scaling:</strong> We normalized and scaled numerical features to ensure consistency and comparability.</li>
  <li><strong>Encoding Categorical Variables:</strong> We encoded categorical variables to represent them numerically for model training.</li>
  <li><strong>Feature Engineering Documentation:</strong> We documented the entire feature engineering process for transparency and reproducibility.</li>
</ol>
<h3>Creation of New Features from Dataset</h3>
<p>We conducted various operations to create new features:</p>
<ul>
  <li>Converted the 'founded_at' column to datetime format and extracted the year.</li>
  <li>Mapped status values to isClosed values and created a new column.</li>
  <li>Performed Min-Max scaling on selected numerical features.</li>
  <li>Applied one-hot encoding to 'country_code' and 'category_code' columns.</li>
  <li>Label encoded the 'status' column for binary classification.</li>
</ul>
<h3>Feature Selection using Mutual Information (MI)</h3>
<p>We computed mutual information between features and the target variable to identify top-ranked features for model training.</p>
<p>After conducting comprehensive feature engineering, our dataset <code>comp_df</code> has undergone significant transformations. Initially containing 11 columns consisting of 3 categorical variables and 8 numerical variables, it has now expanded to encompass 32 columns while maintaining its original 4682 rows. All variables within <code>comp_df</code> have been converted to numerical format, making them suitable for analytical operations. Our data frame is ready to embark on the next phase of model construction with confidence.</p>  
</div>
<div style="align: center;">
    <img src="FE.JPG ">
</div>

# Model Building 
<div style="text-align: center;">
  <p align="justify"> 
<p>Leading up to the Feature Engineering phase, individual interns diligently prepared their datasets to model startup acquisition statuses. After thorough experimentation and evaluation, three standout models emerged for collaborative refinement by the team.</p>
<p>In the capacity of TEAM C lead, I assumed responsibility for overseeing subsequent tasks until deployment. Initially, our team received directives to explore various models for both binary and multiclass classification:</p>
<ul>
  <li><strong>For Binary Classification:</strong>
    <ul>
      <li>We explored Decision Trees.</li>
      <li>We delved into the intricacies of Support Vector Machines (SVM).</li>
    </ul>
  </li>
  <li><strong>For Multiclass Classification:</strong>
    <ul>
      <li>We investigated the applicability of Multinomial Naive Bayes.</li>
      <li>We explored the potentials of Gradient Boosting.</li>
      <li>We considered the robustness of Random Forest.</li>
      <li>We examined the effectiveness of XGBoost.</li>
    </ul>
  </li>
</ul>
<p>Following exhaustive analysis and collective deliberation, we meticulously selected one model each for binary and multiclass classification. Our choices, prioritizing accuracy, were SVM for binary classification and XGBoost for multiclass classification.</p>
<h3>Model Evaluation</h3>
<p>Each model underwent comprehensive evaluation, scrutinizing metrics such as accuracy, precision, recall, and F1-score. This evaluation process resulted in the creation of a detailed classification report for further analysis and refinement.</p>
</div>

# Machine Learning Pipelines Building 
<div style="text-align: center;">
  <p align="justify"> 
<ol>
  <li>
    <h3>Binary Classification Model:</h3>
    <p>We have developed a binary classification model using Random Forest. This model predicts whether a startup will be acquired or not. It analyzes various features of the startup and determines the likelihood of acquisition.</p>
  </li>
  <li>
    <h3>Multiclass Classification Model:</h3>
    <p>Similarly, we have constructed a multiclass classification model using an XGBoost classifier. Unlike the binary model, this classifier predicts multiple classes of startup status: Operating, IPO, Acquired, or Closed. It evaluates various factors to categorize startups into these different status categories.</p>
  </li>
  <li>
    <h3>Combining Pipelines:</h3>
    <p>Our primary objective is to create three distinct pipelines:</p>
    <ol type="a">
      <li>
        <strong>Binary Classification Pipeline:</strong>
        <p>This pipeline will encapsulate the process of preparing data, training the Random Forest model, and making predictions on whether a startup will be acquired.</p>
      </li>
      <li>
        <strong>Multiclass Classification Pipeline:</strong>
        <p>Similarly, this pipeline will handle data preparation, model training using XGBoost, and predicting the status of startups (Operating, IPO, Acquired, or Closed).</p>
      </li>
      <li>
        <strong>Combined Pipeline:</strong>
        <p>The challenge lies in integrating these two models into a single pipeline. We must ensure that the output of the binary classifier is appropriately transformed to serve as input for the multiclass classifier. This combined pipeline will enable us to efficiently predict startup statuses.</p>
      </li>
    </ol>
  </li>
  <li>
    <h3>Testing and Evaluation:</h3>
    <p>After constructing the combined pipeline, extensive testing will be conducted to validate its functionality and accuracy. We will employ various evaluation metrics to assess the performance of the pipeline, ensuring that it reliably predicts startup statuses.</p>
  </li>
</ol>
<div style="align: center;">
    <img src="https://github.com/AdilAhmedunar/Internship_Project_on_Machine_Learning_Pipelines/assets/38765754/d6e92d47-2c50-414e-8c62-cf4b64d10790">
</div>
</div>
</div>

# Deployment of Project - Django 
<div style="text-align: center;">
  <p align="justify">   
Our deployed project leverages Django, a high-level web framework for Python, to provide a user-friendly interface for interacting with our machine-learning model. Users can now make predictions using the model through a web application, without needing to write any code.

With this deployment, we aim to democratize access to machine learning technology, empowering users from various backgrounds to harness the power of predictive analytics for their specific use cases. We have ensured that our deployed project is robust, scalable, and secure, providing a seamless experience for users while maintaining data privacy and integrity.
</div>

<div style="align: center;">
    <img src="Deployement Mood.JPG ">
</div>
Thank you for joining us on this journey from development to deployment. We're excited to see how our project will impact the world of machine learning and beyond.

# Contributions of the Team 
<div style="text-align: center;">
  <table>
    <thead>
      <tr>
        <th>Name</th>
        <th>Assigned Models</th>
        <th>Contribution</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Adil Ahmed Unar</strong></td>
        <td>Decision Trees (Binary) and XGBoost (Multiclass)</td>
        <td>Data Preprocessing, EDA, Feature Engineering, Model Training, Model Deployment, Report</td>
      </tr>
      <tr>
        <td><strong>Ashrith Komuravelly</strong></td>
        <td>Decision Trees (Binary) and Random Forest (Multiclass)</td>
        <td>Data Preprocessing, EDA, Feature Engineering, Model Training</td>
      </tr>
      <tr>
        <td><strong>B Kartheek</strong></td>
        <td>Decision Trees (Binary) and Multinomial Naïve Bayes (Multiclass)</td>
        <td>Data Preprocessing, EDA, Feature Engineering, Model Training</td>
      </tr>
      <tr>
        <td><strong>Charulatha</strong></td>
        <td>Support Vector Machines (Binary) and Gradient Boosting (Multiclass)</td>
        <td>Data Preprocessing, EDA, Feature Engineering, Model Training</td>
      </tr>
      <tr>
        <td><strong>Mayuri Sonawane</strong></td>
        <td>Support Vector Machines (Binary) and Random Forest (Multiclass)</td>
        <td>Data Preprocessing, EDA, Feature Engineering, Model Training</td>
      </tr>
      <tr>
        <td><strong>Pratik Santosh Akole</strong></td>
        <td>Decision Trees (Binary) and XGBoost (Multiclass)</td>
        <td>Data Preprocessing, EDA, Feature Engineering, Model Training</td>
      </tr>
      <tr>
        <td><strong>Shata Rupendra</strong></td>
        <td>Support Vector Machines (Binary) and Gradient Boosting (Multiclass)</td>
        <td>Data Preprocessing, EDA, Feature Engineering, Model Training</td>
      </tr>
      <tr>
        <td><strong>Vaibhavi Vijay</strong></td>
        <td>Support Vector Machines (Binary) and Multinomial Naïve Bayes (Multiclass)</td>
        <td>Data Preprocessing, EDA, Feature Engineering, Model Training</td>
      </tr>
    </tbody>
  </table>
</div>
<div style="text-align: center;">
  <p>
    For the latest updates and contributions, please visit our GitHub repository:
    <a href="https://github.com/AdilAhmedunar" target="_blank">Your Repository</a>
  </p>
</div>



