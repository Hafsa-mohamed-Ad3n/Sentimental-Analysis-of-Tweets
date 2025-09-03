
# Sentiment Analysis of Tweets about Apple and Google Products
![Sentiment Analysis](Image/Sentiment_Analysis_Projects.png)

## Authors

1. [Hafsa M. Aden](https://www.linkedin.com/in/hafsa-m-aden-330451223/)
2. [Ryan Karimi](https://www.linkedin.com/in/ryan-karimi-39a701326/)
3. [Harrison Kuria](https://www.linkedin.com/in/harrison-kuria-md-a35487a4/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
4. [Rose Muthini](https://www.linkedin.com/in/syomiti-muthini-03849a153/)
5. [Lewis Mbugua]
6. [Elizabeth Ogutu](https://www.linkedin.com/in/elizabeth-ogutu-36222b1a6/)

## Project Overview

- The aim of this project is to build a Natural Language Processing (NLP) model that can classify tweets about Apple and Google products into Positive, Negative, or Neutral sentiments.
- By leveraging machine learning techniques, the project seeks to provide a scalable way to analyze public opinion from social media, offering valuable insights into customer perceptions and brand sentiment.
- The ultimate goal is to demonstrate how our analysis can support decision-making and market understanding for technology companies.

### Project Colloborations

We used [Notion](https://www.notion.so/Tweet-Sentiment-Analysis-25fdb8fbd6cf801baef5ce49ed846806?source=copy_link) as our project management tool to organize the project timeline, assign and track tasks, and coordinate team contributions, ensuring smooth and effective collaboration throughout the project.


### Objectives

- Exploratory Data Analysis (EDA)
     - Create visualizations (word clouds) to understand the nature of the text and most common words. 
     - Plot histograms to analyze the distribution of sentiment classes.  

- Data Preprocessing - Preparing the tweet text for sentiment analysis by 
     - Removing URLs, user mentions, hashtags, numbers, and special characters
     - Perform tokenization, stop-word removal, lemmatization, and stemming on the text data.  
     - Encoded target variable and vectorized feature matrix for machine learning.
     - Building pipeline to combine preprocessing, feature extraction, and modelling into a single reproducible process.

- Model Selection & Deplyoment.     
     - Build an Natural Language Processing (NLP) model to analyze Twitter sentiment about Apple and Google products.
     - Evaluate the model (accuracy, precision, recall, F1-score).  
     - Provide insights on how this model can be useful for businesses. 

## Project Workflow    

### Data Source.

The dataset used in this project comes from [CrowdFlower](https://data.world/crowdflower/brands-and-product-emotions) and contains over 9,000 tweets, each labelled by human raters as expressing a positive, negative, or neutral sentiment.

### Exploratory Data Analysis (EDA)

- In order to better understand the dataset and prepare it for sentiment analysis, we focused on the following checks:
      - Preview the data: Inspect the first few rows to quickly grasp the dataset’s structure.
      - Handled any missing data that could introduce bias or cause issues during preprocessing and modelling.
      - Removed duplicated tweets to prevent overrepresentation of certain entries.
      - Reviewed the balance of sentiment categories, since skewed classes may bias the model toward majority classes.
      - Explored brand distribution and compared tweets related to Apple vs. Google to see if one brand dominates the dataset.

![Sentiment Distribution](Image/Sentiment%20Distribution.png)
![Brand Distribution](Image/Brand%20Distribution.png)

### Data Preprocessing

- To prepare the tweets for modelling, we applied several text-cleaning and transformation steps:
     - Text Cleaning: Remove URLs, user mentions, hashtags, numbers, and special characters.
     - Tokenization: Break sentences into words for analysis.
     - Stop-word Removal: Exclude common words (the, and, is) that add little meaning.
     - Lemmatization: Reduce words to their base forms to treat similar terms consistently.
     - Vectorization: Convert text into numerical features using techniques like TF-IDF or CountVectorizer.
     - Label Encoding: Encode sentiment classes into numeric values for model training.

### Feature Engineering

- To convert textual data into machine-readable form, we applied:
      - Bag of Words (BoW): Captures word frequency within tweets.
      - Term Frequency–Inverse Document Frequency (TF-IDF ): Assigning higher weight to distinctive words while reducing weight for common ones.
      - N-grams: Included bi-grams and tri-grams to capture short word sequences that add context beyond single words.

- These allowed our models to better capture the semantic and syntactic patterns of the tweets.

![Feature Engineering](Image/Feature%20Engineering.png)

### Pipelines

- We built pipelines to combine preprocessing, feature extraction, and modelling into a single reproducible process. This ensured:
     - Consistency across training and testing
     - Simplified experimentation with different models
     
### Modelling

- The model building process for this project was carried out in four main steps:
   - Data preparation:
      - Features (X): The text of each tweet (tweet_text).
      - Target (y): Whether an emotion was directed at a brand (is_there_an_emotion_directed_at_a_brand_or_product).
      - The dataset was split into training (80%) and testing (20%) sets.

   - Baseline Model Comparison - We tested three ML classifiers to establish baseline performance:
      - Logistic Regression
      - Support Vector Machine (SVM)
      - Random Forest

   - Each model was trained on the pre-processed training data and evaluated on the test set. Key metrics included:
      - Accuracy
      - Precision, Recall, F1-score (per class)
      - Classification reports

   - To better understand the performance of the model:
      - We plotted Bar charts showing accuracy scores for all models for comparisons.
      - Heatmaps for each model to visualize prediction errors across classes.
      - Detailed breakdown of Logistic Regression performance (precision, recall, F1).

   - We performed hyperparameter optimization using GridSearchCV for the two strongest text classification models
   - The grid search was performed with 5-fold cross-validation, selecting the best parameters based on accuracy.

### Observation

- Model Performance:
   - Among the baseline models, Logistic Regression achieved the highest accuracy (~70%) and overall balanced performance across classes.
   - Linear Support Vector Machine performed slightly lower (~68%) but showed comparable results.
   - Random Forest underperformed (~63%), struggling particularly with minority classes.

- After Hyperparameter Tuning:
   - Both Logistic Regression and Linear SVM converged to similar performance (~65% accuracy), but showed improved recall for the Negative emotion class.
   - Performance on the “I can’t tell” class remained poor across all models, likely due to class imbalance and limited training samples.

| Model                     | Accuracy |
| --------------------------| -------- | 
| Logistic Regression       | 0.70     |
| Linear SVM                | 0.68     |
| Random Forest             | 0.63     |
| Tuned Logistic Regression | 0.65     | 
| Tuned Linear SVM          | 0.65     |


![Classification Heatmap](Image/Classification%20Heatmap.png)

![Model Accuracy Comparison](Image/Model%20Accuracy%20Comparison.png)

![Model Comparison](Image/Model%20Comparison.png)

### Deployment Overview

- We deployed our sentiment analysis model so it can be used outside the notebook environment and accessed by different types of users.
   - The trained model pipeline (TF-IDF + classifier + label encoder) was saved in a reusable format to ensure predictions remain consistent every time.
   - The model logic was kept separate from the serving layer, making the system easier to manage and update.

- We provided two ways to interact with the model:
   - Flask Web App→ website where users can enter text and immediately see the sentiment prediction. This was designed for demonstrations and quick reviews.
   - FastAPI → an API that allows developers to send text to the model and get results programmatically. It includes automatic documentation, making integration straightforward.

- The model is hosted on Render which is connected to our GitHub repository. This means that whenever we push updates to GitHub, Render automatically rebuilds and redeploys the service. This avoids manual server setup and ensures the service is always up to date.

- This setup allows:
    - End-users → to interact with the model through the web interface.
    - Developers → to integrate the model into their applications via the API.
    - Future scaling → the system can grow as demand increases.

Overall, deploying the model moved the project from experimentation to real-world use, balancing simplicity and performance.

[Live Demo on Render](https://deployment-yjiq.onrender.com/) 

### Recommendations:

- The model can be applied to track customer emotions and opinions towards the company’s products helping to capture market sentiment in real time.
- By connecting the model to Twitter's APIs the company can automatically filter tweets mentioning the brand or competitors and classify them into sentiment categories.
- Positive sentiment can be amplified in marketing campaigns to highlight brand strengths while the negative sentiment should be flagged for further analysis, enabling the company to identify pain points and potential areas for product or service improvement.
- Insights from competitor related tweets can help the company understand what customers value in rival products and adapt strategies accordingly.


### Future Improvement
- Explore more advanced Modelling approaches like BERT for richer text representations beyond TF-IDF.
- Implement a feedback loop to retrain the model with newly collected tweets, ensuring it adapts to evolving language such as slang, new product references, and emojis.

