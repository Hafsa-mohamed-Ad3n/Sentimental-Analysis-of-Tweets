
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
     - PIPELINE( Ryan?????)  

- Model Selection & Deplyoment.     
     - Build an Natural Language Processing (NLP) model to analyze Twitter sentiment about Apple and Google products.
     - Evaluate the model (accuracy, precision, recall, F1-score).  
     - Provide insights on how this model can be useful for businesses. 

## Project Workflow    

### Data Source.

The dataset used in this project comes from [CrowdFlower](https://data.world/crowdflower/brands-and-product-emotions) and contains over 9,000 tweets, each labeled by human raters as expressing a positive, negative, or neutral sentiment.

### Exploratory Data Analysis (EDA)

- In order to better understand the dataset and prepare it for sentiment analysis, we focused on the following checks:
      - Preview the data: Inspect the first few rows to quickly grasp the dataset’s structure.
      - Handled any missing data that could introduce bias or cause issues during preprocessing and modeling.
      - Removed duplicated tweets to prevent overrepresentation of certain entries.
      - Reviewed the balance of sentiment categories, since skewed classes may bias the model toward majority classes.
      - Explored brand distribution and compared tweets related to Apple vs. Google to see if one brand dominates the dataset.

### Data Preprocessing

- To prepare the tweets for modeling, we applied several text-cleaning and transformation steps:
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

### Pipelines

- We built pipelines to combine preprocessing, feature extraction, and modeling into a single reproducible process. This ensured:
     - Consistency across training and testing
     - Simplified experimentation with different models
     
### Modeling

- We evaluated five ML classifiers using TF-IDF features extracted from the tweets:
     - Logistic Regression → 69%
     - Naive Bayes → 62%
     - Support Vector Machine (SVM) → 68%
     - Random Forest → 65%
     - XGBoost → 66%

- From the initial comparison, Logistic Regression and Support Vector Machine (SVM) emerged as the top performers. We then fine-tuned both models using GridSearchCV to optimize hyperparameters.

- We selected Logistic Regression as our model of choice because it provided the best balance of performance and simplicity:
     - Consistently accurate across sentiment classes
     - Efficient & fast to train compared to SVM and ensemble methods
     - Robust generalization with minimal overfitting

  
## Conclusion & Recommendations



