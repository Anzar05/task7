# Task 7: Sentiment Analysis on Tweets

## 📌 Project Overview

This project performs **Sentiment Analysis on Tweets** using Natural Language Processing (NLP). The tweets are classified into three categories:

* Positive 🙂
* Neutral 😐
* Negative 🙁

The analysis is performed using **TextBlob** and **VADER Sentiment Analyzer** in Python.

---

## 📊 Dataset

The dataset was taken from Kaggle:

**Twitter Sentiment Dataset:**
[https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset)

The dataset contains tweets that are analyzed to determine their sentiment.

---

## 🛠️ Technologies Used

* Python
* Pandas
* TextBlob
* NLTK
* VADER Sentiment Analyzer
* Matplotlib

---

## 📂 Project Steps

1. Import required libraries
2. Load the dataset
3. Clean the tweet text
4. Apply sentiment analysis using TextBlob
5. Apply sentiment analysis using VADER
6. Classify tweets into Positive, Neutral, or Negative
7. Visualize sentiment distribution

---

## ▶️ Installation

Install the required libraries using:

```
pip install pandas textblob nltk vaderSentiment matplotlib
```

---

## 💻 Example Code

```python
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load dataset
df = pd.read_csv("tweets.csv")

# TextBlob Sentiment

def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


df['TextBlob_Sentiment'] = df['text'].apply(get_sentiment)

# VADER Sentiment
analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = analyzer.polarity_scores(str(text))
    
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"


df['VADER_Sentiment'] = df['text'].apply(vader_sentiment)

print(df.head())
```

---

## 📈 Output

* Tweets classified into Positive, Neutral, and Negative sentiments
* Sentiment distribution visualization

---

## 🎯 Learning Outcomes

* Understanding Natural Language Processing basics
* Performing sentiment analysis on text data
* Using TextBlob and VADER libraries
* Visualizing results using Matplotlib

---

## 📌 Conclusion

This project demonstrates how machine learning and NLP techniques can be used to analyze public sentiment from tweets. Sentiment analysis helps understand opinions and trends from social media data.

---

## 🔗 Author

**Shaik Mahamood Anzar**
Data Science Intern

---

## 📎 Task

Internship Task 7 - Sentiment Analysis on Tweets
