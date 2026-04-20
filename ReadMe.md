# Hindi Fake News & Misinformation Detection System 

An end-to-end Natural Language Processing (NLP) pipeline designed to classify Hindi text as either genuine news or misinformation. This project tackles the growing challenge of fake news in regional languages by leveraging both traditional machine learning and deep learning architectures.

##  Application Preview
The application is currently live and deployed on Hugging Face Spaces. 

 **[Click Here to Test the Hindi Fake News Detector!](https://huggingface.co/spaces/rkc2026/Hindi-Fake-News-Detector)**

##  Overview

The system is built in two primary phases:
1. **Model Training:** Conducted on Kaggle, experimenting with different architectures including Naive Bayes for baseline text classification and Long Short-Term Memory (LSTM) networks for capturing sequential context in Hindi sentences.
2. **Web Application:** A local web interface built to serve the model, allowing users to input Hindi text and receive real-time classification.

##  Tech Stack & Tools
* **Programming Language:** Python
* **Deep Learning:** Pytorch (LSTMs)
* **Machine Learning:** Scikit-learn (Naive Bayes)
* **NLP Processing:** NLTK / IndicNLP (for Hindi text tokenization and stop-word removal)
* **Web Framework:** Streamlit (via `app.py`)
* **Environment:** Kaggle (Training), Local Virtual Environment (Deployment)

##  Step-by-Step Methodology

The development of this system followed a structured machine learning pipeline, moving from raw data processing to web deployment.

**Step 1: Data Acquisition**
* Sourced the Hindi Fake News Detection Dataset (HFDND) containing over 17,000 labeled news records.
* Loaded and explored the dataset on Kaggle to understand the distribution of genuine (0) versus fake (1) text samples.

**Step 2: Text Preprocessing & Tokenization**
* Cleaned the raw Hindi text by removing special characters, English punctuation, and standard Hindi stopwords.
* Applied custom tokenization to break down complex sentences into individual Hindi root words.
* Converted the textual tokens into numerical vectors using TF-IDF (for the machine learning baseline) and word embeddings with sequence padding (for the deep learning model).

**Step 3: Baseline Modeling (Machine Learning)**
* Established a performance baseline using a Naive Bayes classifier.
* Trained on the TF-IDF vectorized data, providing a lightweight, statistically driven method for text classification.

**Step 4: Deep Learning Implementation (LSTM)**
* Architected a Long Short-Term Memory (LSTM) neural network to capture the sequential dependencies and semantic context within Hindi sentences.
* Trained the model using the padded word embeddings, optimizing for accuracy and minimizing loss over multiple epochs.
* Exported the final trained model weights and architecture.

**Step 5: Web Application Deployment**
* Developed a local frontend using Streamlit (`app.py`) to serve the trained model.
* Integrated the preprocessing pipeline into the app so raw user input is automatically cleaned and tokenized before being passed to the model for real-time inference.
