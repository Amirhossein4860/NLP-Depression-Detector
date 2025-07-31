# NLP-Depression-Detector
A Natural Language Processing project focused on detecting signs of depression in Reddit posts. This end-to-end pipeline includes data preprocessing, EDA, feature extraction with TF-IDF, model training using various classifiers, and evaluation to aid mental health research.

---

# 🧠 Reddit Depression Detector  
A complete NLP pipeline for detecting **depression in Reddit posts** using classic machine learning models and linguistic features.  
🧩 Includes a **ready-to-use `.exe` application** for instant predictions!

---

## 🌟 Project Overview

This project builds an end-to-end **natural language processing (NLP)** solution to identify signs of **depression** in Reddit posts. By combining text preprocessing, linguistic feature extraction, and classic ML algorithms, it creates a tool for detecting mental health signals in social media content.

Use cases include:

- 🧠 Mental health awareness tools  
- 📊 Social media monitoring  
- 🧪 Academic research on online behavior  
- 💬 Support bots and screening tools

---

## 📂 Dataset

- **Source**: [Kaggle – Depression Reddit Cleaned](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned)  
- **Data**: Cleaned Reddit posts labeled for depressive content.

| Column         | Description                          |
|----------------|--------------------------------------|
| `clean_text`   | Preprocessed Reddit post             |
| `label`        | `1` = Depressed, `0` = Not Depressed |

---

## 🔄 Pipeline Overview

### 🧹 Text Preprocessing
- Fix Reddit-specific contractions (e.g., "ive", "im")  
- Lowercase, punctuation & number removal  
- Tokenization, lemmatization, and POS tagging

### 🔍 Feature Engineering
- Bag-of-Words  
- TF-IDF Vectorization  
- Word2Vec Embeddings  
- POS tag frequency features

### ⚙️ Model Training
Trained and compared:
- 🔹 Logistic Regression *(best overall performance)*  
- 🔸 Decision Tree Classifier  
- 🔹 Multi-Layer Perceptron (MLP)

### 📈 Evaluation
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrices

---

## 🚀 How to Run the Project

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/Amirhossein4860/NLP-Depression-Detector.git
cd NLP-Depression-Detector
```

### 📦 2. Install Dependencies

```bash
$ pip install -r Depression-requirement.txt
```

### 📓 3. Run the Notebook

```bash
jupyter notebook NLP_RedditDepression_CLF.ipynb
```
Execute the cells step-by-step to reproduce the full NLP pipeline.

### 4️⃣ Run the GUI App (Optional)
- For real-time emotion prediction with a user interface:
```sh
$ python Depression.py
```

### **5️⃣ Use the EXE File (No Python Required)**
- Download the .exe from the Releases page and run the app directly.

## 📊 Key Findings :

### - Depressive posts use more first-person pronouns and emotional keywords

### - Word clouds and POS tag distributions show distinct patterns

### - **TF-IDF + Logistic Regression yielded the highest performance balance**

## 🎨 Project Highlights :

- ✅ Full NLP + ML pipeline

- 📎 Semantic and syntactic features

- 🔍 Model explainability via metrics and visuals

- 🖥️ GUI + .exe app for non-technical users

🧾 Clean and modular code for extensibility

