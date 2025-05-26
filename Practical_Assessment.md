**🔍 Practical Assessment – NLP Data Scientist (Refined)**

You are provided with patient\_behavior\_data.csv, containing structured medical/behavioural features and a doctor\_notes text field. Your task is to explore, model, and anonymise this data using advanced NLP and ML techniques.

---

**✨ SECTION 1: Exploratory Data Analysis (EDA)**

**Tasks:**

1. Load and clean the CSV (semicolon-delimited).  
2. Summarise key metrics:  
   * BMI, blood pressure by gender and medication.  
   * Distribution of behavioural ratings (-2 to 2).  
3. Visualisations:  
   * Correlation heatmap between behaviour fields.  
   * Boxplot: concentration vs medication.  
   * Scatter plot: BMI vs systolic, coloured by hyperactivity.

**Deliverables**:  eda\_analysis.ipynb

---

**💬 SECTION 2: NLP on doctor\_notes**

**Tasks:**

1. Preprocess notes: tokenise, lemmatise, remove stopwords.  
2. Use NER (spaCy or scispaCy) to extract medical terms.  
3. Train a model to classify **mood** from doctor\_notes (baseline \+ explanation).  
   * Use TF-IDF \+ Logistic Regression  
   * Visualise important words using LIME or SHAP

**Deliverables**: nlp\_pipeline.ipynb

---

**🧠 SECTION 3: Model Training – ML & Deep Learning**

**Tasks:**

1. Prepare features: encode categorical and normalise numeric.  
2. Baseline: Predict concentration using Random Forest.  
3. Deep models:  
   * **LSTM**: Predict impulsivity from doctor\_notes.  
   * **CLSTM (CNN \+ LSTM)**: Combine text and tabular features.  
4. Compare model performance (F1, MAE, etc.)

**Deliverables**:

* model\_training.ipynb  
* model\_comparison.pdf (table \+ summary)

---

**🔐 SECTION 4: Anonymisation & Ethics**

**Tasks:**

1. Mask PII: hash/remove name, surname, patient\_id.  
2. Redact PII in doctor\_notes using spaCy NER (names, dates, locations → \[REDACTED\]).  
3. Export anonymized\_patients.csv.

**Deliverables**:

* anonymizer.py or anonymisation.ipynb  
* Final CSV: anonymized\_patients.csv

---

**🐙 SECTION 5: Git & Docker**

**GitHub Tasks:**

* Create a repo nlp-data-assessment  
* Include:  
  * Readme.md with setup instructions  
  * Add gitnore  
  * Issue: Suggest one improvement to the data or pipeline

**Docker Tasks:**

* Build a Dockerfile to:  
  * Install dependencies  
  * Run your model training script  
* Commands to run:

bash

CopyEdit

docker build \-t nlp-app .

docker run nlp-app

**Deliverables**:

* GitHub repo link  
* Dockerfile  
* Screenshot or logs of successful Docker run

 

 

 

