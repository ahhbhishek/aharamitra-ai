# Aharamitra -- AI-Based Food Risk & Portion Intelligence System

## Overview

Aharamitra is a machine learning system designed to analyze food consumption and its impact on health. It predicts:

* **Glucose spike risk (classification)**
* **Safe portion size (regression)**

The system combines user health parameters with food nutritional data to generate actionable insights.

---

## Problem

Food choices are often made without understanding their metabolic impact, especially for individuals with conditions like diabetes. There is a lack of systems that:

* Assess food risk dynamically
* Suggest safe consumption quantities

---

## Solution

Aharamitra models the relationship between:

* User health attributes (age, BMI, diabetes status)
* Context (region, festival, fasting state)
* Food characteristics (glycemic index, carbs, sugar)

It outputs:

* Predicted glucose spike risk
* Recommended safe portion size

---

## Dataset & Features

### Input Features

* Age
* BMI
* Diabetes status
* Fasting state
* BMI category
* Festival
* Region
* Food name
* Glycemic index
* Carbohydrates per item (g)
* Sugar per item (g)

### Targets

* **Glucose spike risk** (classification)
* **Safe portion count** (regression)

---

## Approach

### Data Preprocessing

* Label encoding for categorical variables
* Separate encoding for target labels

### Models Used

* **XGBoost Classifier** → predicts glucose spike risk
* **XGBoost Regressor** → predicts safe portion size

### Train-Test Split

* 80/20 split for evaluation

---

## Inference Example

Input:

```python
{
    "age": 45,
    "bmi": 27.5,
    "diabetes_status": 1,
    "fasting_state": 0,
    "bmi_category": "overweight",
    "festival": "Ganesh Chaturthi",
    "region": "Maharashtra",
    "food_name": "Steamed Modak",
    "glycemic_index": 60,
    "carbs_per_item_g": 20,
    "sugar_per_item_g": 10
}
```

Output:

* Glucose risk prediction
* Recommended portion size

---

## Model Persistence

Models and encoders are saved using `joblib`:

* Risk classifier
* Portion regressor
* Label encoders

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/aharamitra.ipynb
```

---

## Future Work

* Personalized diet recommendation system
* Integration with real-time health data
* Deployment as a web application
* Expansion to larger and more diverse datasets

