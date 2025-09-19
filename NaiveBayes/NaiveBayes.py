#!/usr/bin/env python3
"""
Naive Bayes Classifier - Step by Step Implementation with scikit-learn
This script demonstrates a Naive Bayes classifier with all required steps:

1. Converts dataset into a frequency table.
2. Creates a likelihood table with probabilities.
3. Calculates posterior probabilities for each class.
4. Corrects zero-probability errors using Laplacian correction.
5. Compares results with scikit-learn's MultinomialNB.

"""

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Sample dataset (Play Tennis)
data = {
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain',
                'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool',
             'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No',
             'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

print("=== STEP 0: Dataset ===")
print(df, "\n")


# 1. Frequency table
freq_table = pd.crosstab(index=[df['Weather'], df['Temp']], columns=df['Play'])
print("=== STEP 1: Frequency Table ===")
print(freq_table, "\n")

# 2. Likelihood Table (with Laplace correction)
def calculate_likelihood(feature, label, laplace=1):
    """
    Creates a likelihood table with Laplace correction.
    """
    likelihood = {}
    labels = df['Play'].unique()
    for value in df[feature].unique():
        likelihood[value] = {}
        for l in labels:
            count = len(df[(df[feature] == value) & (df['Play'] == l)])
            total = len(df[df['Play'] == l])
            prob = (count + laplace) / (total + len(df[feature].unique()))
            likelihood[value][l] = round(float(prob), 4)
    return pd.DataFrame(likelihood).T

print("=== STEP 2: Likelihood Table for Weather ===")
print(calculate_likelihood('Weather', 'Play'), "\n")

print("=== STEP 2: Likelihood Table for Temp ===")
print(calculate_likelihood('Temp', 'Play'), "\n")

# 3. Posterior Probability Calculation
def calculate_posterior(weather, temp):
    prior_yes = len(df[df['Play'] == 'Yes']) / len(df)
    prior_no = len(df[df['Play'] == 'No']) / len(df)

    weather_likelihood = calculate_likelihood('Weather', 'Play')
    temp_likelihood = calculate_likelihood('Temp', 'Play')

    p_yes = prior_yes * weather_likelihood.loc[weather, 'Yes'] * temp_likelihood.loc[temp, 'Yes']
    p_no = prior_no * weather_likelihood.loc[weather, 'No'] * temp_likelihood.loc[temp, 'No']

    total = p_yes + p_no
    return {"Yes": round(float(p_yes / total), 4), "No": round(float(p_no / total), 4)}

# 4. Interactive User Input
print("Available Weather options:", df['Weather'].unique())
print("Available Temp options:", df['Temp'].unique())
print()

weather_input = input("Enter Weather (Sunny/Overcast/Rain): ").strip().capitalize()
temp_input = input("Enter Temp (Hot/Mild/Cool): ").strip().capitalize()

if weather_input not in df['Weather'].unique() or temp_input not in df['Temp'].unique():
    print("\nInvalid input. Please enter values from the available options.")
else:
    print(f"\n=== STEP 3 & 4: Posterior Probability for input ({weather_input}, {temp_input}) ===")
    print(calculate_posterior(weather_input, temp_input), "\n")

    # 5. Compare with scikit-learn
    le_weather = LabelEncoder()
    le_temp = LabelEncoder()
    le_play = LabelEncoder()

    X = pd.DataFrame({
        'Weather': le_weather.fit_transform(df['Weather']),
        'Temp': le_temp.fit_transform(df['Temp'])
    })
    y = le_play.fit_transform(df['Play'])

    model = MultinomialNB()
    model.fit(X, y)

    sample = pd.DataFrame({
        'Weather': [le_weather.transform([weather_input])[0]],
        'Temp': [le_temp.transform([temp_input])[0]]
    })
    pred = model.predict(sample)[0]
    pred_prob = model.predict_proba(sample)[0]

    print("=== STEP 5: Validation with scikit-learn ===")
    print("Predicted class:", le_play.inverse_transform([pred])[0])
    print("Probability distribution:", 
          {cls: round(float(prob), 4) for cls, prob in zip(le_play.classes_, pred_prob)})
