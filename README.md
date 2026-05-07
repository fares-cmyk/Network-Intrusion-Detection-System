# Network Intrusion Detection System

A machine learning-based system that detects and classifies cyberattacks in network traffic data.

## Overview
This project processes raw network logs through a full data science pipeline — from data cleaning and statistical analysis to training and evaluating machine learning models that classify network traffic as normal or malicious.

## Features
- Data cleaning and preprocessing
- Statistical analysis with probability distributions and correlation heatmaps
- Detects 5 attack types: TCP-SYN, PortScan, Overflow, Blackhole, Diversion
- Custom from-scratch Naive Bayes classifier
- Scikit-learn models: GaussianNB, MultinomialNB, BernoulliNB
- Model evaluation using Accuracy, Precision, and Recall

## Requirements
```
pip install pandas numpy matplotlib scipy scikit-learn
```

## Dataset
The project requires two CSV files:
- `Train_data_updated.csv` — training data
- `test.csv` — test data

## Results
The system compares multiple Naive Bayes models and reports which best detects network attacks based on Accuracy, Precision, and Recall.
