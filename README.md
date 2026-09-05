# Air Quality Classification Using Machine Learning

## Project Overview

This project is a Machine Learning application that predicts air quality based on different air pollution parameters.

The system uses a **Random Forest Classifier** to classify air quality into five categories:

- Hazardous
- Poor
- Moderate
- Good
- Excellent

The project also includes data visualization, model evaluation, and interactive prediction.

## Problem Statement

Air pollution contains several pollutants that affect the quality of the air. Understanding these pollutant levels and classifying air quality manually can be difficult.

This project aims to develop a Machine Learning system that can classify air quality based on pollutant measurements such as PM2.5, PM10, NO2, SO2, CO, and O3.

## Objectives

1. To load and analyze air quality data.
2. To visualize air quality distributions.
3. To identify relationships between air pollution parameters.
4. To train a Machine Learning classification model.
5. To evaluate the performance of the model.
6. To predict air quality for new input values.

## Dataset

The project uses:

**Dataset:** `air_quality_dataset.csv`

### Input Parameters

| Parameter | Description |
|---|---|
| PM2.5 | Fine particulate matter |
| PM10 | Particulate matter |
| NO2 | Nitrogen dioxide |
| SO2 | Sulfur dioxide |
| CO | Carbon monoxide |
| O3 | Ozone |

### Air Quality Categories

| Value | Category |
|---:|---|
| 1 | Hazardous |
| 2 | Poor |
| 3 | Moderate |
| 4 | Good |
| 5 | Excellent |

## Machine Learning Algorithm

### Random Forest Classifier

This project uses the **Random Forest Classifier** for air quality classification.

The model uses:

- Number of estimators: **100**
- Random state: **42**
- Test size: **20%**
- Training size: **80%**

## Project Workflow


Air Quality Dataset
        ↓
Data Loading
        ↓
Data Visualization
        ↓
Train/Test Split
        ↓
Random Forest Training
        ↓
Model Evaluation
        ↓
Interactive Prediction
        ↓
Air Quality Classification
