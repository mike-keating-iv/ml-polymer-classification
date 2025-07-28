# Polymer Classification using Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mike-keating-iv/polymer-classification/blob/main/polymers_classifier.ipynb)

## Project Overview

This project develops machine learning models to classify polymer materials into their respective subclasses using mechanical and physical property data. The goal is to automate polymer identification for materials databases and R&D applications.

## Background

Polymeric materials are ubiquitous in our world and nearly every industry. Materials are selected based on physical, mechanical, thermal, and economic characteristics for any given application. Polymers as a class of materials have an extremely wide range of potential properties, from relatively soft low-density polyethylene (LDPE) to extremely strong and rigid engineering polymers such as Kevlar (Polyaramid) or PEEK.

In addition to broad classifications such as thermoplastic, elastomer, and thermosets, many polymers can be classified by their polymer subclass or specific chemical composition of the repeat unit. This is often how we think of plastics, with symbols like PP (polypropylene), PET (polyester terephthalate), and ABS (acrylonitrile-butadiene-styrene).

The goal of this project is to classify polymer materials into their respective polymer subclass using mechanical data as features. Examples of these mechanical properties include tensile strength at break, izod impact, flexural modulus, and elongation at break. Other physical properties may also be used as features – such as density, melt flow rate, etc.

## Problem Significance

A large component of medical device research and development is centered around materials selection and lifecycle management. Many companies run their own mechanical testing labs or outsource data collection to vendors to help make their decisions. One way to organize records in the schema of materials databases is by material class and subsequently subclass.

Often the polymer subclass (PP, etc) is known by the supplier, but in R&D this information can get miscommunicated or is not available (for development grades). Automating this subclass assignment would accelerate data upload while keeping the database schema organized.

## Data Source

Data was collected from the Plastics and Elastomers section of [UL Prospector](https://www.ulprospector.com/plastics/en), which is a database containing technical data for over 100,000 materials. Data was queried by polymer symbol (PP, PMMA, etc) and saved to separate CSV files.

## Features

The final dataset includes the following features for classification:

- **density**: The density of the polymer material (g/cm³)
- **tensile_modulus**: The stiffness of the material under tension (MPa)
- **flexural_modulus**: The stiffness of the material under bending (MPa)
- **flexural_strength**: The strength of the material under bending (MPa)
- **tensile_elongation_at_break**: The percentage elongation of the material at failure (%)
- **izod_notched_impact**: The impact resistance measured using the Izod impact test (kJ/m²)

## Polymer Classes

The model classifies materials into the following polymer subclasses:

- ABS (Acrylonitrile Butadiene Styrene)
- HDPE (High-Density Polyethylene)
- LDPE (Low-Density Polyethylene)
- Nylon (Polyamide)
- PC (Polycarbonate)
- PEEK (Polyetheretherketone)
- PMMA (Polymethyl Methacrylate)
- PP (Polypropylene)
- PPS (Polyphenylene Sulfide)
- PVC (Polyvinyl Chloride)

## Machine Learning Models

Four different machine learning algorithms were implemented and compared:

### 1. Logistic Regression

Logistic regression extends linear regression to classification problems by using the logistic function to model the probability of class membership. It's a linear classifier that works well when classes are linearly separable and provides interpretable coefficients for each feature.

### 2. K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a simple, non-parametric algorithm that classifies data points based on the majority class of their k nearest neighbors in the feature space. It's particularly effective for datasets where similar data points tend to have similar class labels, making it well-suited for materials classification where similar properties often indicate similar polymer types.

### 3. Random Forest

Random Forest is an ensemble method that combines multiple decision trees, each trained on a random subset of the data and features. This approach reduces overfitting and typically provides robust performance across various datasets, while also offering interpretable feature importance rankings.

### 4. XGBoost

XGBoost (Extreme Gradient Boosting) is a highly optimized gradient boosting framework that sequentially builds decision trees, with each new tree correcting errors from previous ones. It's known for achieving state-of-the-art performance in many machine learning competitions and real-world applications, particularly excelling with structured tabular data like our polymer properties dataset.

## Results

After comprehensive hyperparameter tuning and evaluation:

- **XGBoost** achieved the highest accuracy (~97%) and was selected as the final model
- **K-Nearest Neighbors** provided excellent performance with the fastest training time
- **Random Forest** showed robust performance with good feature importance insights
- **Logistic Regression** provided a solid baseline with interpretable results

XGBoost was chosen as the final model due to its superior accuracy, while KNN offers an excellent alternative for applications requiring faster training times.

## Project Structure

```
polymer-classification/
├── polymers_classifier.ipynb    # Main Jupyter notebook with analysis
├── README.md                    # Project documentation
├── data/                        # Raw polymer property data
│   ├── polymers_ABS.csv
│   ├── polymers_HDPE.csv
│   ├── polymers_LDPE.csv
│   ├── polymers_Nylon.csv
│   ├── polymers_PAI.csv
│   ├── polymers_PC.csv
│   ├── polymers_PEEK.csv
│   ├── polymers_PEI.csv
│   ├── polymers_PET.csv
│   ├── polymers_PMMA.csv
│   ├── polymers_Polyurethane.csv
│   ├── polymers_POM.csv
│   ├── polymers_PP.csv
│   ├── polymers_PPS.csv
│   ├── polymers_PTFE.csv
│   └── polymers_PVC.csv
└── presentation/
    └── ise538_final_presentation_keating.pptx
```

## Usage

### Running in Google Colab

Click the "Open in Colab" badge above to run the notebook directly in Google Colab.

### Running Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/mike-keating-iv/polymer-classification.git
   cd polymer-classification
   ```

2. Install required dependencies:

   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```

3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook polymers_classifier.ipynb
   ```

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- warnings (built-in)
- os (built-in)
- time (built-in)

## Model Validation

The final XGBoost model was validated using external data from a polycarbonate material datasheet, successfully predicting the correct polymer class (PC) with high confidence.

## Future Work

- Expand the dataset to include more polymer types
- Incorporate additional material properties (thermal, electrical)
- Develop a web application for real-time polymer classification
- Implement deep learning approaches for comparison

## Author

**Mike Keating**  
Student, EM 538  
North Carolina State University

## License

This project is for educational purposes as part of coursework at NC State University.
