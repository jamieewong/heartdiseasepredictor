# Heart Disease Predictor

This project implements a collection of machine learning algorithms and a neural network to predict heart disease. Models are trained on patient data, which includes features such as sex or chest pain type. This project was completed under a project mentor with four teammates through the University of California, Davis chapter of the Artificial Intelligence Student Collective (AISC).

Project Components:
1. Exploratory Analysis
   - Loaded data using Pandas.
   - Created visualizations using Seaborn and Matplotlib to explore the relationships amongst different features, as well as their relationship with heart disease itself.
2. Preprocessing
   - One-hot encoded categorical variables.
   - Performed min-max normalization to enhance model training performance. 
3. Model Selection and Evaluation
   - Trained 6 machine learning models
       - Logistic Regression: Implemented a manual implmentation of logistic regression using a sigmoid function. Used forward and backward propagation with custom           functions to update weight and bias.
       - Decision Tree
       - Random Forest
       - K-Nearest Neighbors (KNN)
       - Support Vector Machine (SVM)
       - Naive Bayes
   - Evaluated each model using the testing dataset, and created a visualization to easily compare accuracies of each model.
4. Neural Network Implementation
   - Trained a neural network using a Sequential model with multiple Dense layers, dropout, and batch normalization.
   - Applied an optimizer algorithm and used EarlyStopping to prevent overfitting.

Dependencies:
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
