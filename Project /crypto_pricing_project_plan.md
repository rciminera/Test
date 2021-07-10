## Crypto Pricing Project Plan

Steps to Creating a ML Model

The process of model -> fit -> predict/transform follows the same general steps across all of data science:

1. Decide on a model, and create a model instance.
2. Split into training and testing sets, and preprocess the data.
3. Train/fit the training data to the model. (Note that "train" and "fit" are used interchangeably in Python libraries as well as the data field.)
4. Use the model for predictions and transformations.


There are two main reasons to limit the number of neurons in a neural network model: overfitting and computation resources. Similar to other machine learning algorithms, neural networks are susceptible to overfitting where the model fits the training data too well. As a result of overfitting, the neural network will not generalize well and won't be able to classify new data correctly. Additionally, a neural network model with a large number of neurons requires equally large training dataset—training a large neural network requires more data, more epochs, and more time. Therefore, it is important that a neural network model has an appropriate number of neurons to match the size of the data, the complexity of the problem, and the amount of input neurons

A good rule of thumb for a basic neural network is to have two to three times the amount of neurons in the hidden layer as the number of inputs.

When a neural network model does not meet performance expectations, it is usually due to one of two causes: inadequate or inappropriate model design for a given dataset, or insufficient or ineffective training data. Although collecting more training/test data is almost always beneficial, it may be impossible due to budget or logistical limitations. Therefore, the most straightforward means of improving neural network performance is tweaking the model design and parameters.

There are a few means of optimizing a neural network:

- Check out your input dataset.
    
    Try plotting a variable using Pandas' Series.plot method to look for outliers that can help identify if a particular numerical variable is causing confusion in a model. Try leaving out a noisy variable from the rest of the training features and see if the model performs better.
- Add more neurons to a hidden layer.
    
    Previously, we explored how to optimize a neural network by adding neurons to the hidden layer. Adding neurons to a hidden layer has diminishing returns—more neurons means more data as well as a risk to overfitting the model.
- Add additional hidden layers.
- Use a different activation function for the hidden layers.


Activation Functions (in order of use)

1. Sigmoid: This function is identified by a characteristic S curve. It transforms the output to a range between 0 and 1.

    The sigmoid function values are normalized to a probability between 0 and 1, which is ideal for binary classification.

2. Tanh: This function is identified by a characteristic S curve; however, it transforms the output to a range between -1 and 1.

    The tanh function can be used for classification or regression, as it expands the range between -1 and 1.

3. ReLU: This function returns a value from 0 to infinity, so any negative input through the activation function is 0. It is the most used activation function in neural networks due to its simplifying output, but might not be appropriate for simpler models.

    The ReLU function is ideal for looking at positive nonlinear input data for classification or regression.

4. Leaky ReLU: This function is an alternative to another activation function, whereby negative input values will return very small negative values.

    The Leaky ReLU function is a good alternative for nonlinear input data with many negative inputs.


By default, the Keras Dense layer will implement the linear activation function, which means that the net sum value is not transformed. In other words:

𝛼(𝑥)=𝑥

The linear activation function limits the neural network model to only perform a linear regression. Therefore, the linear activation function is only appropriate for an output layer.

- Add additional epochs to the training regimen.

    Adding more epochs to the training parameters is not a perfect solution—if the model produces weight coefficients that are too effective, there is an increased risk of model overfitting. Therefore, models should be tested and evaluated each time the number of epochs are increased to reduce the risk of overfitting


Compare the performance of each model, asking the following questions:

- What is the accuracy of my model? Is it acceptable or does it need to be higher?
- How long did it take to train my model? How many minutes or hours? How many epochs?
- Does it look like my model is as complex as my input data? 

These reflective questions will help you identify what steps are needed to make your neural network (and other machine learning and statistical) models even better!

Data Preprocessing

One-hot encoding identifies all unique column values and splits the single categorical column into a series of columns, each containing information about a single unique categorical value

Although one-hot encoding is a very robust solution, it can be very memory-intensive.

The process of reducing the number of unique categorical values in a dataset is known as bucketing or binning. Bucketing data typically follows one of two approaches:

Collapse all of the infrequent and rare categorical values into a single "other" category.
Create generalized categorical values and reassign all data points to the new corresponding values.


Deep Learning Models

Deep neural networks function similarly to the basic neural network, with one major exception. The outputs of one hidden layer of neurons (that have been evaluated and transformed using an activation function) become the inputs to additional hidden layers of neurons

Although the numbers are constantly debated, many data engineers believe that even the most complex interactions can be characterized by as few as three hidden layers.

Logistic Regression v Deep Learning

A logistic regression model is a classification algorithm that can analyze continuous and categorical variables. Using a combination of input variables, logistic regression predicts the probability of the input data belonging to one of two groups. 

At the heart of the logistic regression model is the sigmoid curve, which is used to produce the probability (between 0 and 1) of the input data belonging to the first group. This sigmoid curve is the exact same curve used in the sigmoid activation function of a neural network. In fact, a basic neural network using the sigmoid activation function is effectively a logistic regression model: