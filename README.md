# Neural Network Visualizer with Streamlit

This Streamlit application provides an interactive visualization of a simple feedforward neural network. You can define the network architecture (number of layers and neurons per layer), train it on a basic synthetic dataset, and observe the decision boundary it learns.

## Features

* **Interactive Network Definition:** Easily specify the number of hidden layers and the number of neurons in each layer using Streamlit widgets.
* **Simple Dataset Generation:** Generates a basic two-class classification dataset for demonstration purposes.
* **Network Training:** Trains a simple neural network (using scikit-learn's `MLPClassifier`) on the generated data.
* **Decision Boundary Visualization:** Displays the decision boundary learned by the trained network on a 2D plot.
* **Layer Activation Visualization (Conceptual):** While a full dynamic activation visualization is complex for a simple README example, the UI includes placeholders and ideas for future expansion to show how data flows through the layers.
