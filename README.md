# 🌱 Neural Network Visual Learning Tool

A fully interactive educational tool for understanding and visualizing neural network training on plant growth data.

## Overview

This application provides an intuitive interface for building, training, and visualizing neural networks without requiring deep learning frameworks like TensorFlow or PyTorch. It implements a neural network from scratch using NumPy and visualizes the learning process in real-time.

The tool helps users understand:
- How neural networks learn from data
- The impact of architecture on learning capabilities
- Visualization of decision boundaries and predictions
- Feature interactions and importance analysis

## Features

### 🧠 Neural Network Implementation
- Custom implementation using NumPy (no deep learning frameworks required)
- Configurable architecture with 1-5 hidden layers
- Variable neurons per layer
- Tanh activation for hidden layers and sigmoid for output
- Backpropagation with mean squared error loss

### 📊 Interactive Training
- Step-by-step (epoch-by-epoch) training
- Automatic training for multiple epochs
- Real-time loss visualization
- R² performance metrics

### 🎮 User Interface
- Full control over network architecture
- Interactive prediction with sliders for inputs
- Neural network architecture visualization
- Live decision surface updates

### 📈 Visualization Features
- Network weight visualization
- 2D/3D decision boundaries
- Feature importance analysis
- Training history tracking
- Interactive 3D data explorer with custom viewpoints
- Plant growth formula explorer

## Dataset Generation

The application uses a synthetic dataset modeling realistic plant growth based on three key factors:

### Input Features
1. **Sunlight** (0-10 hours)
2. **Water** (0-5 liters)
3. **Temperature** (5-35°C)

### Target
- **Growth Rate** (0-1 normalized value)

### Plant Growth Formula

The growth rate is calculated using a realistic model incorporating:

```python
def realistic_plant_growth(sunlight, water, temperature=25):
    # Normalize inputs to 0-1 range
    sun_norm = sunlight / 10
    water_norm = water / 5
    temp_norm = (temperature - 5) / 30  # Normalize to 0-1 for range 5-35°C
    
    # Sunlight response (bell curve with optimal around 6-7 hours)
    sun_effect = 4 * sun_norm * (1 - sun_norm) ** 0.5
    
    # Water response (sigmoid-like with diminishing returns)
    water_effect = 1 - np.exp(-3 * water_norm)
    
    # Temperature effect (bell curve centered around 23°C)
    temp_effect = np.exp(-((temp_norm - 0.6) ** 2) / 0.1)
    
    # Interaction effects
    sun_water_interaction = 0.2 * sun_norm * water_norm
    temp_water_interaction = -0.1 * temp_norm * (1 - water_norm)
    sun_temp_interaction = -0.2 * (sun_norm - 0.6) * (temp_norm - 0.6)
    
    # Final growth rate calculation
    growth_rate = 0.3 + (
        0.4 * sun_effect + 
        0.3 * water_effect + 
        0.2 * temp_effect +
        sun_water_interaction +
        temp_water_interaction +
        sun_temp_interaction
    )
    
    # Constrain to 0-1 range
    return np.clip(growth_rate, 0, 1)
```

The formula accounts for:
- **Sunlight**: Bell-shaped response curve (optimal at 60-70% of max)
- **Water**: Diminishing returns curve
- **Temperature**: Bell curve with optimal around 23°C
- **Interactions**: 
  - Sun-Water: Positive interaction (more sun requires more water)
  - Temperature-Water: Negative at high temperatures (hot conditions need more water)
  - Sun-Temperature: Complex interaction (optimal temperature changes with sunlight)

### Mathematical Notations

The core components of the growth model can be expressed as:

1. **Sunlight Response**: $S(x) = 4x(1-x)^{0.5}$ where $x = \text{sunlight}/10$

2. **Water Response**: $W(x) = 1 - e^{-3x}$ where $x = \text{water}/5$

3. **Temperature Response**: $T(x) = e^{-\frac{(x-0.6)^2}{0.1}}$ where $x = \frac{\text{temperature}-5}{30}$

4. **Final Growth Rate**:
   $G = 0.3 + 0.4S(x_s) + 0.3W(x_w) + 0.2T(x_t) + I_{sw} + I_{tw} + I_{st}$

   Where:
   - $I_{sw} = 0.2 \cdot x_s \cdot x_w$ (Sun-Water interaction)
   - $I_{tw} = -0.1 \cdot x_t \cdot (1-x_w)$ (Temperature-Water interaction)
   - $I_{st} = -0.2 \cdot (x_s-0.6) \cdot (x_t-0.6)$ (Sun-Temperature interaction)

## Neural Network Implementation

The neural network in this application is implemented from scratch using NumPy:

### Forward Propagation

For each layer $l$:

$Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}$
$A^{[l]} = g^{[l]}(Z^{[l]})$

Where:
- $W^{[l]}$ is the weight matrix for layer $l$
- $A^{[l-1]}$ is the activation from the previous layer
- $b^{[l]}$ is the bias vector
- $g^{[l]}$ is the activation function (tanh for hidden layers, sigmoid for output)

### Backward Propagation

For output layer:
$dZ^{[L]} = A^{[L]} - Y$

For hidden layers:
$dZ^{[l]} = dA^{[l]} * g'^{[l]}(Z^{[l]})$

Where:
$dA^{[l]} = W^{[l+1]T} \cdot dZ^{[l+1]}$

### Weight Updates

$W^{[l]} = W^{[l]} - \alpha \cdot \frac{1}{m} \cdot dZ^{[l]} \cdot A^{[l-1]T}$
$b^{[l]} = b^{[l]} - \alpha \cdot \frac{1}{m} \cdot \sum dZ^{[l]}$

Where:
- $\alpha$ is the learning rate
- $m$ is the number of training examples

## Usage Examples

### Basic Usage

1. **Set Network Architecture**:
   - Select number of hidden layers (1-5)
   - Choose neurons per layer (2-50)
   - Set learning rate (0.001-0.1)

2. **Train the Network**:
   - Use "Step Train" for incremental learning
   - Use "Train Automatically" for multiple epochs
   - Monitor loss curve and R² score

3. **Make Predictions**:
   - Adjust input sliders for sunlight, water, temperature
   - See predicted vs. actual growth values

### Advanced Features

1. **Visualization Options**:
   - 3D Surface: View decision boundaries in 3D
   - 2D Slices: See cross-sections at different temperatures
   - Feature Importance: Analyze sensitivity to each input

2. **Interactive 3D Explorer**:
   - Rotate and view training data in 3D space
   - Adjust elevation and azimuth angles
   - Observe clusters and patterns in the data

3. **Formula Explorer**:
   - Visualize the true growth formula
   - Compare network predictions to the ground truth
   - Understand feature interactions

## Requirements

- Python 3.6+
- streamlit
- numpy
- pandas
- matplotlib
- seaborn
- networkx

## Installation & Running

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural-net-learner.git
cd neural-net-learner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Educational Value

This tool is designed for:
- Students learning neural networks fundamentals
- Teachers demonstrating backpropagation and gradient descent
- Anyone interested in visualizing how neural networks learn patterns
- Understanding feature interactions and model sensitivity

## License

This project is licensed under the terms of the [MIT License](LICENSE).

## Author

- [Vatsal Khanna](https://github.com/VatsalKhanna)

## Acknowledgements

- Streamlit for the interactive app framework
- Matplotlib, Seaborn and NetworkX for visualizations
