import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

# --- General Settings ---
st.set_page_config(page_title="Neural Net Learner", layout="wide")
sns.set_style("whitegrid")

# --- Realistic Plant Growth Model ---
def realistic_plant_growth(sunlight, water, temperature=25):
    """
    A more realistic plant growth model based on:
    - Sunlight follows a bell curve (plants need sufficient light but can get burned)
    - Water shows diminishing returns
    - Temperature has an optimal range
    - All factors interact with each other
    """
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

# --- Data Generation ---
@st.cache_data
def generate_data(n_samples=3000, seed=0):
    np.random.seed(seed)
    
    # Create grid points for stratified sampling
    grid_size = int(np.sqrt(n_samples * 5))  # Generate more points for stratification
    sunlight = np.linspace(0, 10, grid_size)
    water = np.linspace(0, 5, grid_size)
    temperatures = np.linspace(5, 35, max(5, grid_size // 8))  # Fewer temperature points
    
    # Create combinations
    combinations = []
    for s in sunlight:
        for w in water:
            for t in np.random.choice(temperatures, 3):  # Select random temperatures to reduce combinations
                combinations.append([s, w, t])
    
    # Sample from combinations if too many
    if len(combinations) > n_samples * 5:
        sample_indices = np.random.choice(len(combinations), n_samples * 5, replace=False)
        combinations = [combinations[i] for i in sample_indices]
    
    # Create dataframe and calculate growth
    df_samples = pd.DataFrame(combinations, columns=['sunlight', 'water', 'temperature'])
    df_samples['growth'] = df_samples.apply(
        lambda row: realistic_plant_growth(row['sunlight'], row['water'], row['temperature']), 
        axis=1
    )
    
    # Normalize inputs
    df_samples['sunlight_norm'] = df_samples['sunlight'] / 10
    df_samples['water_norm'] = df_samples['water'] / 5
    df_samples['temp_norm'] = (df_samples['temperature'] - 5) / 30
    
    # Stratify samples
    bins = np.linspace(0, 1, 11)  # 10 bins (0.0–0.1, ..., 0.9–1.0)
    df_samples['growth_bin'] = pd.cut(df_samples['growth'], bins)
    
    # Sample equally from each bin
    balanced_dfs = []
    samples_per_bin = n_samples // 10
    
    for bin_range in df_samples['growth_bin'].unique():
        bin_df = df_samples[df_samples['growth_bin'] == bin_range]
        if len(bin_df) >= samples_per_bin:
            sampled = bin_df.sample(samples_per_bin, random_state=seed)
            balanced_dfs.append(sampled)
        else:
            balanced_dfs.append(bin_df)  # Take all if not enough samples
    
    df_final = pd.concat(balanced_dfs).sample(frac=1, random_state=seed).drop(columns=['growth_bin'])
    
    return df_final.reset_index(drop=True)

# --- Neural Network Class with Multiple Hidden Layers ---
class EnhancedNeuralNet:
    def __init__(self, input_size, hidden_layers, output_size, lr=0.01, seed=42):
        """
        Initialize a neural network with multiple hidden layers
        
        Parameters:
        - input_size: number of input features
        - hidden_layers: list of sizes for each hidden layer
        - output_size: number of output units
        - lr: learning rate
        - seed: random seed for weight initialization
        """
        np.random.seed(seed)
        self.lr = lr
        self.n_layers = len(hidden_layers) + 1  # hidden layers + output layer
        self.hidden_activations = ["tanh"] * len(hidden_layers)
        self.output_activation = "sigmoid"
        
        # Initialize weights and biases for all layers
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_layers[0]) * np.sqrt(2. / input_size))
        self.biases.append(np.zeros((1, hidden_layers[0])))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.weights.append(np.random.randn(hidden_layers[i], hidden_layers[i+1]) * 
                               np.sqrt(2. / hidden_layers[i]))
            self.biases.append(np.zeros((1, hidden_layers[i+1])))
        
        # Last hidden layer to output
        self.weights.append(np.random.randn(hidden_layers[-1], output_size) * 
                           np.sqrt(2. / hidden_layers[-1]))
        self.biases.append(np.zeros((1, output_size)))

    def activation(self, z, func="tanh"):
        if func == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif func == "relu":
            return np.maximum(0, z)
        elif func == "tanh":
            return np.tanh(z)
        else:
            return z  # linear

    def activation_derivative(self, output, func="tanh"):
        if func == "sigmoid":
            return output * (1 - output)
        elif func == "relu":
            return (output > 0).astype(float)
        elif func == "tanh":
            return 1 - output ** 2
        else:
            return 1  # linear

    def forward(self, X):
        self.layer_inputs = []  # Store inputs to each layer (needed for backprop)
        self.layer_outputs = []  # Store outputs of each layer
        
        # Input layer (no activation)
        current_input = X
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            self.layer_inputs.append(current_input)
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            a = self.activation(z, func=self.hidden_activations[i])
            self.layer_outputs.append(a)
            current_input = a
        
        # Output layer
        self.layer_inputs.append(current_input)
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        output = self.activation(z_output, func=self.output_activation)
        self.layer_outputs.append(output)
        
        return output

    def backward(self, X, y):
        m = len(y)
        output = self.layer_outputs[-1]
        
        # Output layer error
        d_layer = (output - y) * self.activation_derivative(output, func=self.output_activation)
        deltas = [d_layer]
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 1, 0, -1):
            d_layer = np.dot(deltas[0], self.weights[i].T) * self.activation_derivative(
                self.layer_outputs[i-1], func=self.hidden_activations[i-1])
            deltas.insert(0, d_layer)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            layer_input = X if i == 0 else self.layer_outputs[i-1]
            self.weights[i] -= self.lr * np.dot(layer_input.T, deltas[i]) / m
            self.biases[i] -= self.lr * np.sum(deltas[i], axis=0, keepdims=True) / m

    def train_one_epoch(self, X, y):
        self.forward(X)
        self.backward(X, y)
        loss = np.mean((self.layer_outputs[-1] - y) ** 2)
        return loss

    def predict(self, X):
        return self.forward(X)

# --- Visualization Function ---
def draw_network(nn, ax=None):
    # Count the total number of nodes
    input_size = nn.weights[0].shape[0]
    hidden_layers = [w.shape[1] for w in nn.weights[:-1]]
    output_size = nn.weights[-1].shape[1]
    total_layers = len(hidden_layers) + 2  # +2 for input and output layers
    
    G = nx.DiGraph()
    pos = {}
    layer_nodes = {i: [] for i in range(total_layers)}
    
    # Add input nodes
    for i in range(input_size):
        node = f"I{i}"
        G.add_node(node)
        pos[node] = (0, -i * 2)
        layer_nodes[0].append(node)
    
    # Add hidden layer nodes
    for layer_idx, layer_size in enumerate(hidden_layers):
        for i in range(layer_size):
            node = f"H{layer_idx}_{i}"
            G.add_node(node)
            pos[node] = (layer_idx + 1, -i * 2)
            layer_nodes[layer_idx + 1].append(node)
    
    # Add output nodes
    for i in range(output_size):
        node = f"O{i}"
        G.add_node(node)
        pos[node] = (total_layers - 1, -i * 2)
        layer_nodes[total_layers - 1].append(node)
    
    # Add edges between input and first hidden layer
    for i in range(input_size):
        for j in range(hidden_layers[0]):
            G.add_edge(f"I{i}", f"H0_{j}", weight=nn.weights[0][i, j])
    
    # Add edges between hidden layers
    for layer_idx in range(len(hidden_layers) - 1):
        for i in range(hidden_layers[layer_idx]):
            for j in range(hidden_layers[layer_idx + 1]):
                G.add_edge(f"H{layer_idx}_{i}", f"H{layer_idx+1}_{j}", 
                           weight=nn.weights[layer_idx + 1][i, j])
    
    # Add edges between last hidden layer and output
    for i in range(hidden_layers[-1]):
        for j in range(output_size):
            G.add_edge(f"H{len(hidden_layers)-1}_{i}", f"O{j}", 
                       weight=nn.weights[-1][i, j])
    
    # Edge weights
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    norm = plt.Normalize(-1, 1)
    cmap = plt.cm.seismic
    
    # Draw network
    nx.draw(G, pos, ax=ax, with_labels=True, edge_color=edge_colors, edge_cmap=cmap,
            node_color='lightblue', node_size=1200, width=2, arrows=True,
            font_size=8)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
    ax.set_title("Neural Network Architecture with Weights")

# --- App Interface ---
st.title("🌱 Enhanced Neural Network Visual Learning Tool")
st.markdown("A fully interactive journey into training neural networks with customizable architectures on realistic plant growth data.")

with st.sidebar:
    st.header("🧠 Network Config")
    
    # Network architecture settings
    st.subheader("Architecture")
    num_hidden_layers = st.slider("Number of Hidden Layers", 1, 5, 1)
    
    # Create layer size inputs
    hidden_layers = []
    for i in range(num_hidden_layers):
        layer_size = st.slider(f"Neurons in Hidden Layer {i+1}", 2, 50, 5)
        hidden_layers.append(layer_size)
    
    # Input features selection
    st.subheader("Input Features")
    use_temperature = st.checkbox("Include Temperature as Input", True)
    input_size = 3 if use_temperature else 2
    
    # Training parameters
    st.subheader("Training Parameters")
    learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
    epochs = st.slider("Auto-Train Epochs", 100, 5000, 1000, step=100)
    seed = st.number_input("Random Seed", value=42)
    
    # 3D Visualization settings
    st.subheader("Visualization Settings")
    if use_temperature:
        vis_mode = st.radio("Decision Surface Visualization", 
                         ["3D Surface", "2D Slices", "Feature Importance"])
    else:
        vis_mode = "2D Surface"
    
    if st.button("🔄 Initialize Network"):
        st.session_state.nn = EnhancedNeuralNet(input_size, hidden_layers, 1, lr=learning_rate, seed=seed)
        st.session_state.history = []
        st.session_state.epoch = 0
        st.session_state.trained = False
        st.session_state.use_temperature = use_temperature
        st.session_state.vis_mode = vis_mode
        st.success(f"Network initialized with {len(hidden_layers)} hidden layers!")

# --- Initialization ---
if "nn" not in st.session_state:
    # Default initialization with a single hidden layer
    st.session_state.nn = EnhancedNeuralNet(3, [5], 1)
    st.session_state.history = []
    st.session_state.epoch = 0
    st.session_state.trained = False
    st.session_state.use_temperature = True
    st.session_state.vis_mode = "3D Surface"

# --- Data ---
data = generate_data()

# Select features based on user choice
if st.session_state.use_temperature:
    X = data[["sunlight_norm", "water_norm", "temp_norm"]].values
else:
    X = data[["sunlight_norm", "water_norm"]].values

y = data["growth"].values.reshape(-1, 1)

# --- Training ---
st.subheader("📊 Train the Network")

col_train, col_vis = st.columns(2)

with col_train:
    if st.button("👣 Step Train (1 Epoch)"):
        loss = st.session_state.nn.train_one_epoch(X, y)
        st.session_state.history.append(loss)
        st.session_state.epoch += 1
        st.session_state.trained = True
        st.rerun()
    
    if st.button("🚀 Train Automatically"):
        with st.spinner("Training in progress..."):
            for _ in range(epochs):
                loss = st.session_state.nn.train_one_epoch(X, y)
                st.session_state.history.append(loss)
                st.session_state.epoch += 1
            st.session_state.trained = True
        st.success(f"Auto Training Complete! ✅ Trained for {epochs} more epochs.")
        st.rerun()

    if st.session_state.history:
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(st.session_state.history, label="MSE Loss", color="purple")
        ax_loss.set_title("Loss over Epochs")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        st.pyplot(fig_loss)

    if st.session_state.trained:
        preds = st.session_state.nn.predict(X)
        r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
        st.metric("Final R² Score", f"{r2:.4f}")
        st.metric("Current Epoch", st.session_state.epoch)

with col_vis:
    if len(st.session_state.nn.weights) <= 3:  # Only draw if network is not too complex
        fig_net, ax_net = plt.subplots(figsize=(8, 8))
        draw_network(st.session_state.nn, ax=ax_net)
        st.pyplot(fig_net)
    else:
        st.info("Network visualization is simplified for complex architectures with many layers.")
        # Create a simpler visualization for complex networks
        fig_net, ax_net = plt.subplots(figsize=(8, 4))
        
        # Display a simplified diagram
        layer_sizes = [st.session_state.nn.weights[0].shape[0]] + \
                      [w.shape[1] for w in st.session_state.nn.weights]
        
        ax_net.axis('off')
        max_neurons = max(layer_sizes)
        
        for i, size in enumerate(layer_sizes):
            for j in range(size):
                circle = plt.Circle((i, j * (max_neurons / max(1, size))), 0.2, fill=True, color='lightblue')
                ax_net.add_patch(circle)
            
            # Add connections if not the last layer
            if i < len(layer_sizes) - 1:
                for j in range(size):
                    for k in range(layer_sizes[i+1]):
                        ax_net.plot([i, i+1], 
                                  [j * (max_neurons / max(1, size)), 
                                   k * (max_neurons / max(1, layer_sizes[i+1]))], 
                                  'gray', alpha=0.3)
        
        ax_net.set_xlim(-0.5, len(layer_sizes) - 0.5)
        ax_net.set_ylim(-0.5, max_neurons - 0.5)
        ax_net.set_title("Neural Network Architecture (Simplified View)")
        st.pyplot(fig_net)

# --- Prediction Interface ---
st.subheader("🔮 Make a Prediction")
sun = st.slider("Sunlight (hours)", 0.0, 10.0, 5.0)
water = st.slider("Water (liters)", 0.0, 5.0, 2.5)

# Add temperature input if enabled
if st.session_state.use_temperature:
    temperature = st.slider("Temperature (°C)", 5.0, 35.0, 25.0)
    input_val = np.array([[sun / 10, water / 5, (temperature - 5) / 30]])
else:
    temperature = 25  # Default
    input_val = np.array([[sun / 10, water / 5]])

if st.session_state.trained:
    pred = st.session_state.nn.predict(input_val)[0][0]
    # Get the actual value from the realistic model
    true_val = realistic_plant_growth(sun, water, temperature)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🌿 Predicted Growth: **{pred:.4f}**")
    with col2:
        st.info(f"🔬 True Growth (Formula): **{true_val:.4f}**")
else:
    st.warning("Train the network to see predictions.")

# --- Additional Visualizations ---
if st.session_state.trained:
    st.subheader("📈 Additional Graphs for Deeper Insight")

    pred_all = st.session_state.nn.predict(X)
    
    # Calculate R² score with current model
    r2 = 1 - np.sum((y - pred_all) ** 2) / np.sum((y - y.mean()) ** 2)
    
    # For R² history, use a simulated improvement curve based on loss history
    if st.session_state.history:
        losses = np.array(st.session_state.history)
        # Convert loss history to approximate R² history
        r2_start = 0.1  # Starting R² value
        loss_improvement = (losses[0] - losses) / (losses[0] + 1e-10)  # Normalized improvement
        r2_history = r2_start + (r2 - r2_start) * (loss_improvement ** 0.5)  # Square root for non-linear improvement
        # Ensure R² is within valid range
        r2_history = np.clip(r2_history, 0, 1)
        
        # Sample to reduce points if history is very long
        sample_rate = max(1, len(r2_history) // 100)
        sampled_epochs = list(range(0, len(r2_history), sample_rate))
        sampled_r2 = r2_history[sampled_epochs]
    else:
        sampled_epochs = []
        sampled_r2 = []
    
    col1, col2 = st.columns(2)

    # 1. Predicted vs Actual (for all training samples)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.scatter(y, pred_all, alpha=0.6, color="teal")
        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax1.set_xlabel("Actual Growth")
        ax1.set_ylabel("Predicted Growth")
        ax1.set_title("Predicted vs Actual (Training Set)")
        st.pyplot(fig1)

    # 2. Predicted Value vs Real Formula
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.bar(["Model Prediction", "True (Formula)"], [pred, true_val], color=["orange", "green"])
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Growth Value")
        ax2.set_title("User Input Prediction vs True Value")
        st.pyplot(fig2)

    # 3. R² Accuracy Over Time (if available)
    if len(sampled_epochs) > 0:
        st.markdown("### 📊 R² Score Progress")
        fig3, ax3 = plt.subplots()
        ax3.plot(sampled_epochs, sampled_r2, color="darkred", linewidth=2)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Estimated R² Score")
        ax3.set_title("Approximate Accuracy (R²) Over Training")
        ax3.set_ylim(0, 1)
        st.pyplot(fig3)

    # --- Decision Visualization Section ---
    st.markdown("### 🌐 Neural Network Decision Visualization")
    
    # Case 1: 2D visualization when only 2 inputs are used
    if not st.session_state.use_temperature:
        try:
            # Create a mesh grid for visualization
            sun_range = np.linspace(0, 1, 40)  # Reduced grid size for better performance
            water_range = np.linspace(0, 1, 40)
            sun_grid, water_grid = np.meshgrid(sun_range, water_range)
            
            # Reshape for prediction
            X_mesh = np.c_[sun_grid.ravel(), water_grid.ravel()]
            
            # Make predictions
            Z = st.session_state.nn.predict(X_mesh).reshape(sun_grid.shape)
            
            # Plot 2D contour
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            contour = ax4.contourf(sun_grid, water_grid, Z, cmap='viridis', alpha=0.8, levels=15)
            plt.colorbar(contour, ax=ax4)
            
            # Mark current input point
            ax4.scatter([input_val[0, 0]], [input_val[0, 1]], c='red', marker='*', s=200, 
                       edgecolor='white', linewidth=1.5, label='Current Input')
            
            ax4.set_xlabel('Normalized Sunlight')
            ax4.set_ylabel('Normalized Water')
            ax4.set_title('Neural Network Decision Surface (2D)')
            ax4.legend()
            
            st.pyplot(fig4)
        except Exception as e:
            st.error(f"Could not generate 2D decision surface: {str(e)}")
            st.info("This can happen with complex network architectures. Try simplifying the network.")
    # Case 2: 3D visualization when all 3 inputs are used
    else:
        if st.session_state.vis_mode == "3D Surface":
            try:
                # Create grid points for 3D visualization (keeping third dimension constant)
                resolution = 25  # Lower resolution for better performance
                sun_range = np.linspace(0, 1, resolution)
                water_range = np.linspace(0, 1, resolution)
                sun_grid, water_grid = np.meshgrid(sun_range, water_range)
                
                # Use current temperature value for the visualization
                temp_val = (temperature - 5) / 30
                
                # Prepare inputs
                grid_shape = sun_grid.shape
                temp_grid = np.ones_like(sun_grid) * temp_val
                
                X_mesh = np.column_stack((sun_grid.ravel(), water_grid.ravel(), temp_grid.ravel()))
                
                # Make predictions
                Z = st.session_state.nn.predict(X_mesh).reshape(grid_shape)
                
                # Create 3D surface plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot the surface
                surf = ax.plot_surface(sun_grid, water_grid, Z, cmap='viridis', alpha=0.8, 
                                     rstride=1, cstride=1, edgecolor='none', antialiased=True)
                
                # Mark current prediction point
                ax.scatter([input_val[0, 0]], [input_val[0, 1]], [pred], c='red', marker='o', s=100, 
                         edgecolor='white', linewidth=1.5, label='Prediction')
                
                # Add color bar
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                
                # Customize the plot
                ax.set_xlabel('Normalized Sunlight')
                ax.set_ylabel('Normalized Water')
                ax.set_zlabel('Predicted Growth')
                ax.set_title(f'3D Decision Surface (Temperature = {temperature}°C)')
                
                # Adjust view angle
                ax.view_init(30, 135)
                
                st.pyplot(fig)
                
                # Show information about visualization
                st.info("This 3D visualization shows the relationship between sunlight, water, and predicted growth rate while holding temperature constant at your selected value. The red marker shows your current input's prediction.")
                
            except Exception as e:
                st.error(f"Could not generate 3D surface: {str(e)}")
                st.info("Try simplifying the network or reducing the resolution.")
        
        elif st.session_state.vis_mode == "2D Slices":
            try:
                # Create slices at different temperature values
                temps = [10, 20, 25, 30]
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
                
                # Create grid points
                resolution = 30
                sun_range = np.linspace(0, 1, resolution)
                water_range = np.linspace(0, 1, resolution)
                sun_grid, water_grid = np.meshgrid(sun_range, water_range)
                
                for i, temp in enumerate(temps):
                    temp_val = (temp - 5) / 30  # Normalize
                    
                    # Prepare inputs
                    grid_shape = sun_grid.shape
                    temp_grid = np.ones_like(sun_grid) * temp_val
                    
                    X_mesh = np.column_stack((sun_grid.ravel(), water_grid.ravel(), temp_grid.ravel()))
                    
                    # Make predictions
                    Z = st.session_state.nn.predict(X_mesh).reshape(grid_shape)
                    
                    # Plot slice
                    contour = axes[i].contourf(sun_grid, water_grid, Z, cmap='viridis', levels=15)
                    
                    # Mark current point if temperature is close
                    if abs(temp - temperature) < 5:
                        axes[i].scatter([input_val[0, 0]], [input_val[0, 1]], c='red', marker='*', s=150,
                                      label='Current Input')
                    
                    axes[i].set_xlabel('Normalized Sunlight')
                    axes[i].set_ylabel('Normalized Water')
                    axes[i].set_title(f'Temperature = {temp}°C')
                
                # Add colorbar
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
                fig.colorbar(contour, cax=cbar_ax, label='Predicted Growth')
                
                fig.suptitle('Neural Network Decision Surface at Different Temperatures')
                plt.tight_layout(rect=[0, 0, 0.85, 0.95])
                
                st.pyplot(fig)
                
                st.info("These 2D slices show how the relationship between sunlight, water, and growth changes across different temperature values. Notice how the optimal growing conditions shift!")
                
            except Exception as e:
                st.error(f"Could not generate 2D slices: {str(e)}")
        
        elif st.session_state.vis_mode == "Feature Importance":
            try:
                # Simple sensitivity analysis
                base_input = np.array([[sun/10, water/5, (temperature-5)/30]])
                base_pred = st.session_state.nn.predict(base_input)[0][0]
                
                # Test sensitivity to each input with finer increments
                delta = 0.05
                n_points = 20
                
                # Create variable ranges
                sun_range = np.linspace(max(0, base_input[0, 0]-0.3), min(1, base_input[0, 0]+0.3), n_points)
                water_range = np.linspace(max(0, base_input[0, 1]-0.3), min(1, base_input[0, 1]+0.3), n_points)
                temp_range = np.linspace(max(0, base_input[0, 2]-0.3), min(1, base_input[0, 2]+0.3), n_points)
                
                # Get predictions varying only one feature at a time
                sun_preds = []
                water_preds = []
                temp_preds = []
                
                for s in sun_range:
                    test_input = base_input.copy()
                    test_input[0, 0] = s
                    sun_preds.append(st.session_state.nn.predict(test_input)[0][0])
                
                for w in water_range:
                    test_input = base_input.copy()
                    test_input[0, 1] = w
                    water_preds.append(st.session_state.nn.predict(test_input)[0][0])
                
                for t in temp_range:
                    test_input = base_input.copy()
                    test_input[0, 2] = t
                    temp_preds.append(st.session_state.nn.predict(test_input)[0][0])
                
                # Plot sensitivity curves
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.plot(sun_range*10, sun_preds, 'r-', linewidth=2, label='Sunlight (hours)')
                ax.plot(water_range*5, water_preds, 'b-', linewidth=2, label='Water (liters)')
                ax.plot(temp_range*30+5, temp_preds, 'g-', linewidth=2, label='Temperature (°C)')
                
                # Add vertical lines at current values
                ax.axvline(sun, color='r', linestyle='--', alpha=0.5)
                ax.axvline(water, color='b', linestyle='--', alpha=0.5)
                ax.axvline(temperature, color='g', linestyle='--', alpha=0.5)
                
                # Add horizontal line at current prediction
                ax.axhline(pred, color='k', linestyle='--', alpha=0.3)
                
                ax.set_xlabel('Input Value (in original units)')
                ax.set_ylabel('Predicted Growth')
                ax.set_title('Feature Sensitivity Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Calculate sensitivity scores (using standard deviation of predictions)
                sun_sensitivity = np.std(sun_preds)
                water_sensitivity = np.std(water_preds)
                temp_sensitivity = np.std(temp_preds)
                
                # Normalize to sum to 100%
                total_sensitivity = sun_sensitivity + water_sensitivity + temp_sensitivity
                sun_importance = sun_sensitivity / total_sensitivity * 100
                water_importance = water_sensitivity / total_sensitivity * 100
                temp_importance = temp_sensitivity / total_sensitivity * 100
                
                # Create bar chart of relative importances
                fig5, ax5 = plt.subplots(figsize=(8, 5))
                features = ['Sunlight', 'Water', 'Temperature']
                importances = [sun_importance, water_importance, temp_importance]
                colors = ['#FF9966', '#66B2FF', '#99CC99']
                
                bars = ax5.bar(features, importances, color=colors)
                ax5.set_ylabel('Relative Importance (%)')
                ax5.set_title('Feature Importance Analysis')
                
                # Add value labels on top of each bar
                for bar in bars:
                    height = bar.get_height()
                    ax5.annotate(f'{height:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
                
                st.pyplot(fig5)
                
                st.info("The sensitivity analysis shows how changing each input affects the predicted growth. The steeper the curve, the more sensitive the model is to that feature at your current operating point.")
                
            except Exception as e:
                st.error(f"Could not perform sensitivity analysis: {str(e)}")

    # --- Interactive 3D Explorer (For 3 features only) ---
    if st.session_state.use_temperature and st.session_state.trained:
        st.markdown("### 🔍 Interactive 3D Data Explorer")
        
        try:
            # Create a 3D scatter plot of training data
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Sample points to avoid overcrowding
            sample_size = min(500, len(X))
            sample_idx = np.random.choice(range(len(X)), sample_size, replace=False)
            
            # Create scatter plot with color based on growth
            scatter = ax.scatter(X[sample_idx, 0], X[sample_idx, 1], X[sample_idx, 2], 
                               c=y[sample_idx].flatten(), cmap='viridis', 
                               s=30, alpha=0.7)
            
            # Add current prediction point
            ax.scatter([input_val[0, 0]], [input_val[0, 1]], [input_val[0, 2]], 
                     color='red', s=200, marker='*', edgecolor='white',
                     label='Current Input')
            
            # Add color bar
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
            cbar.set_label('Growth Rate')
            
            # Customize the plot
            ax.set_xlabel('Normalized Sunlight')
            ax.set_ylabel('Normalized Water')
            ax.set_zlabel('Normalized Temperature')
            ax.set_title('Training Data in 3D Feature Space')
            ax.legend()
            
            # Add grid for better depth perception
            ax.grid(True)
            
            # Set initial viewpoint
            ax.view_init(30, 120)
            
            st.pyplot(fig)
            
            # Add rotation controls
            st.write("Change the 3D view angle:")
            col1, col2 = st.columns(2)
            with col1:
                elev = st.slider("Elevation", -90, 90, 30)
            with col2:
                azim = st.slider("Azimuth", 0, 360, 120)
            
            # Show rotated view
            fig_rot = plt.figure(figsize=(10, 8))
            ax_rot = fig_rot.add_subplot(111, projection='3d')
            
            # Recreate plot with new viewpoint
            scatter = ax_rot.scatter(X[sample_idx, 0], X[sample_idx, 1], X[sample_idx, 2], 
                                   c=y[sample_idx].flatten(), cmap='viridis', 
                                   s=30, alpha=0.7)
            
            ax_rot.scatter([input_val[0, 0]], [input_val[0, 1]], [input_val[0, 2]], 
                         color='red', s=200, marker='*', edgecolor='white',
                         label='Current Input')
            
            cbar = fig_rot.colorbar(scatter, ax=ax_rot, shrink=0.5, aspect=5)
            cbar.set_label('Growth Rate')
            
            ax_rot.set_xlabel('Normalized Sunlight')
            ax_rot.set_ylabel('Normalized Water')
            ax_rot.set_zlabel('Normalized Temperature')
            ax_rot.set_title('Training Data with Custom View Angle')
            ax_rot.legend()
            ax_rot.grid(True)
            
            # Set custom viewpoint
            ax_rot.view_init(elev, azim)
            
            st.pyplot(fig_rot)
            
        except Exception as e:
            st.error(f"Could not generate interactive 3D plot: {str(e)}")

# --- Add Formula Explorer Section ---
st.markdown("---")
st.subheader("🧪 Plant Growth Formula Explorer")

# Create a tool to visualize the actual formula
with st.expander("Explore the True Growth Formula"):
    st.write("""
    This section allows you to visualize the true plant growth formula that the neural network is trying to learn.
    You can see how each parameter affects growth independently and their interactions.
    """)
    
    # Choose which relationship to plot
    plot_type = st.radio("Select Relationship to Visualize:", 
                       ["Sunlight Effect", "Water Effect", "Temperature Effect", 
                        "Sun-Water Interaction", "Sun-Temperature Interaction",
                        "Water-Temperature Interaction"])
    
    # Create basic plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if plot_type == "Sunlight Effect":
        # Show sunlight effect while keeping other variables constant
        sun_range = np.linspace(0, 10, 100)
        growth_vals = [realistic_plant_growth(s, water, temperature) for s in sun_range]
        
        ax.plot(sun_range, growth_vals, linewidth=3, color='orange')
        ax.axvline(sun, color='r', linestyle='--')
        ax.set_xlabel("Sunlight (hours)")
        ax.set_ylabel("Growth Rate")
        ax.set_title("Effect of Sunlight on Plant Growth")
        
    elif plot_type == "Water Effect":
        # Show water effect
        water_range = np.linspace(0, 5, 100)
        growth_vals = [realistic_plant_growth(sun, w, temperature) for w in water_range]
        
        ax.plot(water_range, growth_vals, linewidth=3, color='blue')
        ax.axvline(water, color='b', linestyle='--')
        ax.set_xlabel("Water (liters)")
        ax.set_ylabel("Growth Rate")
        ax.set_title("Effect of Water on Plant Growth")
        
    elif plot_type == "Temperature Effect":
        # Show temperature effect
        temp_range = np.linspace(5, 35, 100)
        growth_vals = [realistic_plant_growth(sun, water, t) for t in temp_range]
        
        ax.plot(temp_range, growth_vals, linewidth=3, color='green')
        ax.axvline(temperature, color='g', linestyle='--')
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Growth Rate")
        ax.set_title("Effect of Temperature on Plant Growth")
        
    elif plot_type == "Sun-Water Interaction":
        # Create 2D heatmap of sun-water interaction
        sun_range = np.linspace(0, 10, 50)
        water_range = np.linspace(0, 5, 50)
        sun_grid, water_grid = np.meshgrid(sun_range, water_range)
        
        growth_grid = np.zeros_like(sun_grid)
        for i in range(len(water_range)):
            for j in range(len(sun_range)):
                growth_grid[i, j] = realistic_plant_growth(sun_grid[i, j], water_grid[i, j], temperature)
        
        contour = ax.contourf(sun_grid, water_grid, growth_grid, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax)
        ax.scatter([sun], [water], color='r', marker='*', s=200, edgecolor='white')
        
        ax.set_xlabel("Sunlight (hours)")
        ax.set_ylabel("Water (liters)")
        ax.set_title(f"Sun-Water Interaction (Temperature = {temperature}°C)")
        
    elif plot_type == "Sun-Temperature Interaction":
        # Create 2D heatmap of sun-temperature interaction
        sun_range = np.linspace(0, 10, 50)
        temp_range = np.linspace(5, 35, 50)
        sun_grid, temp_grid = np.meshgrid(sun_range, temp_range)
        
        growth_grid = np.zeros_like(sun_grid)
        for i in range(len(temp_range)):
            for j in range(len(sun_range)):
                growth_grid[i, j] = realistic_plant_growth(sun_grid[i, j], water, temp_grid[i, j])
        
        contour = ax.contourf(sun_grid, temp_grid, growth_grid, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax)
        ax.scatter([sun], [temperature], color='r', marker='*', s=200, edgecolor='white')
        
        ax.set_xlabel("Sunlight (hours)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"Sun-Temperature Interaction (Water = {water} liters)")
        
    elif plot_type == "Water-Temperature Interaction":
        # Create 2D heatmap of water-temperature interaction
        water_range = np.linspace(0, 5, 50)
        temp_range = np.linspace(5, 35, 50)
        water_grid, temp_grid = np.meshgrid(water_range, temp_range)
        
        growth_grid = np.zeros_like(water_grid)
        for i in range(len(temp_range)):
            for j in range(len(water_range)):
                growth_grid[i, j] = realistic_plant_growth(sun, water_grid[i, j], temp_grid[i, j])
        
        contour = ax.contourf(water_grid, temp_grid, growth_grid, levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax)
        ax.scatter([water], [temperature], color='r', marker='*', s=200, edgecolor='white')
        
        ax.set_xlabel("Water (liters)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"Water-Temperature Interaction (Sunlight = {sun} hours)")
    
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Show formula explanation
    st.markdown("""
    #### 📚 About the Plant Growth Formula
    
    The realistic plant growth model considers:
    
    1. **Sunlight response**: Bell-shaped curve with optimal around 6-7 hours (gets worse if too low or too high)
    2. **Water response**: Diminishing returns (more water helps, but with decreasing benefit)
    3. **Temperature effect**: Bell curve with optimal around 23°C
    4. **Interaction effects**:
       - Sun-Water: Positive interaction (more sun requires more water)
       - Temperature-Water: Negative at high temperatures (hot conditions need more water)
       - Sun-Temperature: Complex interaction (optimal temp changes with sunlight)
    
    The neural network tries to learn these relationships and their interactions from the data.
    """)

st.markdown("---")
st.markdown("Made with ❤️ for students learning about neural nets. Fine-tune, visualize, and play around to understand how training affects the network.")
                