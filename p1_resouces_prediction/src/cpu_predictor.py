import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Generate Synthetic Data (Network Connections vs CPU Usage)
np.random.seed(42)
X_connections = np.random.randint(100, 1000, (100, 1))  # Active connections
# Linear relationship: CPU = 0.08*connections + noise
y_cpu_load = 0.08 * X_connections + np.random.normal(0, 5, (100, 1)) 

# 2. Initialize and Train the Model
model = LinearRegression()
model.fit(X_connections, y_cpu_load)

# 3. Make a Prediction
test_connection = np.array([[1200]]) # Predict for 1200 connections
prediction = model.predict(test_connection)
print(f"Predicted CPU Load for {test_connection[0][0]} connections: {prediction[0][0]:.2f}%")

# 4. Visualization
plt.scatter(X_connections, y_cpu_load, color='blue', label='Actual Data')
plt.plot(X_connections, model.predict(X_connections), color='red', linewidth=2, label='Regression Line')
plt.title('Network Connections vs CPU Load')
plt.xlabel('Number of Active Connections')
plt.ylabel('CPU Utilization (%)')
plt.legend()
plt.grid(True)
plt.savefig('cpu_prediction_plot.png') # Save for the README!
plt.show()