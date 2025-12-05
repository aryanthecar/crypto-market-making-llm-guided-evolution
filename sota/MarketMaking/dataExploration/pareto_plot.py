import numpy as np
import matplotlib.pyplot as plt

# Data points
roi = np.array([-1.60, -1.55, -1.50, -1.45, -1.40, -1.35, -1.30])
max_drawdown = np.array([0.143, 0.147, 0.148, 0.143, 0.172, 0.149, 0.160])

# Create the plot
plt.figure(figsize=(10, 6))

# Plot all points in blue
plt.scatter(roi, max_drawdown, color='blue', s=50)

# Plot the seed point in red (at -1.45, 0.143)
plt.scatter(-1.45, 0.143, color='red', s=50)

# Plot the bottom-left point in blue (at -1.60, 0.143)
plt.scatter(-1.60, 0.143, color='blue', s=50)

# Create Pareto frontier line with only horizontal and vertical segments
# Start from top left corner
frontier_x = [-1.60, -1.60, -1.45, -1.45, -1.30, -1.30]
frontier_y = [0.18, 0.143, 0.143, 0.143, 0.143, 0.143]

# Plot the Pareto frontier line
plt.plot(frontier_x, frontier_y, 'r--', linewidth=1.5)

# Labels and title
plt.xlabel('ROI')
plt.ylabel('MaxDrawdown')
plt.title('ROI vs MaxDrawdown with Pareto Frontier')

# Set y-axis limits to ensure we see the top left corner
plt.ylim(0.14, 0.185)

# Grid
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot
plt.savefig('pareto_frontier.png', dpi=300, bbox_inches='tight')
plt.close()