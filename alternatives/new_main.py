import numpy as np
import matplotlib.pyplot as plt

# Objective Function (calculate processing delay)
def calculate_processing_delay(task_to_node_mapping, task_durations, node_capacities):
    total_delay = 0
    node_load = np.zeros(len(node_capacities))  # Tracks the load of each node
    
    for task_idx, node_idx in enumerate(task_to_node_mapping):
        task_duration = task_durations[task_idx]
        node_capacity = node_capacities[node_idx]
        
        node_load[node_idx] += task_duration
        total_delay += task_duration * (node_load[node_idx] / node_capacity)
    
    return total_delay

# Particle Swarm Optimization (PSO) implementation
def pso(task_durations, node_capacities, n_particles=30, n_iterations=1000):
    n_tasks = len(task_durations)
    n_nodes = len(node_capacities)
    
    # Initialize particles with random unique assignments (node ids for tasks)
    particles = np.array([np.random.permutation(n_nodes)[:n_tasks] for _ in range(n_particles)])
    velocities = np.random.randn(n_particles, n_tasks)  # Random initial velocities
    
    # Personal bests and global best
    personal_best = particles.copy()
    personal_best_score = np.array([calculate_processing_delay(p, task_durations, node_capacities) for p in particles])
    global_best = personal_best[np.argmin(personal_best_score)]
    
    # PSO hyperparameters
    w = 0.5  # inertia weight
    c1 = 1.5  # cognitive coefficient
    c2 = 1.5  # social coefficient
    
    # To store the processing delays at each iteration
    all_delays = []

    # Main PSO loop
    for iteration in range(n_iterations):
        iteration_delays = []
        
        for i in range(n_particles):
            # Calculate current score (processing delay)
            current_score = calculate_processing_delay(particles[i], task_durations, node_capacities)
            iteration_delays.append(current_score)
            
            # Update personal best
            if current_score < personal_best_score[i]:
                personal_best[i] = particles[i]
                personal_best_score[i] = current_score
            
            # Update global best
            if current_score < calculate_processing_delay(global_best, task_durations, node_capacities):
                global_best = particles[i]
        
        # Store delays of this iteration
        all_delays.append(iteration_delays)
        
        # Update velocity and position of each particle
        for i in range(n_particles):
            r1 = np.random.rand(n_tasks)
            r2 = np.random.rand(n_tasks)
            
            # Update velocity
            velocities[i] = (w * velocities[i] 
                             + c1 * r1 * (personal_best[i] - particles[i]) 
                             + c2 * r2 * (global_best - particles[i]))
            
            # Update position (ensure unique node assignments)
            new_position = particles[i] + np.round(velocities[i]).astype(int)
            new_position = np.clip(new_position, 0, n_nodes - 1)
            
            # Ensure unique node assignments and that we have exactly n_tasks unique nodes
            unique_position = np.unique(new_position[:n_nodes])[:n_tasks]
            
            if len(unique_position) < n_tasks:
                remaining_tasks = n_tasks - len(unique_position)
                available_nodes = set(range(n_nodes)) - set(unique_position)
                unique_position = np.concatenate([unique_position, list(available_nodes)[:remaining_tasks]])
            
            particles[i] = unique_position[:n_tasks]
        
        # Print the current best processing delay (optional)
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Global best delay = {calculate_processing_delay(global_best, task_durations, node_capacities)}")
    
    # Return the global best result and the list of delays
    return global_best, all_delays

# Example usage
task_durations = [10, 15, 20, 25]  # durations of tasks
node_capacities = [50, 60, 70, 80]  # capacities of nodes

# Run PSO to find the optimal task placement
best_task_to_node_mapping, all_delays = pso(task_durations, node_capacities)

# Compute statistics for plotting
min_delays = [min(iteration) for iteration in all_delays]
max_delays = [max(iteration) for iteration in all_delays]
avg_delays = [np.mean(iteration) for iteration in all_delays]

# Plot the processing delays
plt.figure(figsize=(10, 5))
plt.plot(min_delays, label="Min Delay", color='green')
plt.plot(max_delays, label="Max Delay", color='red')
plt.plot(avg_delays, label="Avg Delay", color='blue', linestyle='dashed')
plt.xlabel("Iteration")
plt.ylabel("Processing Delay")
plt.title("Processing Delay vs Iterations in PSO")
plt.legend()
plt.grid(True)
plt.show()
