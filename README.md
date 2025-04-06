# PSO_FL_Sim
<p align="center">
  <img height="50%" width="50%" src="https://github.com/user-attachments/assets/c55ca5cd-2d12-480e-9576-ccd2846dad23"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-ffffff?logo=Python&logoColor=000000&logoSize=auto" alt="Language - Python">
  <img src="https://img.shields.io/badge/PSO_FL_SIM-white?logoSize=auto&logo=github&logoColor=black" alt="PSO_FL_SIM">
</p>

## Overview 📚
PSO_FL_SIM is a Black Box PSO simulator designed to reduce the computational costs for Federated Learning (FL) clients during training, all while keeping client data private and secure.

## Description 📖
A major challenge with Federated Learning (FL) is its high computational cost, which can slow down the training process. To address this, we're focusing on optimizing the structure of the FL system without compromising privacy. The goal is to figure out the most efficient client configuration, reducing delays while keeping the data private. This simulation looks at ways to improve FL performance by tweaking the system's architecture and workflow.

## System Simulation Model 🏗️
Our system uses a hierarchical structure of clients, with each client being either an Aggregator Trainer (AgTrainer) or a Trainer. Here’s a breakdown of how it works:

### Hierarchy Structure
- **DEPTH :** How many levels the hierarchy has.
- **WIDTH :** The number of clients at each level.
- **AgTrainers :** These are middle-level nodes in the hierarchy, managing child nodes.
- **Trainers :** These clients are at the last level and don’t have child nodes by default. However, they can be reassigned to AgTrainers later, allowing them to manage a non-empty processing buffer.

### Client Class
Clients are defined by the following resources/features:
- **Unique Client ID**
- **Processing Speed**
- **Memory Capacity**
- **Model Data Size** (fixed at 5)

Each client keeps a processing buffer for child nodes. Trainers don’t have children initially, so their buffers stay empty unless their role changes. This allows dynamic role changes, which helps optimize the training process and resource use within the FL system.

### Black Box PSO
The way Particle Swarm Optimization (PSO) works allows us to have multiple possible solutions that evolve over time, improving until we reach an optimal solution.

Here are the key parts of our PSO model:
- **iw (Inertia Weight) :** Helps balance exploration (searching for new solutions) vs. exploitation (focusing on known good solutions).
- **c₁ (Pbest Coefficient) :** Controls how much a particle focuses on its best solution.
- **c₂ (Gbest Coefficient) :** Controls how much a particle focuses on the best solution found by the entire swarm.
- **pop_n (Swarm Population Size) :** The number of particles in the swarm.
- **velocity_factor :** A factor that limits how fast a particle can move, preventing it from making excessive jumps in the search space.

### Particle Class
The `Particle` class is where we define an individual solution in PSO. Each particle has:
- **position :** A vector of Client IDs that represents the current solution by encoding how the clients are assigned.
- **fitness :** A float value that tells us how good the current solution is. Since we’re solving a maximization problem, a higher fitness value means a better solution.
- **velocity :** A vector that tells us how much the position should change.
- **best_position :** The best solution the particle has found so far.
- **best_position_fitness :** The fitness value of the `best_position`, showing the quality of the best solution found by the particle.

### Swarm Class
The `Swarm` class represents a group of particles with the following attributes:
- **particles :** The collection of particles (solutions) in the swarm.
- **global_best_particle :** The best particle among all particles in the swarm.

In Black Box PSO, each particle represents a vector of AgTrainer client IDs, defining their placement order in the hierarchy. This setup enables efficient optimization and a flexible system design, ultimately reducing training and processing time delays.

### Fitness function 
Since our problem is a minimization problem, the objective is to maximize the value of -1 × total_processing_delay.

## Simulation Process 💻
1. Generate the hierarchy of clients.
2. Populate the swarm with random positions for the particles.
3. For each particle in the swarm:
   - Update the particle's velocity using the current position, best position, global best position, iw ,c1 and c2.
   - Apply the updated velocity to the current particle's position.
   - Rearrange the hierarchy based on the new position.
   - Compute the new fitness of the Particle and the total processing delay (tpd).
   - If the new fitness is better than the particle's best position fitness, update its best position and best position's fitness.
   - If the new fitness is better than the global best particle's fitness, update the global best particle.
4. Repeat case 3's steps until reaches the maximum iteration number.

## Screenshots 📸
**Note 1 :** The TPD plots shown below do not include the initial total processing delay of the hierarchy before the swarm is created. We excluded it from the plots to focus on observing the progress of solution refinement.
**Note 2 :** Decreasing TPD is not the sole objective of this algorithm. It also offers several core features, including preserving client privacy (eliminating the need for clients to share their resource information), adaptability to various circumstances, optimization, and more.

<p align="left">
  <img height=60% width=60% src="https://github.com/user-attachments/assets/29c1b8c4-7fda-47c1-a4e4-351ca397ae14"/>
  <img height=60% width=60% src="https://github.com/user-attachments/assets/72e9d465-8b77-434f-9d33-b47da4cbc000"/>
</p>

**Parameters :**  
- DEPTH : 4
- WIDTH : 4
- dimensions : 85
- randomness_seed : 11
- iw : 0.3
- c1 : 0.3
- c2 : 1
- pop_n : 10
- max_iter : 100
- velocity_factor : 0.5

**Results :**
- Initial Hierarchy TPD Before PSO : **33.33**
- Final Best Hierarchy TPD After PSO : **22.91**

<br>
<p>
  <img height=60% width=60% src="https://github.com/user-attachments/assets/ae9d14bd-6458-48fe-9a44-68036eb17f13"/>
  <img height=60% width=60% src="https://github.com/user-attachments/assets/07e44ec3-bd93-4948-b12b-2d9c39ec57f6"/>
</p>

**Parameters :**
- DEPTH : 3
- WIDTH : 5
- dimensions : 31
- randomness_seed : 11
- iw : 0.1
- c1 : 0.1
- c2 : 1
- pop_n : 10
- max_iter : 100
- velocity_factor : 0.1

**Results :**
- Initial Hierarchy TPD Before PSO : **23.5**
- Final Best Hierarchy TPD After PSO : **16.78**

<br>
<p>
  <img height=60% width=60% src="https://github.com/user-attachments/assets/df341e30-d8be-4378-b4f5-9233ab4b23a2"/>
  <img height=60% width=60% src="https://github.com/user-attachments/assets/b4e3dd83-ea5d-456e-aede-18fe48256f0b"/>
</p>

**Parameters :** 
- DEPTH : 4
- WIDTH : 5
- dimensions : 156
- randomness_seed : 11
- iw : 0.3
- c1 : 0.3
- c2 : 1
- pop_n : 10
- max_iter : 100
- velocity_factor : 0.5

**Results :**
- Initial Hierarchy TPD Before PSO : **38.5**
- Final Best Hierarchy TPD After PSO : **31.78**

## Installation ⚙️
1. Clone the project 
```
git clone https://github.com/10xComrade/PSO_FL_SIM.git
```
2. Change directory
```
cd PSO_FL_SIM
```
3. Install requirements
```
pip3 install -r requirements.txt
```
4. Run main.py
```
python3 main.py
```
## Contributions 🤝
We appreciate you wanting to contribute to PSO_FL_SIM! Whether you’re fixing bugs, adding new features, improving documentation, or suggesting improvements, your help is welcome!
