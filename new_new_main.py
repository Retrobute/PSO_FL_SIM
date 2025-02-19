
from random import randint , random , sample , seed 
from math import floor 
import copy 
import numpy as np
import matplotlib.pyplot as plt

# System parameters
DEPTH = 2
WIDTH = 3
dimension = 0 if DEPTH <= 0 or WIDTH <= 0 else sum(WIDTH**i for i in range(DEPTH))                          
Client_list = []
Role_buffer = []
Role_dictionary = {}
randomness_seed = 22
tracking_mode = True   
                                            
# TXT output file required parameters
txt_info = []
file_path = "./measurements/results/main_result.txt"


        
class Client :
    def __init__(self, memcap, mdatasize, client_id , label , pspeed , is_aggregator=False) :
        self.memcap = memcap 
        self.mdatasize = mdatasize
        self.label = label 
        self.pspeed = pspeed
        self.is_aggregator = is_aggregator
        self.client_id = client_id  
        self.processing_buffer = []
        self.memscore = 0

    def changeRole(self , new_pos) :         # This function traverses the Client_list to find the client with equal client_id then it first buffers the role of the client if the role is trainer, and then associates the new_role_label to the selected client 
        if not self.is_aggregator : 
            Role_buffer.append(self.label) 
        self.processing_buffer = []
        self.label = list(Role_dictionary.keys())[new_pos]
        self.is_aggregator = True
    
    def takeAwayRole(self) :                 # This function traverses the Client_list and checks for the client that has the selected role in the arguments then it nulls the label and the processing_buffer   
        self.label = None
        self.processing_buffer = []  

# Fitness function
def processing_fitness(master):
    bft_queue = [master]                     # Start with the root node
    levels = []                              # List to store nodes level by level
    total_process_delay = 0
    total_memscore = 0 

    # Perform BFT to group nodes by levels
    while bft_queue:
        level_size = len(bft_queue)
        current_level = []

        for _ in range(level_size):
            current_node = bft_queue.pop(0)
            current_level.append(current_node)

            if current_node.is_aggregator :
                bft_queue.extend(current_node.processing_buffer)  

        levels.append(current_level)  

    levels.reverse()

    # Calculate delays level by level
    for level in levels:
        cluster_delays = []  

        for node in level:
            if node.is_aggregator :
                # Update the node's mdatasize with its children's cumulative memory size
                cluster_head_memcons = node.mdatasize + sum(
                    child.mdatasize for child in node.processing_buffer
                )
                node.memscore = node.memcap - cluster_head_memcons
                total_memscore += node.memscore
                cluster_delay = cluster_head_memcons / node.pspeed
                cluster_delays.append(cluster_delay)

        # Find the maximum cluster delay for the level
        if cluster_delays:
            max_cluster_delay = max(cluster_delays)
            total_process_delay += max_cluster_delay  # Add max delay of the level to the total delay
            # print(f"Level Max Cluster Delay: {max_cluster_delay:.2f}\n")

    return  total_process_delay

def generate_hierarchy(depth, width):
    level_agtrainer_list = []
    agtrainer_list = []
    trainer_list = []

    def create_agtrainer(label_prefix, level):
        pspeed = randint(5, 15)
        memcap = randint(10, 50)
        mdatasize = 5                         # in the beginning it's a fixed value but in the future as a stretch goal we can have variable MDataSize due to quantization and knowledge distillation techniques
        length = len(Client_list) 
        new_client = Client(memcap, mdatasize, length, f"t{label_prefix}ag{level}", pspeed, True)
        Client_list.append(new_client)
        agtrainer_list.append(new_client)
        level_agtrainer_list.append(new_client)
        return new_client

    def create_trainer(label_prefix):
        pspeed = randint(5, 15)
        memcap = randint(10, 50)
        mdatasize = 5 
        length = len(Client_list)
        new_client = Client(memcap, mdatasize, length, label_prefix , pspeed)
        Client_list.append(new_client)
        trainer_list.append(new_client)
        return new_client

    root = create_agtrainer(0, 0)
    current_level = [root]
    level_agtrainer_list = []

    for d in range(1, depth):
        next_level = []
        for parent in current_level:
            for _ in range(width):
                child = create_agtrainer(len(level_agtrainer_list), d)
                parent.processing_buffer.append(child)
                next_level.append(child)

                for role in [parent , child] :
                    Role_dictionary[role.label] = [child.label for child in role.processing_buffer]

        if d == depth - 1:    
            for client in level_agtrainer_list :
                for j in range(2):          
                    trainer = create_trainer(f"{client.label}_{j+1}")
                    client.processing_buffer.append(trainer)

                for role in [client , trainer] :
                    Role_dictionary[role.label] = [child.label for child in role.processing_buffer]

        level_agtrainer_list = []
        current_level = next_level

    return root

def printTree(node, level=0, is_last=True, prefix=""):
    connector = "└── " if is_last else "├── "
    if node.is_aggregator : 
        print(f"{prefix}{connector}{node.label} (MemCap: {node.memcap}, MDataSize: {node.mdatasize} Pspeed: {node.pspeed}, ID: {node.client_id}, MemScore: {node.memscore})")

    elif node.is_aggregator == False :
        print(f"{prefix}{connector}{node.label} (MemCap: {node.memcap}, MDataSize: {node.mdatasize}, ID: {node.client_id}, MemScore: {node.memscore})")

    if node.is_aggregator :
        for i, child in enumerate(node.processing_buffer):
            new_prefix = prefix + ("    " if is_last else "│   ")
            printTree(child, level + 1, i == len(node.processing_buffer) - 1, new_prefix)

def changeRole(client , new_pos) :                # This function traverses the Client_list to find the client with equal client_id then it first buffers the role of the client if the role is trainer, and then associates the new_role_label to the selected client 
    if not client.is_aggregator : 
        Role_buffer.append(client.label) 
    client.processing_buffer = []
    client.label = list(Role_dictionary.keys())[new_pos]
    client.is_aggregator = True
    
def takeAwayRole(client) :                        # This function traverses the Client_list and checks for the client that has the selected role in the arguments then it nulls the label and the processing_buffer   
    client.label = None
    client.processing_buffer = []   

def reArrangeHierarchy(pso_particle) :            # This function has the iterative approach to perform change role and take away role
    for new_pos , clid in enumerate(pso_particle) : 
        for client in Client_list : 
            if client.label == list(Role_dictionary.keys())[new_pos] :
                client.takeAwayRole()

            if client.client_id == clid : 
                client.changeRole(new_pos)
                
            client.memscore = 0
            
    for client in Client_list : 
        if client.label == None :
            client.label = Role_buffer.pop()    
            client.is_aggregator = False 
    
        if client.is_aggregator : 
            if len(client.processing_buffer) == 0 : 
                temp = Role_dictionary[client.label]
                for role in temp : 
                    for c in Client_list :
                        if  c.label == role : 
                            client.processing_buffer.append(c) 
                        
    for client in Client_list :
        if client.label == list(Role_dictionary.keys())[0] :
            return client


# Particle Swarm Optimization (PSO) implementation
def pso(root_node, particle_size, num_clients, seed, n_particles=30, n_iterations=20):
    
    n_tasks = particle_size
    n_nodes = num_clients
    # np.random.seed(seed)
    # Initialize particles with random unique assignments (node ids for tasks)
    particles = np.array([np.random.permutation(n_nodes)[:n_tasks] for _ in range(n_particles)])
    velocities = np.random.randn(n_particles, n_tasks)  # Random initial velocities
    
    # Personal bests and global best
    personal_best = particles.copy()
    personal_best_score = np.array([processing_fitness(root_node) for p in particles])
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
            root_node = reArrangeHierarchy(particles[i])
            current_score = processing_fitness(root_node)
            iteration_delays.append(current_score)
            
            # Update personal best
            if current_score < personal_best_score[i]:
                personal_best[i] = particles[i]
                personal_best_score[i] = current_score
            
            # Update global best
            if current_score < processing_fitness(root_node):
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
            print(f"Iteration {iteration}: Global best delay = {processing_fitness(root_node)}")
    
    # Return the global best result and the list of delays
    return global_best, all_delays




def PSO_FL_SIM() :    
    
    # randomness_seed = 20  
    # if tracking_mode : 
    #     seed(randomness_seed)
        

    root = generate_hierarchy(DEPTH , WIDTH)
    p_delay = processing_fitness(root)
    # root = reArrangeHierarchy(new_position)
    
    num_clients = len(list(Role_dictionary.keys()))
    best_task_to_node_mapping, all_delays = pso(root,dimension,num_clients, 0, n_particles=10,n_iterations=300)

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

    

if __name__ == "__main__" : 
    PSO_FL_SIM()
