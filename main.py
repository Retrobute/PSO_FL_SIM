from measurements.tools.display_output import *
from measurements.tools.store_output import *
from random import randint , random , sample , seed 
from datetime import datetime as d
import os
import copy 
import sys

# Global parameters
# PSO parameters                            
iw = .3                                     # Inertia Weight (Higher => Exploration | Lower => Exploitation)   (0.1 , 0.5)
c1 = .3                                     # Pbest coefficient (0.01 , 0.1)
c2 = 1                                      # Gbest coefficient 
pop_n = 10                                  # Population number (3 , 5 , 10 , 15 , 20*)
max_iter = 100                              # Maximum iteration

# System parameters
DEPTH = 4
WIDTH = 5
dimensions = 0 if DEPTH <= 0 or WIDTH <= 0 else sum(WIDTH**i for i in range(DEPTH))   
Client_list = []
Role_buffer = []
Role_dictionary = {}
randomness_seed = 11
tracking_mode = True   
velocity_factor = 0.5                       # Increasing velocity_factor causes more exploration resulting higher fluctuations in the particles plot (default range between 0 and 1 (Guess))

# Experiment parameters
scenario_file_name = f"width_{WIDTH}_{d.now().strftime("%Y-%m-%d_%H:%M:%S")}" 
scenario_folder_number = DEPTH                       
scenario_folder_name = f"depth_{scenario_folder_number}_scenarios"

# Graph illustration required parameters 
particles_fitness_fig_path = f"./measurements/results/{scenario_folder_name}/particles_fitness_{scenario_file_name}.png"
swarm_best_fitness_fig_path = f"./measurements/results/{scenario_folder_name}/swarm_best_fitness_{scenario_file_name}.png"
tpd_fig_path = f"./measurements/results/{scenario_folder_name}/tpd_{scenario_file_name}.png"

sbpfl = ("iteration" , "best particle fitness")
pfl = ("iteration" , "particles fitness") 
tpdl = ("iteration" , "total processing delay")

# Plot titles, empty for now
sbpft = ""
pft = ""
tpdt = ""

gbest_particle_fitness_results = []
particles_fitnesses_buffer = []
particles_fitnesses_tuples = []

tpd_buffer = []
tpd_tuples = []
iterations = []

# CSV output required parameters
csv_particles_output_file_name = f"particles_data_{scenario_file_name}"
csv_swarm_best_output_file_name = f"swarm_best_data_{scenario_file_name}"
csv_tpd_output_file_name = f"tpd_data_{scenario_file_name}"

csv_particles_data_path = f"./measurements/results/{scenario_folder_name}/{csv_particles_output_file_name}.csv"
csv_swarm_best_data_path = f"./measurements/results/{scenario_folder_name}/{csv_swarm_best_output_file_name}.csv"
csv_tpd_data_path = f"./measurements/results/{scenario_folder_name}/{csv_tpd_output_file_name}.csv"

particles_columns = ["iteration"] + [f"particle_{i+1}_fitness" for i in range(pop_n)]
swarm_best_columns = ["iteration", "swarm_best_fitness"]
tpd_columns = ["iteration"] + [f"tpd_particle_{i+1}" for i in range(pop_n)]

csv_cols = [particles_columns, swarm_best_columns, tpd_columns]
csv_rows = [[], [], []]

# JSON output required parameters (Particles constant metadata)
json_path = f"./measurements/results/{scenario_folder_name}/pso_scenario_case_{scenario_file_name}.json"
json_pso_dict = {
    "DEPTH" : DEPTH,
    "WIDTH" : WIDTH,
    "dimensions" : dimensions,
    "randomness_seed" : randomness_seed,
    "iw" : iw,
    "c1" : c1,
    "c2" : c2,
    "pop_n" : pop_n,
    "max_iter" : max_iter,
    "velocity_factor" : velocity_factor
} 

# Particle class
class Particle :
    def __init__(self, pos , fitness , velocity , best_pos_fitness) : 
        self.pos = pos
        self.fitness = fitness
        self.velocity = velocity
        self.best_pos = self.pos.copy()
        self.best_pos_fitness = best_pos_fitness

# Swarm class
class Swarm : 
    def __init__(self , pop_n , dimensions , root) :
        self.particles = self.__generate_random_particles(pop_n , dimensions , root)
        self.global_best_particle = copy.deepcopy(max(self.particles, key=lambda particle: particle.fitness))

    def __generate_random_particles(self, pop_n, dimensions , root):
        cll = range(len(Client_list))
        particles = []

        for _ in range(pop_n):
            particle_pos = sample(cll, dimensions)
            root = rearrange_hierarchy(particle_pos)  
            fitness, _ = processing_fitness(root)
            velocity = [0] * dimensions 
            best_pos_fitness = fitness                   
            particles.append(Particle(particle_pos, fitness, velocity, best_pos_fitness))

        return particles
        
class Client :
    def __init__(self, memcap, mdatasize, client_id , label , pspeed , is_aggregator=False) :
        self.memcap = memcap 
        self.mdatasize = mdatasize
        self.label = label 
        self.pspeed = pspeed
        self.is_aggregator = is_aggregator
        self.client_id = client_id  
        self.processing_buffer = []

    def change_role(self , new_pos) :         # This function traverses the Client_list to find the client with equal client_id then it first buffers the role of the client if the role is trainer, and then associates the new_role_label to the selected client 
        if not self.is_aggregator : 
            Role_buffer.append(self.label) 
        self.processing_buffer = []
        self.label = list(Role_dictionary.keys())[new_pos]
        self.is_aggregator = True
    
    def take_away_role(self) :                 # This function traverses the Client_list and checks for the client that has the selected role in the arguments then it nulls the label and the processing_buffer   
        self.label = None
        self.processing_buffer = []  

# Fitness function
def processing_fitness(master):
    bft_queue = [master]                     # Start with the root node
    levels = []                              # List to store nodes level by level
    total_process_delay = 0

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
        
                cluster_delay = cluster_head_memcons / node.pspeed
                cluster_delays.append(cluster_delay)

        # Find the maximum cluster delay for the level
        if cluster_delays:
            max_cluster_delay = max(cluster_delays)
            total_process_delay += max_cluster_delay  # Add max delay of the level to the total delay
    
    return -total_process_delay , total_process_delay

def generate_hierarchy(depth, width):
    level_agtrainer_list = []
    agtrainer_list = []
    trainer_list = []

    def create_agtrainer(label_prefix, level):
        pspeed = randint(2, 8)
        memcap = randint(10, 50)
        mdatasize = 5                         # in the beginning it's a fixed value but in the future as a stretch goal we can have variable MDataSize due to quantization and knowledge distillation techniques
        length = len(Client_list) 
        new_client = Client(memcap, mdatasize, length, f"t{label_prefix}ag{level}", pspeed, True)
        Client_list.append(new_client)
        agtrainer_list.append(new_client)
        level_agtrainer_list.append(new_client)
        return new_client

    def create_trainer(label_prefix):
        pspeed = randint(2, 8)
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

def print_hierarchy(node, level=0, is_last=True, prefix=""):
    connector = "└── " if is_last else "├── "
    if node.is_aggregator : 
        print(f"{prefix}{connector}{node.label} (MemCap: {node.memcap}, MDataSize: {node.mdatasize} Pspeed: {node.pspeed}, ID: {node.client_id})")

    elif node.is_aggregator == False :
        print(f"{prefix}{connector}{node.label} (MemCap: {node.memcap}, MDataSize: {node.mdatasize}, ID: {node.client_id})")

    if node.is_aggregator :
        for i, child in enumerate(node.processing_buffer):
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_hierarchy(child, level + 1, i == len(node.processing_buffer) - 1, new_prefix) 


def rearrange_hierarchy(pso_particle) :            # This function has the iterative approach to perform change role and take away role
    for new_pos , clid in enumerate(pso_particle) : 
        for client in Client_list : 
            if client.label == list(Role_dictionary.keys())[new_pos] :
                client.take_away_role()

            if client.client_id == clid : 
                client.change_role(new_pos)
                
    for client in Client_list : 
        if client.label == None :
            client.label = Role_buffer.pop()    
            client.is_aggregator = False 
    
        if client.is_aggregator : 
            if len(client.processing_buffer) == 0 : 
                temp = Role_dictionary[client.label]
                for role in temp : 
                    for c in Client_list :
                        if c.label == role : 
                            client.processing_buffer.append(c) 
                        
    for client in Client_list :
        if client.label == list(Role_dictionary.keys())[0] :
            return client

def update_velocity(current_velocity, current_position, personal_best, global_best, iw, c1, c2):
    r1 = [random() for _ in range(len(current_velocity))]
    r2 = [random() for _ in range(len(current_velocity))]

    inertia = [iw * v for v in current_velocity]
    cognitive = [c1 * r1[i] * (personal_best[i] - current_position[i]) for i in range(len(current_velocity))]
    social = [c2 * r2[i] * (global_best[i] - current_position[i]) for i in range(len(current_velocity))]
    
    max_velocity = max(1, int(len(current_velocity) * velocity_factor))
    new_velocity = [round(inertia[i] + cognitive[i] + social[i]) for i in range(len(current_velocity))]
    new_velocity = [max(min(v, max_velocity), -max_velocity) for v in new_velocity]  # Apply velocity limits

    return new_velocity

def apply_velocity(p_position, p_velocity):
    new_position = []
    client_count = len(p_position)

    for a, b in zip(p_position, p_velocity):
        np = (a + b) % client_count  

        while np in new_position:
            np = (np + 1) % client_count  

        new_position.append(np)

    return new_position

def pso_fl_sim() :    
    global iw

    if tracking_mode : 
        seed(randomness_seed)

    root = generate_hierarchy(DEPTH , WIDTH)
    initial_root = copy.deepcopy(root)
    _ , initial_tpd = processing_fitness(root)
    
    counter = 1

    swarm = Swarm(pop_n , dimensions , root)

    while counter <= max_iter: 
        for particle in swarm.particles :
            particles_fitnesses_buffer.append(particle.fitness)
            
            new_velocity = update_velocity(particle.velocity, particle.pos, particle.best_pos, swarm.global_best_particle.best_pos, iw, c1, c2)
            new_position = apply_velocity(particle.pos, new_velocity)
            root = rearrange_hierarchy(new_position)

            new_pos_fitness, tpd = processing_fitness(root)
            particle.pos = new_position
            particle.fitness = new_pos_fitness
            particle.velocity = new_velocity
            
            if particle.fitness > particle.best_pos_fitness :  
                particle.best_pos = particle.pos.copy()
                particle.best_pos_fitness = copy.copy(particle.fitness)

            if particle.fitness > swarm.global_best_particle.fitness:
                swarm.global_best_particle = copy.deepcopy(particle)              
            
            tpd_buffer.append(tpd)

        iterations.append(counter)
        
        gbest_particle_fitness_results.append(swarm.global_best_particle.fitness)
        tpd_tuples.append(tpd_buffer.copy())
        particles_fitnesses_tuples.append(particles_fitnesses_buffer.copy()) # We could simply reverse the TPD plot and get Particles Fitnesses Plot but as the fitness function might change later this method is not reliable 
        
        particles_row = [counter] + [round(fitness , 2) for fitness in particles_fitnesses_buffer]
        csv_rows[0].append(particles_row)
        
        swarm_best_row = [counter, round(swarm.global_best_particle.fitness , 2)]
        csv_rows[1].append(swarm_best_row)
        
        tpd_row = [counter] + [round(tpd , 2) for tpd in tpd_buffer]
        csv_rows[2].append(tpd_row)

        os.system("cls") if sys.platform == "win32" else os.system("clear")
        print(f"Iteration : {counter}") 
        
        tpd_buffer.clear()
        particles_fitnesses_buffer.clear()
        
        counter += 1

    print_hierarchy(initial_root)
    print("Dimensions : " , dimensions)
    print(f"Initial TPD Before PSO : {initial_tpd}")
    print(f"Final Best TPD After PSO : {-swarm.global_best_particle.fitness}\n")

    save_data_to_csv(csv_cols[0] , csv_rows[0] , csv_particles_data_path)
    save_data_to_csv(csv_cols[1] , csv_rows[1] , csv_swarm_best_data_path)
    save_data_to_csv(csv_cols[2] , csv_rows[2] , csv_tpd_data_path)
    save_metadata_to_json(json_pso_dict , json_path)

    illustrate_plot(gbest_particle_fitness_results , sbpfl , sbpft , swarm_best_fitness_fig_path)
    plot_tuple_curves(particles_fitnesses_tuples , pfl , pft , particles_fitness_fig_path)
    plot_tuple_curves(tpd_tuples , tpdl , tpdt , tpd_fig_path)
    

if __name__ == "__main__" : 
    pso_fl_sim()
