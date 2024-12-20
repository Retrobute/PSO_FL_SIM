from random import randint , random , sample
from time import sleep as delay
from math import floor 

# Global parameters
Client_list = []
Role_buffer = []
Role_dictionary = {}

# System parameters
DEPTH = 3
WIDTH = 2

# PSO parameters
b = 0.00316                                 # Penalty coefficient
a = 0.025                                   # Total delay coefficient
iw = 0.5                                    # Inertia Weight    
c1 = 2.0                                    # Pbest coefficient
c2 = 2.0                                    # Gbest coefficient
pop_n = 3                                   # Population number
vmax = 2                                    # Maximum velocity
max_iter = 1000                              # Maximum iteration
conv = 0.1                                  # Convergence value
dimensions = 7                              # TODO : Make it dynamic, later.
global_best = 0.8                           # NOTE : Temporary value , change it later !!!

# Particle class
class Particle :
    def __init__(self, pos , fitness , velocity) : 
        self.pos = pos
        self.fitness = fitness
        self.velocity = velocity
        self.best_pos = self.pos.copy()

# Swarm class
class Swarm : 
    def __init__(self , pop_n , dimensions) :
        self.particles = self.__generate_random_particles(pop_n , dimensions)
        self.global_best_particle = max(self.particles, key=lambda particle: particle.fitness)

    def __generate_random_particles(self, pop_n, dimensions):
        root = generate_hierarchy(DEPTH , WIDTH)
        init_particle_pos = [client.client_id for client in Client_list if client.is_aggregator]
        cll = len(Client_list)
        particles = []

        for i in range(pop_n):
            if i != 0 : 
                particle_pos = sample(range(cll), dimensions)
                root = reArrangeHierarchy(particle_pos)  
            else : 
                particle_pos = init_particle_pos
            fitness, _ , _ = processing_cost(root)
            # velocity = [randint(-cll, cll) for _ in range(dimensions)]          # method 1 
            velocity = [0 for _ in range(dimensions)]                          # method 2
            particles.append(Particle(particle_pos, fitness, velocity))

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


# Cost function
def processing_cost(master):
    bfs_queue = [master]                     # Start with the root node
    levels = []                              # List to store nodes level by level
    total_process_delay = 0
    total_memscore = 0 

    # Perform BFS to group nodes by levels
    while bfs_queue:
        level_size = len(bfs_queue)
        current_level = []

        for _ in range(level_size):
            current_node = bfs_queue.pop(0)
            current_level.append(current_node)

            if current_node.is_aggregator :
                bfs_queue.extend(current_node.processing_buffer)  

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

                # Print details for the cluster
                # print(f"AgTrainer: {node.label}, MDataSize: {node.mdatasize} Memory Consumption : {cluster_head_memcons}, Cluster Head Delay: {cluster_delay:.2f}")
                
                # for child in node.processing_buffer:
                    # print(f"Trainer: {child.label}, MDataSize: {child.mdatasize}")

        # Find the maximum cluster delay for the level
        if cluster_delays:
            max_cluster_delay = max(cluster_delays)
            total_process_delay += max_cluster_delay  # Add max delay of the level to the total delay
            # print(f"Level Max Cluster Delay: {max_cluster_delay:.2f}\n")

    # print(f"Total Processing Delay: {total_process_delay:.2f}")
    # print(f"Total Memory Score: {total_memscore}")

    return ((a * 1/total_process_delay) + (b * total_memscore)) , total_process_delay , total_memscore


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

def updateVelocity(current_velocity, current_position, personal_best, global_best, w, c1, c2):
    r1 = [random() for _ in range(len(current_velocity))]
    r2 = [random() for _ in range(len(current_velocity))]
    
    inertia = [floor(w * v) for v in current_velocity]
    cognitive = [floor(c1 * r1[i] * (personal_best[i] - current_position[i])) for i in range(len(current_velocity))]
    social = [floor(c2 * r2[i] * (global_best[i] - current_position[i])) for i in range(len(current_velocity))]
    
    new_velocity = [inertia[i] + cognitive[i] + social[i] for i in range(len(current_velocity))]
    return new_velocity    

def applyVelocity(p_position , p_velocity) : 
    # new_position = [a + b for a , b in zip(p_position , p_velocity)]
    new_position = []
    for a , b in zip(p_position , p_velocity) : 
        np = a + b
        direction =  -1 if np < 0 else 1
        while True : 

            if np < 0 : 
                np += len(Client_list)
                continue

            if np >= len(Client_list) : 
                np -= len(Client_list)
                continue

            if np in new_position :
                np += direction
                continue
            
            break

        new_position.append(np)

        # if a + b < 0 : 
        #     new_position.append(0)

        # elif a + b > len(Client_list) : 
        #     new_position.append(len(Client_list) - 1)
        
        # elif (a + b) in new_position : 
        #     new_position.append(a)
        
        # else : 
        #     new_position.append(a + b)

    return new_position 

def main() :

    # print("\n############## TREE BEFORE REARRANGEMENT ##################\n")
    # root = generate_hierarchy(depth=DEPTH, width=WIDTH)

    # _ , _ , mem_score = processing_cost(root)

    # print("\nGenerated Hierarchy : ")
    # printTree(root)
    # print("Mem Score : " , mem_score)
    
    # print("\n############## TREE AFTER REARRANGEMENT ##################\n")
    # pos = [8 , 9 , 10 , 11 , 12 , 13 , 14]
    # root = reArrangeHierarchy(pos)

    # _ , _ , mem_score = processing_cost(root)
    
    # print("\nGenerated Hierarchy : ")
    # printTree(root)
    # print("Mem Score : " , mem_score)

    counter = 1 

    swarm = Swarm(pop_n , dimensions)


    while counter < max_iter : 
        for particle in swarm.particles :   
            
            root = reArrangeHierarchy(particle.best_pos)
            old_pos_fitness , _ , _ = processing_cost(root)

            new_velocity = updateVelocity(particle.velocity , particle.pos , particle.best_pos , swarm.global_best_particle.best_pos , iw , c1 , c2)
            new_position = applyVelocity(particle.pos , new_velocity)
            root = reArrangeHierarchy(new_position)
            particle.pos = new_position 

            new_pos_fitness , tp , tm = processing_cost(root)
            particle.fitness = new_pos_fitness
            
            if new_pos_fitness > old_pos_fitness : 
                particle.best_pos = particle.pos

            if particle.fitness > swarm.global_best_particle.fitness :
                swarm.global_best_particle = particle
            
            print(f"Fitness : {new_pos_fitness} , Total Processing Delay : {tp} , Total Memory Score : {tm}")
            
            with open("./result.txt" , "a+") as file : 
                file.write(f"Fitness : {new_pos_fitness:.4f} , \tTotal Processing Delay : {tp:.4f} , \tTotal Memory Score : {tm:.4f}\n")

        if abs(swarm.global_best_particle.fitness - global_best) < conv : 
            break

        counter += 1
    


if __name__ == "__main__" :
    main()





