from random import randint , random
from time import sleep as delay

# System parameters
DEPTH = 3
WIDTH = 2

# Global parameters
Client_list = []
Role_buffer = []
Role_dictionary = {}

# PSO parameters
b = 0.00316             # Penalty coefficient
a = 0.015               # Total delay coefficient
iw = 0.5                # Inertia Weight    
pc = 2.0                # Pbest coefficient
gc = 2.0                # Gbest coefficient
pop_n = 30              # Population number
vmax = 0.1              # Maximum velocity
max_iter = 100          # Maximum iteration
conv = 0.0001           # Convergence value
particle_length = (WIDTH * DEPTH) - 1 / (WIDTH - 1)         

# Particle class
class Particle :
    def __init__(self, pos , velocity) : 
        self.pos = pos
        self.velocity = velocity
        self.best_pos = self.pos.copy()

# Swarm class
class Swarm : 
    def __init__(self , pop_n) :
        self.particles = [self.__generate_particles() for _ in range(pop_n)]
        self.best_pos = None 
            
    def __generate_particles(self) : 
        root = generate_hierarchy(DEPTH , WIDTH)
        particle = [root]
        
        def extract_inner_clients(parent) : 
            for child in parent.processing_buffer : 
                if child.is_aggregator : 
                    particle.append(child)
                    extract_inner_clients(child)

        extract_inner_clients(root)
        
        return particle
        
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

    def changeRoleToAggregator(self) :
        self.is_aggregator = True

    def changeRoleToTrainer(self) :
        self.processing_buffer = []
        self.is_aggregator = False

# Evaluator class has the necessary measurement methods
class Evaluator :  
    @staticmethod 
    def fitness() :     # Current fitness is  based on Total delay & Memory consumption    
        pass

    @staticmethod 
    def penalty() :     # Penalty function is supposed to be a part of fitness function
        pass

    @staticmethod
    def calculate_processing_cost(master):
        bfs_queue = [master]  # Start with the root node
        levels = []           # List to store nodes level by level
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
                    print(f"AgTrainer: {node.label}, Memory Consumption : {cluster_head_memcons}, Cluster Delay: {cluster_delay:.2f}, Memory Score: {node.memscore}")
                    
                    for child in node.processing_buffer:
                        print(f"Trainer: {child.label}, Memory Score: {child.mdatasize}")

            # Find the maximum cluster delay for the level
            if cluster_delays:
                max_cluster_delay = max(cluster_delays)
                total_process_delay += max_cluster_delay  # Add max delay of the level to the total delay
                print(f"Level Max Cluster Delay: {max_cluster_delay:.2f}\n")

        print(f"Total Processing Delay: {total_process_delay:.2f}")
        print(f"Total Memory Score: {total_memscore}")

        return total_process_delay , total_memscore 


def generate_hierarchy(depth, width):
    level_agtrainer_list = []
    agtrainer_list = []
    trainer_list = []

    def create_agtrainer(label_prefix, level):
        pspeed = randint(5, 15)
        memcap = randint(10, 50)
        mdatasize = 5 # in the beggining it's a fixed value but in the future as a stretch goal we can have variable MDataSize due to quantization and knowledge distillation techniques
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


def changeRole(client , new_pos) :         # This function traverses the Client_list to find the client with equal client_id then it first buffers the role of the client if the role is trainer, and then associates the new_role_label to the selected client 
    if not client.is_aggregator : 
        Role_buffer.append(client.label) 
    client.processing_buffer = []
    client.label = list(Role_dictionary.keys())[new_pos]
    client.is_aggregator = True
    
def takeAwayRole(client) :                 # This function traverses the Client_list and checks for the client that has the selected role in the arguments then it nulls the label and the processing_buffer   
    client.label = None
    client.processing_buffer = []             

#                        new pos  -> [0 , 1 , 2 , 3 , 4 , 5 , 6]
def reArrangeHierarchy(pso_particle=[8 , 9 , 10 , 11 , 12 , 13 , 14]) :            # This function has the iterative approach to perform change role and take away role
    for new_pos , clid in enumerate(pso_particle) : 
        for client in Client_list : 
            if client.label == list(Role_dictionary.keys())[new_pos] :
                takeAwayRole(client)

            if client.client_id == clid : 
                changeRole(client , new_pos)
            
    for client in Client_list : 
        if client.label == None :
            client.label = Role_buffer.pop()    
            client.is_aggregator = False 
            client.memscore = 0
    
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
    r1 = [random.random() for _ in range(len(current_velocity))]
    r2 = [random.random() for _ in range(len(current_velocity))]
    
    inertia = [w * v for v in current_velocity]
    cognitive = [c1 * r1[i] * (personal_best[i] - current_position[i]) for i in range(len(current_velocity))]
    social = [c2 * r2[i] * (global_best[i] - current_position[i]) for i in range(len(current_velocity))]
    
    new_velocity = [inertia[i] + cognitive[i] + social[i] for i in range(len(current_velocity))]
    return new_velocity    

def main() :
    print("\n############## TREE BEFORE REARRANGEMENT ##################\n")
    root = generate_hierarchy(depth=DEPTH, width=WIDTH)

    Evaluator.calculate_processing_cost(root)

    print("\nGenerated Hierarchy : ")
    printTree(root)
    
    print("\n############## TREE AFTER REARRANGEMENT ##################\n")
    root = reArrangeHierarchy()

    Evaluator.calculate_processing_cost(root)
    
    print("\nGenerated Hierarchy : ")
    printTree(root)

    # swarm = Swarm(pop_n)

    # current_position = [8, 9, 10, 11, 12, 13, 14]
    # current_velocity = [0, 0, 0, 0, 0, 0, 0]

    # updateVelocity()



if __name__ == "__main__" :
    main()





