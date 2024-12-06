from random import randint
import re
from time import sleep as delay

# Global parameters
Client_list = []
Role_buffer = []
Role_dictionary = {}
trainer_pattern = r".*_.*"

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

# System parameters
DEPTH = 3
WIDTH = 2

# Particle class
class Particle :
    def __init__(self, pos , velocity) : 
        self.pos = pos 
        self.velocity = velocity
        self.best_pos = self.pos.copy()
    
# Swarm class
class Swarm : 
    def __init__(self , pop_n , vmax) :
        self.particles = []
        self.best_pos = None 

        for _ in range(pop_n) :
            pass 

class Client :
    def __init__(self, memcap, mdatasize, client_id , label , pspeed , is_aggregator=False) :
        self.memcap = memcap 
        self.mdatasize = mdatasize
        self.label = label 
        self.pspeed = pspeed
        self.is_aggregator = is_aggregator
        self.client_id = client_id  
        self.processing_buffer = []

    def changeRoleToAggregator(self , pspeed) :
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
        temp_mdatasize = 0 

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

        total_delay = 0

        # Calculate delays level by level
        for level in levels:
            cluster_delays = []  

            for node in level:
                if node.is_aggregator :

                    # Update the node's mdatasize with its children's cumulative memory size
                    cluster_mem = node.mdatasize + sum(
                        child.mdatasize for child in node.processing_buffer
                    )
                    temp_mdatasize = cluster_mem
                    # node.mdatasize = cluster_mem  # Update the node's mdatasize for parent-level calculations

                    cluster_delay = cluster_mem / node.pspeed
                    cluster_delays.append(cluster_delay)

                    # Print details for the cluster
                    print(f"AgTrainer: {node.label}, Cluster Mem Consumption: {cluster_mem}, Cluster Delay: {cluster_delay:.2f}")
                    for child in node.processing_buffer:
                        print(f"  Trainer: {child.label}, Mem Consumption: {child.mdatasize}")

            # Find the maximum cluster delay for the level
            if cluster_delays:
                max_cluster_delay = max(cluster_delays)
                total_delay += max_cluster_delay  # Add max delay of the level to the total delay
                print(f"Level Max Cluster Delay: {max_cluster_delay:.2f}\n")

        print(f"Total Processing Delay: {total_delay:.2f}")


def generate_hierarchy(depth, width):
    level_agtrainer_list = []
    agtrainer_list = []
    trainer_list = []

    def create_agtrainer(label_prefix, level):
        pspeed = randint(5, 15)
        memcap = randint(10, 50)
        mdatasize = randint(5, 20)
        length = len(Client_list) 
        new_client = Client(memcap, mdatasize, length, f"t{label_prefix}ag{level}", pspeed, True)
        Client_list.append(new_client)
        agtrainer_list.append(new_client)
        level_agtrainer_list.append(new_client)
        return new_client

    def create_trainer(label_prefix):
        memcap = randint(5, 15)
        mdatasize = randint(5, 15)
        pspeed = randint(5, 15)    
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

                if d == depth - 1:  # If this is the last depth, create trainers (leaf nodes)
                    for j in range(2):  # Binary leaves
                        trainer = create_trainer(f"{child.label}_{j+1}")
                        child.processing_buffer.append(trainer)

                for role in [parent , child] :
                    Role_dictionary[role.label] = [child.label for child in role.processing_buffer]

        level_agtrainer_list = []
        current_level = next_level

    return root


def print_tree(node, level=0, is_last=True, prefix=""):
    connector = "└── " if is_last else "├── "
    if node.is_aggregator : 
        print(f"{prefix}{connector}{node.label} (MemCap: {node.memcap}, MDataSize: {node.mdatasize} Pspeed: {node.pspeed}, ID: {node.client_id}")

    elif node.is_aggregator == False :
        print(f"{prefix}{connector}{node.label} (MemCap: {node.memcap}, MDataSize: {node.mdatasize}, ID: {node.client_id}")

    if node.is_aggregator :
        for i, child in enumerate(node.processing_buffer):
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(child, level + 1, i == len(node.processing_buffer) - 1, new_prefix)


def changeRole(index , new_pos) :         # This function traverses the Client_list to find the client with equal client_id then it first buffers the role of the client if the role is trainer, and then associates the new_role_label to the selected client 
    if Client_list[index].is_aggregator == False : 
        Role_buffer.append(Client_list[index].label)  
    Client_list[index].processing_buffer = []
    Client_list[index].label = list(Role_dictionary.keys())[new_pos]
    Client_list[index].is_aggregator = True

def takeAwayRole(index) :  
#    if re.search(trainer_pattern , Client_list[index].label):                         # This function traverses the Client_list and checks for the client that has the selected role in the arguments then it nulls the label and the processing_buffer   
    Client_list[index].label = None
    Client_list[index].processing_buffer = []             

#                        new pos  -> [0 , 1 , 2 , 3]
def reArrangeHierarchy(pso_particle=[2 , 4 , 6 , 3 , 1 , 7 , 11]) :            # This function has the iterative approach to perform change role and take away role
    
    # loop 1 : iterativly perform 1) takeAwayRole 2) changeRole for all the PSO particle elements
    for new_pos , clid in enumerate(pso_particle) : 
        for i in range(len(Client_list)) : 
            if Client_list[i].label == list(Role_dictionary.keys())[new_pos] :
                takeAwayRole(i)

        for i in range(len(Client_list)) : 
            if Client_list[i].client_id == clid : 
                changeRole(i , new_pos)

    for i in range(len(Client_list)) : 
        if Client_list[i].label == None :
            Client_list[i].label = Role_buffer.pop()    
            Client_list[i].is_aggregator = False 
    
    for i in range(len(Client_list)) : 
        if Client_list[i].is_aggregator : 
            if len(Client_list[i].processing_buffer) == 0 : 
                temp = Role_dictionary[Client_list[i].label]
                for role in temp : 
                    for c in Client_list :
                        if  c.label == role : 
                            Client_list[i].processing_buffer.append(c) 
                        
    for i in range(len(Client_list)) :
        if Client_list[i].label == list(Role_dictionary.keys())[0] :
            return Client_list[i]
        
    # loop 2 : traverse the Client_list to find clients with no roles then associate the trainer roles in the buffer to the selected client

    # loop 3 : traverse the Client_list one last time and associates client with proper roles according to the role dictionary to the processing buffer of clients that are the aggregators

    pass 


def reLabelClients() :
    pass


def main() :
    root = generate_hierarchy(depth=DEPTH, width=WIDTH)
    
    # # Display the tree
    # print("Generated Hierarchy Tree:")
    print_tree(root)
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    # print([client.client_id for client in Client_list])

    Evaluator.calculate_processing_cost(root)
    root = reArrangeHierarchy()
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    print_tree(root)
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    Evaluator.calculate_processing_cost(root)

    # print(Role_dictionary)

    # print([f"{client.label}" for client in Client_list])

if __name__ == "__main__" :
    main()





#   * According to the labels dictionary we rearrange the hierarchy , when the changeRole function is invoked the asscociated label value of the client changes. this necesitates rearranging the hierarchy 
#   * The rule in the rearrangement is that no two clients should have the same label. to ensure that before we allocate the new label to the new selected client we search for the client that has that label and remove that label from that client then we associate to the newly selected client.
#   * After associating the new labels to all the selected clients we traverse again the list of clients and search for the client that has no label.
#   * The process of rearrangement is two folds the first fold is to change the label of the clients. then we traverse the list of clients and update the processing_buffer according to the clients label and the role dictionary. this means that for each client we first look at the label then according to the role dictionary identify the clients that have the roles as the children to that specific role
#   No two IDs must be identical in the PSO particle