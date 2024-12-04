from random import randint
from time import sleep as delay

# Global parameters
Client_list = []


# TODO : Create a Role Dictionary , define processing buffer for each role.

dummy_role_dictionary = {   # width=2 , depth=3
    "t1ag1_1" : [],
    "t1ag1_2" : [],
    "t2ag1_1" : [],
    "t2ag1_2" : [],
    "t1ag1" : ["t1ag1_1" , "t1ag1_2"],
    "t2ag1" : ["t2ag1_1" , "t2ag1_2"],
    "t1ag2" : ["t1ag1" , "t2ag1"],
}

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
agtrainers_count = 4
trainers_count = 2       # Per agtrainer
hlevel = 3

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
    def __init__(self, memcap, mdatasize, client_id , label , pspeed=0 , is_aggregator=False) :
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

# class Trainer:
#     def __init__(self, memcap, mdatasize, label):
#         self.memcap = memcap
#         self.mdatasize = mdatasize
#         self.label = label

# class AgTrainer(Trainer):
#     def __init__(self, pspeed, memcap, mdatasize, label):
#         super().__init__(memcap, mdatasize, label)
#         self.pspeed = pspeed
#         self.processing_buffer = []

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
                    node.mdatasize = cluster_mem  # Update the node's mdatasize for parent-level calculations

                    cluster_delay = cluster_mem / node.pspeed
                    cluster_delays.append(cluster_delay)

                    # Print details for the cluster
                    print(f"AgTrainer: {node.label}, Cluster Mem Consumption: {cluster_mem}, Cluster Delay: {cluster_delay:.2f}")
                    for child in node.processing_buffer:
                        print(f"  Trainer: {child.label}, Mem Consumption: {child.mdatasize}")

                elif node.is_aggregator == False :
                    
                    # Trainers work independently; no cluster delay for them
                    print(f"Trainer: {node.label}, Mem Consumption: {node.mdatasize}")

            # Find the maximum cluster delay for the level
            if cluster_delays:
                max_cluster_delay = max(cluster_delays)
                total_delay += max_cluster_delay  # Add max delay of the level to the total delay
                print(f"Level Max Cluster Delay: {max_cluster_delay:.2f}\n")

        print(f"Total Processing Delay: {total_delay:.2f}")


def generate_hierarchy(depth, width):
    """
    Generate a random hierarchy with binary leaves.

    :param depth: The depth of the tree (number of levels).
    :param width: The number of AgTrainers per level (excluding leaves).
    :return: The root AgTrainer object.
    """
    def create_agtrainer(level, label_prefix):
        pspeed = randint(5, 15)
        memcap = randint(10, 50)
        mdatasize = randint(5, 20)
        new_client = Client(memcap, mdatasize,len(Client_list), f"t{label_prefix}ag{level}", pspeed , True)
        Client_list.append(new_client)
        return new_client

    def create_trainer(label_prefix):
        memcap = randint(5, 15)
        mdatasize = randint(5, 15)
        new_client = Client(memcap, mdatasize,len(Client_list), label_prefix )
        Client_list.append(new_client)
        return new_client

    # def create_ag_client() :
    #     memcap = randint(10, 50)
    #     mdatasize = randint(5, 20)

    # Create the root AgTrainer
    root = create_agtrainer(0 ,"1")

    def generate_clients(parent, level):
        if level >= depth:
            return

        for i in range(width):
            ag_trainer = create_agtrainer(level ,f"{i+1}")
            if parent.is_aggregator :
                parent.processing_buffer.append(ag_trainer)

            # Add binary leaves if we're at the last level
            if level == depth - 1:
                for j in range(2):  # Binary leaves
                    trainer = create_trainer(f"{ag_trainer.label}_{j+1}")
                    ag_trainer.processing_buffer.append(trainer)
            else:
                generate_clients(ag_trainer, level + 1)

    generate_clients(root, 1)
    return root

def print_tree(node, level=0, is_last=True, prefix=""):
    """
    Print the hierarchy tree in a structured format.

    :param node: The current node to print.
    :param level: The depth level of the current node.
    :param is_last: Whether the current node is the last child of its parent.
    :param prefix: The prefix string for the tree structure.
    """

    connector = "└── " if is_last else "├── "
    if node.is_aggregator : 
        print(f"{prefix}{connector}{node.label} (MemCap: {node.memcap}, MDataSize: {node.mdatasize} Pspeed: {node.pspeed}) ")

    elif node.is_aggregator == False :
        print(f"{prefix}{connector}{node.label} (MemCap: {node.memcap}, MDataSize: {node.mdatasize})")

    if node.is_aggregator :
        for i, child in enumerate(node.processing_buffer):
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(child, level + 1, i == len(node.processing_buffer) - 1, new_prefix)

def changeRole(label) :
    pass 

def reArrangeHierarchy() :      #   * According to the labels dictionary we rearrange the hierarchy , when the changeRole function is invoked the asscociated label value of the client changes. this necesitates rearranging the hierarchy 
    for node in Client_list :   #   * The rule in the rearrangement is that no two clients should have the same label. to ensure that before we allocate the new label to the new selected client we search for the client that has that label and remove that label from that client then we associate to the newly selected client.
        pass                    #   * After associating the new labels to all the selected clients we traverse again the list of clients and search for the client that has no label.
                                #   * The process of rearrangement is two folds the first fold is to change the label of the clients. then we traverse the list of clients and update the processing_buffer according to the clients label and the role dictionary. this means that for each client we first look at the label then according to the role dictionary identify the clients that have the roles as the children to that specific role


def main() :
    root = generate_hierarchy(depth=3, width=2)
    
    # Display the tree
    print("Generated Hierarchy Tree:")
    print_tree(root)
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    Evaluator.calculate_processing_cost(root)


if __name__ == "__main__" :
    main()