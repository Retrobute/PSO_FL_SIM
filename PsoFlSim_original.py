from random import randint
from time import sleep as delay

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

class Trainer:
    def __init__(self, memcap, mdatasize, label):
        self.memcap = memcap
        self.mdatasize = mdatasize
        self.label = label

class AgTrainer(Trainer):
    def __init__(self, pspeed, memcap, mdatasize, label):
        super().__init__(memcap, mdatasize, label)
        self.pspeed = pspeed
        self.processing_buffer = []

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

                if isinstance(current_node, AgTrainer):
                    bfs_queue.extend(current_node.processing_buffer)  

            levels.append(current_level)  

        levels.reverse()

        total_delay = 0

        # Calculate delays level by level
        for level in levels:
            cluster_delays = []  

            for node in level:
                if isinstance(node, AgTrainer):

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

                elif isinstance(node, Trainer):
                    
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
    def create_ag_trainer(level, label_prefix):
        pspeed = randint(5, 15)
        memcap = randint(10, 50)
        mdatasize = randint(5, 20)
        return AgTrainer(pspeed, memcap, mdatasize, f"t{label_prefix}ag{level}")

    def create_trainer(label_prefix):
        memcap = randint(5, 15)
        mdatasize = randint(5, 15)
        return Trainer(memcap, mdatasize, label_prefix)

    # Create the root AgTrainer
    root = create_ag_trainer(0, "1")

    def build_tree(parent, level):
        if level >= depth:
            return

        for i in range(width):
            ag_trainer = create_ag_trainer(level, f"{i+1}")
            parent.processing_buffer.append(ag_trainer)

            # Add binary leaves if we're at the last level
            if level == depth - 1:
                for j in range(2):  # Binary leaves
                    trainer = create_trainer(f"{ag_trainer.label}_{j+1}")
                    ag_trainer.processing_buffer.append(trainer)
            else:
                build_tree(ag_trainer, level + 1)

    build_tree(root, 1)
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
    if isinstance(node , AgTrainer) : 
        print(f"{prefix}{connector}{node.label} (MemCap: {node.memcap}, MDataSize: {node.mdatasize} Pspeed: {node.pspeed}) ")

    elif isinstance(node, Trainer) :
        print(f"{prefix}{connector}{node.label} (MemCap: {node.memcap}, MDataSize: {node.mdatasize})")

    if isinstance(node, AgTrainer):
        for i, child in enumerate(node.processing_buffer):
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(child, level + 1, i == len(node.processing_buffer) - 1, new_prefix)

def main() :
    root = generate_hierarchy(depth=3, width=2)

    # Display the tree
    print("Generated Hierarchy Tree:")
    print_tree(root)
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    Evaluator.calculate_processing_cost(root)


if __name__ == "__main__" :
    main()