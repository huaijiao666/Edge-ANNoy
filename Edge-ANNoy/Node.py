import numpy as np
import h5py
class Node:
    def __init__(self, data_ids=None):
        self.data_ids = data_ids 
        self.left_child = None  
        self.right_child = None  
        self.nil = None
        self.order = None
       