import csv
import math
import sys


class Node:

    def __init__(self, attribute_index, label):
        self.positive = None
        self.negative = None
        self.attribute_index = attribute_index
        self.label = label

    def PrintTree(self, attribute_names):
        if isinstance(self.positive, Node):
            print()
            print("{} = 0 : ".format(attribute_names[self.attribute_index]), end="", flush=True)
            self.negative.PrintTreeRecursive(1, attribute_names)
            print()
            print("{} = 1 : ".format(attribute_names[self.attribute_index]), end="", flush=True)
            self.positive.PrintTreeRecursive(1, attribute_names)
        else:
            print(self.label)

    def PrintTreeRecursive(self, level, attribute_names):
        if isinstance(self.positive, Node):
            print()
            print("{}{} = 0 : ".format("| " * level, attribute_names[self.attribute_index]), end="", flush=True)
            self.negative.PrintTreeRecursive(level + 1, attribute_names)
            print()
            print("{}{} = 1 : ".format("| " * level, attribute_names[self.attribute_index]), end="", flush=True)
            self.positive.PrintTreeRecursive(level + 1, attribute_names)
        else:
            print(self.label, end="", flush=True)


def entropy(my_list):
    size = len(my_list)

    if size == 0:
        return 0

    pos_size = sum(1 for i in my_list if i[20] == '1')
    neg_size = size - pos_size

    return pos_size * neg_size / size


def gain_sans_entropy(my_list, attribute_index):
    size = len(my_list)

    pos_attr_list = []
    neg_attr_list = []

    for x in my_list:
        if x[attribute_index] == '1':
            pos_attr_list.append(x)
        else:
            neg_attr_list.append(x)

    return (len(pos_attr_list) * entropy(pos_attr_list) + len(neg_attr_list) * entropy(neg_attr_list)) / size


def best_attribute_index(my_list, attributes):
    gain_list = []
    for i in attributes:
        gain_list.append(gain_sans_entropy(my_list, i))
    return attributes[gain_list.index(min(gain_list))]


def id3(examples, attributes):
    node = Node(None, None)

    size = len(examples)
    pos_size = sum(1 for i in examples if i[20] == '1')
    neg_size = size - pos_size

    if pos_size == size:
        node.label = 1
    elif neg_size == size:
        node.label = 0
    elif len(attributes) == 0:
        if pos_size > neg_size:
            node.label = 1
        else:
            node.label = 0
    else:
        attribute_index = best_attribute_index(examples, attributes)
        node.attribute_index = attribute_index

        pos_attr_list = []
        neg_attr_list = []

        for x in examples:
            if x[attribute_index] == '1':
                pos_attr_list.append(x)
            else:
                neg_attr_list.append(x)

        most_common_target_value = 1 if len(pos_attr_list) > len(neg_attr_list) else 0
        new_attributes = attributes.copy()
        new_attributes.remove(attribute_index)

        if len(pos_attr_list) == 0:
            node.positive = Node(None, most_common_target_value)
        else:
            node.positive = id3(pos_attr_list, new_attributes)

        if len(neg_attr_list) == 0:
            node.negative = Node(None, most_common_target_value)
        else:
            node.negative = id3(neg_attr_list, new_attributes)

    return node


l = sys.argv[1]
k = sys.argv[2]
training_set_file = sys.argv[3]
validation_set_file = sys.argv[4]
test_set_file = sys.argv[5]
to_print = sys.argv[6]

with open(training_set_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    my_list = list(csv_reader)
    attribute_names = my_list.pop(0)
    attribute_names.remove('Class')
    root = id3(my_list, list(range(0, 20)))
    if to_print == "yes":
        root.PrintTree(attribute_names)
