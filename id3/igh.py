import csv
import math
import sys
import random


class Node:

    def __init__(self, attribute_index, label):
        self.positive = None
        self.negative = None
        self.attribute_index = attribute_index
        self.label = label
        self.examples = None

    def clone(self):
        clone = Node(self.attribute_index, self.label)
        clone.examples = self.examples
        if isinstance(self.positive, Node):
            clone.negative = self.negative.clone()
            clone.positive = self.positive.clone()
        return clone

    def PrintTree(self, attribute_names):
        if isinstance(self.positive, Node):
            print()
            print("{} = 0 : ".format(attribute_names[self.attribute_index]), end="", flush=True)
            self.negative.PrintTreeRecursive(1, attribute_names)
            print()
            print("{} = 1 : ".format(attribute_names[self.attribute_index]), end="", flush=True)
            self.positive.PrintTreeRecursive(1, attribute_names)
            print()
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

    def non_leaf_nodes(self):
        non_leaf_nodes = []
        if isinstance(self.positive, Node):
            non_leaf_nodes.append(self)
            non_leaf_nodes += self.negative.non_leaf_nodes()
            non_leaf_nodes += self.positive.non_leaf_nodes()
        return non_leaf_nodes

    def predictValue(self, row):
        if isinstance(self.positive, Node):
            if row[self.attribute_index] == '1':
                return self.positive.predictValue(row)
            else:
                return self.negative.predictValue(row)
        else:
            return self.label


def entropy(my_list):
    size = len(my_list)

    if size == 0:
        return 0

    pos_size = sum(1 for i in my_list if i[20] == '1')
    neg_size = size - pos_size
    pos_over_size = pos_size / size
    neg_over_size = neg_size / size
    a = 0 if pos_over_size == 0 else math.log(pos_over_size, 2)
    b = 0 if neg_over_size == 0 else math.log(neg_over_size, 2)
    return - (pos_over_size * a) - (neg_over_size * b)


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
        node.examples = examples
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


def accuracy(d_tree, examples):
    size = len(examples)
    num_right = 0
    for x in examples:
        if int(x[20]) == d_tree.predictValue(x):
            num_right += 1
    return num_right / size


def post_pruning(d_tree, l, k, validation_examples):
    best = d_tree.clone()
    best_accuracy = accuracy(best, validation_examples)
    for i in range(0, l + 1):
        d_prime = best.clone()

        m = random.randint(0, k)
        for j in range(0, m + 1):
            non_leaf_nodes = d_prime.non_leaf_nodes()

            if len(non_leaf_nodes) > 0:
                the_chosen_one = non_leaf_nodes[random.randint(0, len(non_leaf_nodes) - 1)]

                the_chosen_one.positive = None
                the_chosen_one.negative = None
                the_chosen_one.attribute_index = None

                size = len(the_chosen_one.examples)
                pos_size = sum(1 for i in the_chosen_one.examples if i[20] == '1')
                neg_size = size - pos_size

                if pos_size > neg_size:
                    the_chosen_one.label = 1
                else:
                    the_chosen_one.label = 0

        d_prime_accuracy = accuracy(d_prime, validation_examples)
        if d_prime_accuracy > best_accuracy:
            best_accuracy = d_prime_accuracy
            best = d_prime
    return best


l = sys.argv[1]
k = sys.argv[2]
training_set_file = sys.argv[3]
validation_set_file = sys.argv[4]
test_set_file = sys.argv[5]
to_print = sys.argv[6]

with open(training_set_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    training_examples = list(csv_reader)
    attribute_names = training_examples.pop(0)
    attribute_names.remove('Class')

with open(validation_set_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    validation_examples = list(csv_reader)
    validation_examples.pop(0)

with open(test_set_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    test_examples = list(csv_reader)
    test_examples.pop(0)

d_tree = id3(training_examples, list(range(0, 20)))
d_tree.PrintTree(attribute_names)

post_pruned_d_tree = post_pruning(d_tree, int(l), int(k), validation_examples)

if to_print == "yes":
    # d_tree.PrintTree(attribute_names)
    post_pruned_d_tree.PrintTree(attribute_names)

print("accuracy on test examples BEFORE post pruning : {}".format(accuracy(d_tree, test_examples)))
print("accuracy on test examples AFTER  post pruning : {}".format(accuracy(post_pruned_d_tree, test_examples)))
