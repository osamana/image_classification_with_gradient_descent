import random
from random import seed, randrange
from csv import reader
from math import exp
import math

# Load a CSV file


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column


def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Rescale dataset columns to the range 0-1


def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Calculate accuracy percentage


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Split a dataset into k folds


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Evaluate an algorithm


def evaluate_algorithm(dataset, algorithm, *args):
    folds = cross_validation_split(dataset, 5)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])  # flatten list of lists
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None  # remove the label
        predicted = algorithm(train_set, test_set, *args)

        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

        # print actual and predicted values for each fold
        print("Actual: ", actual)
        print("Predicted: ", predicted)

    return scores

# Calculate neuron activation for an input


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation


def transfer(activation, activation_function):
    if activation_function == "sigmoid":
        return 1.0 / (1.0 + exp(-activation))
    elif activation_function == "tanh":
        return 2 / (1 + exp(-2 * activation)) - 1  # return 1 - np.tanh(x)**2
    elif activation_function == "relu":
        return max(0, activation)
    else:
        raise ValueError("Activation function not supported")


# Calculate the derivative of an neuron output


def transfer_derivative(output, activation_function):
    if activation_function == "sigmoid":
        return output * (1.0 - output)
    elif activation_function == "tanh":
        return 1 - math.tanh(output)**2
    elif activation_function == "relu":
        return 1 if output > 0 else 0
    else:
        raise ValueError("Activation function not supported")

# Forward propagate input to a network output


def forward_propagate(network, row, activation_function):
    inputs = row
    for i in range(len(network)):
        new_inputs = []
        for neuron in network[i]:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation, activation_function)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Backpropagate error and store in neurons


def backward_propagate_error(network, expected, activation_function):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * \
                transfer_derivative(neuron['output'], activation_function)

# Update network weights with error


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs


def train_network(network, train, l_rate, n_epoch, n_outputs, activation_function):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row, activation_function)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected, activation_function)
            update_weights(network, row, l_rate)

# Initialize a network


def initialize_network(n_inputs, n_hidden_layers, n_neurons_per_hidden_layer, n_outputs):
    network = list()

    # create the first hidden layer
    hidden_layer = [{'weights': [random.random() for i in range(n_inputs + 1)]}
                    for i in range(n_neurons_per_hidden_layer)]
    network.append(hidden_layer)

    # create the remaining hidden layers
    for i in range(n_hidden_layers-1):
        hidden_layer = [{'weights': [random.random() for i in range(n_neurons_per_hidden_layer + 1)]}
                        for i in range(n_neurons_per_hidden_layer)]
        network.append(hidden_layer)

    # create the output layer
    output_layer = [{'weights': [random.random() for i in range(n_neurons_per_hidden_layer + 1)]}
                    for i in range(n_outputs)]
    network.append(output_layer)

    return network


# Make a prediction with a network


def predict(network, row, activation_function):
    outputs = forward_propagate(network, row, activation_function)
    return outputs.index(max(outputs))

# display the network


def display_network(network):
    print("Network: ")
    for layer in network:
        print(layer)

# Backpropagation Algorithm With Stochastic Gradient Descent


def back_propagation(train, test, l_rate, n_epoch, n_hidden_layers, n_neurons_per_layer, activation_function):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(
        n_inputs, n_hidden_layers, n_neurons_per_layer, n_outputs)

    train_network(network, train, l_rate, n_epoch,
                  n_outputs, activation_function)
    predictions = list()
    for row in test:
        prediction = predict(network, row, activation_function)
        predictions.append(prediction)
    return (predictions)


def debug_back_propagation_algorithm(
    file_name, l_rate, n_epoch, n_hidden_layers, n_neurons_per_layer, activation_function
):
    # Test back propagation algorithm
    seed(1)
    # load and prepare data
    dataset = load_csv(file_name)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    # convert class column to integers

    str_column_to_int(dataset, len(dataset[0])-1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    # evaluate algorithm
    # l_rate = 0.3
    # n_epoch = 600
    # activation_function = 'sigmoid'
    # following the rule of thumb for number of hidden layers (mean of input and output layers)
    # n_hidden_layers = 1
    # n_neurons_per_layer = 22  # floor((35 + 10) / 2)
    scores = evaluate_algorithm(
        dataset, back_propagation, l_rate, n_epoch, n_hidden_layers, n_neurons_per_layer, activation_function)

    return scores


if __name__ == "__main__":
    scores = debug_back_propagation_algorithm(
        'training_data.csv', 0.3, 1, 1, 22, 'sigmoid')
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
