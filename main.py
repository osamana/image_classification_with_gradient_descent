import tkinter as tk
from tkinter import ttk
from nn import debug_back_propagation_algorithm
from utils import generate_training_data
import threading
import time


def nn(n_hidden_layers, n_neurons_per_layer,
       learning_rate, n_epochs, activation_function):
    start_time = time.time()
    # run the back propagation algorithm
    scores = debug_back_propagation_algorithm(
        "training_data.csv", learning_rate, n_epochs, n_hidden_layers, n_neurons_per_layer, activation_function)
    end_time = time.time()
    elspsed_time = end_time - start_time
    result = 'Scores: %s' % scores + '\n'
    result += 'Mean Accuracy: %.3f%%' % (sum(scores) /
                                         float(len(scores))) + '\n'
    result += 'Elapsed Time: %.3f seconds' % (elspsed_time) + '\n'
    return result


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("A.I. 2nd Project")

        self.result = tk.StringVar()

        # root frame for padding only
        root_frame = tk.Frame(self, padx=10, pady=10)
        root_frame.pack(fill=tk.BOTH, expand=True)

        self.frm_main = tk.Frame(root_frame)
        self.frm_actions = tk.Frame(root_frame, padx=10, pady=10)
        self.frm_credit = tk.Frame(root_frame, padx=10, pady=10)

        self.frm_widgets = tk.Frame(master=self.frm_main, width=100,
                                    height=100, padx=10, pady=10)
        self.frm_info = tk.Frame(master=self.frm_main, relief=tk.RAISED,
                                 borderwidth=1, width=100, height=100, padx=10, pady=10)
        # make self.frm_info have a dark background
        # self.frm_info["background"] = "#000000"

        self.frm_main.pack(fill=tk.BOTH, expand=True)
        self.frm_actions.pack(fill=tk.X, expand=True)
        self.frm_credit.pack(fill=tk.X, expand=True)

        # add credits text to the frm_credit frame
        self.lbl_credit = ttk.Label(master=self.frm_credit, anchor=tk.W,
                                    text="Student: Osama Abu Omar (12255288)\nModule: A.I. A Modern Approach (2nd Project)\nDate: 2022-Dec-5", justify=tk.LEFT)
        self.lbl_credit.pack(side="top", fill=tk.X)
        # lbl_credit["font"] = ("TkDefaultFont", 14, "bold")

        self.frm_widgets.pack(fill=tk.BOTH, expand=False, side=tk.LEFT)
        self.frm_info.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # add info text to the frm_info frame
        self.lbl_info = ttk.Label(master=self.frm_info, anchor=tk.W,
                                  textvariable=self.result, justify=tk.LEFT)
        self.lbl_info.pack(side="top", fill=tk.X)
        # lbl_info["font"] = ("TkDefaultFont", 14, "bold")

        # heading for widgets
        self.lbl_widgets = ttk.Label(
            self.frm_widgets, anchor=tk.W, text="Attributes")
        self.lbl_widgets.pack(side="top", fill=tk.X)
        # same font as heading
        self.lbl_widgets["font"] = ("TkDefaultFont", 10, "bold")

        self.lbl_hidden_layers = ttk.Label(self.frm_widgets, anchor=tk.W)
        self.lbl_hidden_layers["text"] = "Number of hidden layers"
        self.lbl_hidden_layers.pack(side="top", fill=tk.X)
        # the text entry box
        self.ent_hidden_layers_entry = ttk.Entry(self.frm_widgets)
        self.ent_hidden_layers_entry.pack(side="top", fill=tk.X)
        self.ent_hidden_layers_entry.insert(0, "1")

        self.lbl_neurons_per_layer = ttk.Label(self.frm_widgets, anchor=tk.W)
        self.lbl_neurons_per_layer["text"] = "Number of nodes per hidden layer"
        self.lbl_neurons_per_layer.pack(side="top", fill=tk.X)

        # the text entry box
        self.ent_neurons_per_layer_entry = ttk.Entry(self.frm_widgets)
        self.ent_neurons_per_layer_entry.pack(side="top", fill=tk.X)
        self.ent_neurons_per_layer_entry.insert(0, "22")

        self.lbl_learning_rate = ttk.Label(self.frm_widgets, anchor=tk.W)
        self.lbl_learning_rate["text"] = "Learning rate"
        self.lbl_learning_rate.pack(side="top", fill=tk.X)
        # the text entry box, float
        self.ent_learning_rate_entry = ttk.Entry(self.frm_widgets)
        self.ent_learning_rate_entry.pack(side="top", fill=tk.X)
        self.ent_learning_rate_entry.insert(0, "0.3")

        self.lbl_epochs = ttk.Label(self.frm_widgets, anchor=tk.W)
        self.lbl_epochs["text"] = "Number of epochs"
        self.lbl_epochs.pack(side="top", fill=tk.X)
        # the text entry box
        self.ent_epochs_entry = ttk.Entry(self.frm_widgets)
        self.ent_epochs_entry.pack(side="top", fill=tk.X)
        self.ent_epochs_entry.insert(0, "600")

        # activation function select, tanh, relu
        self.lbl_activation_function = ttk.Label(
            self.frm_widgets, anchor=tk.W)
        self.lbl_activation_function["text"] = "Activation function"
        self.lbl_activation_function.pack(side="top", fill=tk.X)

        # make it a drop down menu, tanh, relu. don't allow writing to it
        self.ent_activation_function_select = ttk.Combobox(
            self.frm_widgets, values=["tanh", "relu", "sigmoid"])
        self.ent_activation_function_select.current(0)

        self.ent_activation_function_select.pack(side="top", fill=tk.X)

        # generate training data button
        self.btn_generate_training_data = ttk.Button(
            self.frm_actions, text="Generate training data")
        self.btn_generate_training_data["text"] = "Generate training data"
        self.btn_generate_training_data["command"] = generate_training_data
        self.btn_generate_training_data.pack(side="left")

        # start button
        self.btn_run = ttk.Button(self.frm_actions, text="Train")
        self.btn_run["text"] = "Train"
        self.btn_run["command"] = self.run
        self.btn_run.pack(side="left")

        # close button
        self.btn_close = ttk.Button(
            self.frm_actions, text="Close", command=self.close)
        self.btn_close.pack(side="left")

    def close(self):
        self.destroy()

    def run(self):
        # print running...
        self.result.set("Running..."+'\n')
        # Start the process in a separate thread
        thread = threading.Thread(target=self.run_process)
        thread.start()

    def run_process(self):

        # get the values from the widgets
        n_hidden_layers = int(self.ent_hidden_layers_entry.get())  # 1
        n_neurons_per_layer = int(self.ent_neurons_per_layer_entry.get())  # 22
        learning_rate = float(self.ent_learning_rate_entry.get())  # 0.3
        # 600 (!iterations -> all dataset)
        n_epochs = int(self.ent_epochs_entry.get())
        activation_function = self.ent_activation_function_select.get()  # sigmoid

        # run the back propagation algorithm
        # Run the process and get the result
        result = nn(n_hidden_layers, n_neurons_per_layer,
                    learning_rate, n_epochs, activation_function)
        # Update the label with the result
        # self.result.set(result)
        self.after(0, self.update_ui, result)

    def update_ui(self, result):
        # Update the label with the result
        self.result.set(result)


if __name__ == "__main__":
    app = Application()
    app.mainloop()
