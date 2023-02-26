import os

global param_list
global initial_seed
global current_seed
global num_engines
global working_dir
global max_tries
global reps
global num_sampled_graphs


def get_param_list():
    global param_list
    return param_list


def set_param_list(par_list):
    global param_list
    param_list = par_list


def get_seed():
    global initial_seed
    return initial_seed


def set_seed(seed):
    global initial_seed
    initial_seed = seed


def get_num_engines():
    global num_engines
    return num_engines


def set_num_engines():
    global num_engines
    num_engines = os.cpu_count()


def set_num_sampled_graphs(num_reps):
    global num_sampled_graphs
    num_sampled_graphs = num_reps


def get_num_sampled_graphs():
    global num_sampled_graphs
    return num_sampled_graphs


def set_working_dir(working_dir_path):
    global working_dir
    working_dir = working_dir_path


def get_working_dir():
    global working_dir
    return working_dir


def get_data_basic_dir():
    global working_dir
    return working_dir + "data/"


def get_log_dir():
    global working_dir
    return working_dir + "logs/"


def set_max_tries(tries):
    global max_tries
    max_tries = tries


def get_max_tries():
    global max_tries
    return max_tries


def set_num_reps(num_reps):
    global reps
    reps = num_reps


def get_num_reps():
    global reps
    return reps
