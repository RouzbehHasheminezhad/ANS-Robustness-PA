import pickle
import shutil
import logging
from engine.config.config import *


def mkdir_(dir_):
    if not os.path.isdir(dir_):
        os.mkdir(dir_)
    else:
        shutil.rmtree(dir_)
        os.mkdir(dir_)


def mkdir(dir_):
    if not os.path.isdir(dir_):
        os.mkdir(dir_)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def reset_logger():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


def set_logger(file_name):
    reset_logger()
    logging.basicConfig(filename=get_log_dir() + "/" + file_name,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filemode="w",
                        level=logging.INFO)


def remove_dir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)


def log_initial_parameters():
    if get_working_dir().endswith("basic/"):
        mkdir(get_working_dir()[:-len("basic/")])
    else:
        mkdir(get_working_dir()[:-len("extended/")])
    mkdir_(get_working_dir())
    mkdir_(get_log_dir())
    set_logger("initial_params.log")
    logging.info("num_engines: %s", get_num_engines())
    logging.info("data_dir: %s", get_data_basic_dir())
    logging.info("log_dir: %s", get_log_dir())
    logging.info("num_reps: %s", get_num_sampled_graphs())
    logging.info("max_tries: %s", get_max_tries())
    logging.info("seed: %s", get_seed())
    reset_logger()
