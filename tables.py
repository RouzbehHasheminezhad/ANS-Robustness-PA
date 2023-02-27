import warnings

from engine.utils.io import *
from engine.utils.tables import generate_table_1, generate_table_2


def generate_tables():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mkdir_(os.getcwd() + "/results/tables/")
        generate_table_1()
        generate_table_2()


if __name__ == '__main__':
    generate_tables()
