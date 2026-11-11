import sys

import pathology_reconstruction_experiment as pre


if __name__ == "__main__":
    pre.run_model_interface_cli("graphsage", sys.argv[1:])
