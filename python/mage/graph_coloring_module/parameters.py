from enum import Enum


class Parameter(Enum):
    ALGORITHM = "algorithm"
    NUM_OF_COLORS = "num_of_colors"

    NUM_OF_PROCESSES = "num_of_processes"

    POPULATION_SIZE = "population_size"
    POPULATION_FACTORY = "population_factory"
    INIT_ALGORITHMS = "init_algorithms"

    ERROR = "error"
    MAX_ITERATIONS = "max_iterations"
    ITERATION_CALLBACKS = "iteration_callbacks"
    COMMUNICATION_DALAY = "communication_delay"
    LOGGING_DELAY = "logging_delay"

    QA_TEMPERATURE = "QA_temperature"
    QA_MAX_STEPS = "QA_max_steps"
    CONFLICT_ERR_ALPHA = "conflict_err_alpha"
    CONFLICT_ERR_BETA = "conflict_err_beta"

    MUTATION = "mutation"
    MULTIPLE_MUTATION_NODES_NUM_OF_NODES = "multiple_mutation_num_of_nodes"
    RANDOM_MUTATION_PROBABILITY = "random_mutation_probability"

    SIMPLE_TUNNELING_MUTATION = "simple_tunneling_mutation"
    SIMPLE_TUNNELING_PROBABILITY = "simple_tunneling_probability"
    SIMPLE_TUNNELING_ERROR_CORRECTION = "simple_tunneling_error_correction"
    SIMPLE_TUNNELING_MAX_ATTEMPTS = "simple_tunneling_max_attempts"

    CONVERGENCE_CALLBACK_TOLERANCE = "convergence_callback_tolerance"
    CONVERGENCE_CALLBACK_ACTIONS = "convergence_callback_actions"
