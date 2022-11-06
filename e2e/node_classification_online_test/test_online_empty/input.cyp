setup: |-
    CALL mg.load("node_classification");
    CALL node_classification.set_model_parameters() YIELD *;

queries: |-
    

cleanup: |-
    CALL node_classification.reset() YIELD *;