setup: |-
    CALL link_prediction.set_model_parameters({hidden_features_size: [3, 2], split_ratio: 1.0, target_relation: ["PAPER:BLACK", "CITES", "PAPER:BLACK"], num_epochs: 1, add_reverse_edges: False}) YIELD *

queries: 
    - |-
        CREATE (v1:PAPER:BLACK {id: 10, features: [1, 2, 3]});
        CREATE (v2:PAPER:BLACK {id: 11, features: [1.54, 0.3, 1.78]});
        CREATE (v3:PAPER:BLACK {id: 12, features: [0.5, 1, 4.5]});
        MATCH (v1:PAPER:BLACK {id: 10}), (v2:PAPER:BLACK {id: 11}) CREATE (v1)-[e:CITES {}]->(v2);
        MATCH (v2:PAPER:BLACK {id: 11}), (v3:PAPER:BLACK {id: 12}) CREATE (v2)-[e:CITES {}]->(v3);

cleanup: |-
    CALL link_prediction.reset_parameters() YIELD *;