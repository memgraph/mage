use crate::graphs::Graph;
use crate::graphs::Vertex;
use rsmgp_sys::memgraph::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::os::raw::c_int;
use std::panic;

fn initialize_p(node_list: &[i64], num_particles: f32) -> HashMap<i64, f32> {
    // initialize map of node_id -> num_nodes by distributing particles equally
    // among all source nodes.
    node_list
        .iter()
        .map(|&n| (n, (1.0 / node_list.len() as f32) * num_particles))
        .collect()
}

fn compute_node_neighbors<G: Graph>(graph: &G, starting_node: &Vertex) -> BinaryHeap<Neighbour> {

    // let total_weight = graph.out_edges(starting_node).unwrap().len() as f32;
    // todo: correctly calculate total weight of outgoing edges for starting_node, but that will
    // require impolementing an out_edges method for the Vertex struct
    let total_weight = 8.0;
    // graph
    //     .out_edges(starting_node)
    //     .unwrap()
    //     .iter()
    //     .map(|edge| Node {
    //         id: edge.target.id(),
    //         score: 1.0 / total_weight,
    //     })
    //     .collect()

    //
    let neighbors = graph.neighbors(*starting_node).unwrap();
    let total_weight = neighbors.len() as f32;
    neighbors
        .iter()
        .map(|node| Neighbour {
            id: node.id,
            score: (1.0 / total_weight as f32),
        })
        .collect()
}

fn filter_ppr_nodes(v: &HashMap<i64, f32>, min_threshold: f32) -> HashMap<i64, f32> {
    let mut ppr_nodes = HashMap::new();
    for (node, score) in v {
        if *score >= min_threshold {
            ppr_nodes.insert(*node, *score);
        }
    }
    ppr_nodes
}

pub fn particle_filtering<G: Graph>(
    graph: G,
    node_list: &[i64],
    min_threshold: f32,
    num_particles: f32,
) -> HashMap<i64, f32> {

    // todo: make optional parameters
    let c = 0.15;
    let tau = 1.0 / num_particles;

    // initialize the probability distribution by distributing particles equally
    let mut p = initialize_p(node_list, num_particles);
    let mut v = p.clone();

    while !p.is_empty() {
        // map to store the number of particles per node
        let mut aux = HashMap::new();

        for (starting_node_id, n_particles) in &p {

            // c is the probability of teleporting to a random node, so 1-c of the particles continue
            let mut n_particles = n_particles * (1.0 - c);
            // if there are no more particles to pass (less than tau threshold), break the loop
            if n_particles <= tau {
                break;
            }

            // retrieve the vertex object for starting_node, so we can get its neighbors
            let starting_node = graph.get_vertex_by_id(*starting_node_id).unwrap();
            // compute neighours and normalize their scores
            let mut neighbors = compute_node_neighbors(&graph, &starting_node);

            // todo: use a heap to get the node with the highest score
            while let Some(node) = neighbors.pop() {
                // if there are no more particles to pass (less than tau threshold), break the loop
                if n_particles <= tau {
                    break;
                }
                // pass particles to neighbors, based on their score
                let weight = node.score;
                let mut passing = n_particles * weight;

                // threshold the number of particles that can be passed to a node as tau
                if passing <= tau {
                    passing = tau;
                }

                n_particles -= passing;

                // check if aux already has an entry for node.id, if so, add passing to the existing value
                if let Some(n_particles) = aux.get_mut(&node.id) {
                    *n_particles += passing;
                } else {
                    aux.insert(node.id, passing);
                }

            }
        }

        p = aux;

        // accumulate the number of particles per node over all iterations. Can't be stored in p, because p is reset
        for (node, n_particles) in &p {
            *v.entry(*node).or_insert(0.0) += n_particles;
        }
    }

    // filter out nodes with score below min_threshold
    let ppr_nodes = filter_ppr_nodes(&v, min_threshold);

    ppr_nodes
}

// Neighbour struct is used to store the id and score of a neighbouring node. The trait
// implementations are used to allow sorting the neighbours by their score efficiently using a
// BinaryHeap.
pub struct Neighbour {
    pub id: i64,
    pub score: f32,
}

impl Ord for Neighbour {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap().reverse()
    }
}

impl PartialOrd for Neighbour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Neighbour {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for Neighbour {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::particle_filtering;
    use crate::graphs::PetgraphGraphWrapper;
    use petgraph::prelude::*;
    use petgraph::Graph as PetgraphGraph;

    #[test]
    fn test_particle_filtering() {
        let mut graph = PetgraphGraph::new();
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");
        graph.add_edge(a, b, 1.0);
        graph.add_edge(a, c, 2.0);
        graph.add_edge(b, d, 3.0);
        graph.add_edge(c, d, 4.0);

        let mut petgraph = PetgraphGraphWrapper::new(&mut graph);

        let node_list = vec![0, 2];
        let min_threshold = 0.05;
        let num_particles = 10.0;

        let result = particle_filtering(petgraph, &node_list, min_threshold, num_particles);

        println!("{:?}", result);
        assert_eq!(result.len(), 4);
        // assert!(result.contains_key(&0));
        // assert!(result.contains_key(&1));
        // assert!(result.contains_key(&2));
        // assert!(result.contains_key(&4));

        // todo: add test that actually does something
    }
}
