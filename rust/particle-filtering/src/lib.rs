use rsmgp_sys::memgraph::*;
use rsmgp_sys::result::*;
use rsmgp_sys::map::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use c_str_macro::c_str;
use rsmgp_sys::list::*;
use rsmgp_sys::memgraph::*;
use rsmgp_sys::mgp::*;
use rsmgp_sys::property::*;
use rsmgp_sys::result::*;
use rsmgp_sys::rsmgp::*;
use rsmgp_sys::value::*;
use rsmgp_sys::{close_module, define_optional_type, define_procedure, define_type, init_module};
use std::ffi::CString;
use std::os::raw::c_int;
use std::panic;
use rsmgp_sys::vertex::Vertex;

init_module!(|memgraph: &Memgraph| -> Result<()> {

    memgraph.add_read_procedure(
    particle_filtering,
    c_str!("particle_filtering"),
    &[],
    &[],
    &[
    define_type!("node_id", Type::Int),
    define_type!("score", Type::Double),
    ],
    )?;

    Ok(())
    });

pub struct Node {
    pub id: i64,
    pub score: f32,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap().reverse()
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for Node {}

// pub struct ParticleFiltering<'a> {
//     memgraph: &'a Memgraph,
//     neighbors_dict: HashMap<Vertex, Vec<Vertex>>,
// }

// impl<'a> ParticleFiltering<'a> {
//     pub fn new(memgraph: &'a Memgraph) -> Result<Self> {
//         let mut neighbors_dict = HashMap::new();
//
//         for vertex in memgraph.vertices_iter()? {
//             let neighbors: Vec<Vertex> = vertex
//             .out_edges()?
//             .map(|edge| edge.to_vertex().unwrap())
//             .collect();
//             neighbors_dict.insert(vertex, neighbors);
//         }
//
//         Ok(Self {
//             memgraph,
//             neighbors_dict,
//         })
//     }
//
//     pub fn search(
//     &self,
//     node_list: &[i64],
//     min_threshold: f32,
//     num_particles: f32,
//     ) -> HashMap<i64, f32> {
//         let mut p: HashMap<i64, f32> = node_list
//         .iter()
//         .map(|&n| (n, (1.0 / node_list.len() as f32) * num_particles))
//         .collect();
//         let mut v: HashMap<i64, f32> = p.clone();
//         let c = 0.15;
//         let tao = 1.0 / num_particles;
//         let mut ppr_nodes = HashMap::new();
//
//         while !p.is_empty() {
//             let mut aux = HashMap::new();
//             for (starting_node, particles) in &p {
//                 let mut particles = particles * (1.0 - c);
//                 let total_weight = self.neighbors_dict[starting_node].len() as f32;
//                 let mut neighbors: BinaryHeap<Node> = self.neighbors_dict[starting_node].iter()
//                 .map(|&n| Node {
//                         id: n,
//                         score: 1.0 / total_weight,
//                     })
//                 .collect();
//
//                 while let Some(node) = neighbors.pop() {
//                     let weight = node.score;
//                     let mut passing = particles * weight;
//                     if passing <= tao {
//                         passing = tao;
//                     }
//                     particles -= passing;
//                     if let Some(particles) = aux.get_mut(&node.id) {
//                         *particles += passing;
//                     } else {
//                         aux.insert(node.id, passing);
//                     }
//                     if particles <= tao {
//                         break;
//                     }
//                 }
//             }
//
//             p = aux;
//             for (node, particles) in &p {
//                 *v.entry(*node).or_insert(0.0) += particles;
//             }
//         }
//
//         // filtering v with min_threshold
//         for (node, score) in &v {
//             if *score >= min_threshold {
//                 ppr_nodes.insert(*node, *score);
//             }
//         }
//
//         // returning <ids, scores>
//         ppr_nodes
//     }
// }



// define_procedure!(particle_filtering, |memgraph: &Memgraph| -> Result<()> {
//
//     let p = ParticleFiltering::new(memgraph)?;
//     let node_list = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
//     let min_threshold = 0.0;
//     let num_particles = 1000.0;
//     let ppr_nodes = p.search(&node_list, min_threshold, num_particles);
//
//     let result = memgraph.result_record()?;
//     // create  rsmgp_sys::map::Map
//     let mut map = Map::make_empty(&memgraph)?;
//     for (node, score) in &ppr_nodes {
//         let node_cstring = CString::new(node.to_string()).unwrap();
//
//         map.insert(node_cstring.as_c_str(), &MgpValue::make_double(*score as f64, &memgraph)?.to_value()?);
//     }
//
//
//     result.insert_map(c_str!("output_int"), &map)?;
//
//     // // This procedure just forwards the input parameters as procedure results.
//     // let result = memgraph.result_record()?;
//     // let args = memgraph.args()?;
//     // let input_string = args.value_at(0)?;
//     // let input_int = args.value_at(1)?;
//     // result.insert_mgp_value(
//     //     c_str!("output_string"),
//     //     &input_string.to_mgp_value(&memgraph)?,
//     // )?;
//     // result.insert_mgp_value(c_str!("output_int"), &input_int.to_mgp_value(&memgraph)?)?;
//     Ok(())
// });

define_procedure!(particle_filtering, |memgraph: &Memgraph| -> Result<()> {
    // let node_list: Vec<i64> = vec![49422];
    let node_list = vec![memgraph.vertex_by_id(49422)?];
    let num_particles = 1000.0;
    let min_threshold = 0.0;
    let mut p: HashMap<i64, f32> = node_list
        .iter()
        .map(|n| (n.id(), (1.0 / node_list.len() as f32) * num_particles))
        .collect();

    let mut v: HashMap<i64, f32> = p.clone();
    let c = 0.15;
    let tao = 1.0 / num_particles;

    while !p.is_empty() {
            let mut aux = HashMap::new();
            for (starting_node, particles) in &p {
                let starting_node = memgraph.vertex_by_id(*starting_node)?;
                let mut particles = particles * (1.0 - c);
                let total_weight = starting_node.out_edges()?.count() as f32;
                let mut neighbors: BinaryHeap<Node> = starting_node
                .out_edges()?
                .map(|edge| edge.to_vertex())
                .map(|n| Node {
                        id: n.unwrap().id(),
                        score: 1.0 / total_weight,
                    })
                .collect();

                // let total_weight = self.neighbors_dict[starting_node].len() as f32;

                while let Some(node) = neighbors.pop() {
                    let weight = node.score;
                    let mut passing = particles * weight;
                    if passing <= tao {
                        passing = tao;
                    }
                    particles -= passing;
                    if let Some(particles) = aux.get_mut(&node.id) {
                        *particles += passing;
                    } else {
                        aux.insert(node.id, passing);
                    }
                    if particles <= tao {
                        break;
                    }
                }
            }

            p = aux;
            for (node, particles) in &p {
                *v.entry(*node).or_insert(0.0) += particles;
            }
        }

        // filtering v with min_threshold
        for (node, score) in &v {
            if *score >= min_threshold {
                let result = memgraph.result_record()?;
                result.insert_int(c_str!("node_id"), *node)?;
                result.insert_double(c_str!("score"), *score as f64)?;
            }
        }
    Ok(())
}
);


close_module!(|| -> Result<()> { Ok(()) });

