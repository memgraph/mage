use c_str_macro::c_str;
use rayon::prelude::*;
use rsmgp_sys::memgraph::*;
use rsmgp_sys::result::Error as MgpError;
use rsmgp_sys::value::*;
use std::io;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vertex {
    pub id: i64,
}

#[derive(Debug)]
pub enum GraphError {
    IoError(io::Error),
    MgpError(MgpError),
}

impl From<io::Error> for GraphError {
    fn from(error: io::Error) -> Self {
        Self::IoError(error)
    }
}

impl From<MgpError> for GraphError {
    fn from(error: MgpError) -> Self {
        Self::MgpError(error)
    }
}

pub trait Graph {
    fn vertices_iter(&self) -> Result<Vec<Vertex>, GraphError>;
    fn neighbors(&self, vertex: Vertex) -> Result<Vec<Vertex>, GraphError>;
    fn weighted_neighbors(&self, vertex: Vertex) -> Result<Vec<(Vertex, f64)>, GraphError>;
    fn add_vertex(&mut self, vertex: Vertex) -> Result<(), GraphError>;
    fn add_edge(&mut self, source: Vertex, target: Vertex, weight: f32) -> Result<(), GraphError>;
    fn num_vertices(&self) -> usize;
    fn get_vertex_by_id(&self, id: i64) -> Option<Vertex>;
    fn outgoing_edges(&self, vertex: Vertex) -> Result<Vec<(Vertex, f64)>, GraphError>;
    fn incoming_edges(&self, vertex: Vertex) -> Result<Vec<(Vertex, f64)>, GraphError>;
}

pub struct MemgraphGraph<'a> {
    graph: &'a Memgraph,
}

impl<'a> MemgraphGraph<'a> {
    pub fn from_graph(graph: &'a Memgraph) -> Self {
        Self { graph }
    }
}

impl<'a> Graph for MemgraphGraph<'a> {
    fn vertices_iter(&self) -> Result<Vec<Vertex>, GraphError> {
        let vertices_iter = self.graph.vertices_iter()?;
        let vertices: Vec<_> = vertices_iter.map(|v| Vertex { id: v.id() }).collect();
        Ok(vertices)
    }

    fn incoming_edges(&self, vertex: Vertex) -> Result<Vec<(Vertex, f64)>, GraphError> {
        let vertex_mgp = self.graph.vertex_by_id(vertex.id)?;
        let iter = vertex_mgp.in_edges()?.map(|e| {
            let target_vertex = e.from_vertex().unwrap();
            // if the vertex doesn't have a weight, we assume it's 1.0
            let weight = e
                .property(&c_str!("weight"))
                .ok()
                .and_then(|p| {
                    if let Value::Float(f) = p.value {
                        Some(f)
                    } else {
                        None
                    }
                })
                .unwrap_or(1.0);

            Ok::<(Vertex, f64), GraphError>((
                Vertex {
                    id: target_vertex.id(),
                },
                weight,
            ))
            .unwrap()
        });
        let incoming_edges: Vec<_> = iter.collect();
        Ok(incoming_edges)
    }

    fn outgoing_edges(&self, vertex: Vertex) -> Result<Vec<(Vertex, f64)>, GraphError> {
        let vertex_mgp = self.graph.vertex_by_id(vertex.id)?;
        let outgoing_edges_iter = vertex_mgp.out_edges()?.map(|e| {
            let target_vertex = e.to_vertex().unwrap();
            // if the vertex doesn't have a weight, we assume it's 1.0
            let weight = e
                .property(&c_str!("weight"))
                .ok()
                .and_then(|p| {
                    if let Value::Float(f) = p.value {
                        Some(f)
                    } else {
                        None
                    }
                })
                .unwrap_or(1.0);

            Ok::<(Vertex, f64), GraphError>((
                Vertex {
                    id: target_vertex.id(),
                },
                weight,
            ))
            .unwrap()
        });
        let outgoing_edges: Vec<_> = outgoing_edges_iter.collect();
        Ok(outgoing_edges)
    }

    fn weighted_neighbors(&self, vertex: Vertex) -> Result<Vec<(Vertex, f64)>, GraphError> {
        let mut outgoing_edges = self.outgoing_edges(vertex).unwrap();
        let incoming_edges = self.incoming_edges(vertex).unwrap();

        outgoing_edges.extend(incoming_edges);

        Ok(outgoing_edges)
    }

    fn neighbors(&self, vertex: Vertex) -> Result<Vec<Vertex>, GraphError> {
        let mut neighbors = vec![];
        let vertex_mgp = self.graph.vertex_by_id(vertex.id)?;
        let neighbors_iter = vertex_mgp.out_edges()?.map(|e| e.to_vertex());
        for neighbor_mgp in neighbors_iter {
            neighbors.push(Vertex {
                id: neighbor_mgp?.id(),
            });
        }
        let neighbors_in = vertex_mgp.in_edges()?.map(|e| e.from_vertex());
        for neighbor_mgp in neighbors_in {
            neighbors.push(Vertex {
                id: neighbor_mgp?.id(),
            });
        }
        Ok(neighbors)
    }

    fn add_vertex(&mut self, _vertex: Vertex) -> Result<(), GraphError> {
        !unimplemented!()
    }

    fn add_edge(
        &mut self,
        _source: Vertex,
        _target: Vertex,
        _weight: f32,
    ) -> Result<(), GraphError> {
        // let source_mgp = self.graph.vertex_by_id(source.id)?;
        // let target_mgp = self.graph.vertex_by_id(target.id)?;
        // self.graph.create_edge(source_mgp, target_mgp, weight)?;
        // Ok(())
        !unimplemented!()
    }

    fn num_vertices(&self) -> usize {
        self.graph.vertices_iter().unwrap().count()
    }

    fn get_vertex_by_id(&self, id: i64) -> Option<Vertex> {
        match self.graph.vertex_by_id(id) {
            Ok(_) => Some(Vertex { id }),
            Err(_) => None,
        }
    }
}

pub fn example<G: Graph>(graph: G, node_list: &[i64]) -> Vec<i64> {
    node_list
        .par_iter()
        .filter_map(|&node_id| graph.get_vertex_by_id(node_id))
        .flat_map(|node| graph.neighbors(node).unwrap_or_else(|_| Vec::new()))
        .map(|vertex| vertex.id)
        .collect()
}
