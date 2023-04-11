use petgraph::prelude::*;
use petgraph::Graph as PetgraphGraph;
use rsmgp_sys::memgraph::*;
use rsmgp_sys::result::Error as MgpError;
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

pub trait Graph {
    fn vertices_iter(&self) -> Result<Vec<Vertex>, GraphError>;
    fn neighbors(&self, vertex: Vertex) -> Result<Vec<Vertex>, GraphError>;
    fn add_vertex(&mut self, vertex: Vertex) -> Result<(), GraphError>;
    fn add_edge(&mut self, source: Vertex, target: Vertex, weight: f32) -> Result<(), GraphError>;
    fn num_vertices(&self) -> usize;
    fn get_vertex_by_id(&self, id: i64) -> Option<Vertex>;
}

pub struct PetgraphGraphWrapper<'a> {
    graph: &'a mut PetgraphGraph<&'a str, f32>,
}

impl<'a> PetgraphGraphWrapper<'a> {
    pub fn new(graph: &'a mut PetgraphGraph<&'a str, f32>) -> Self {
        Self { graph }
    }
}

impl<'a> Graph for PetgraphGraphWrapper<'a> {
    fn vertices_iter(&self) -> Result<Vec<Vertex>, GraphError> {
        let vertices: Vec<_> = self
            .graph
            .node_indices()
            .map(|n| Vertex {
                id: n.index() as i64,
            })
            .collect();
        Ok(vertices)
    }

    fn neighbors(&self, vertex: Vertex) -> Result<Vec<Vertex>, GraphError> {
        let node_index = NodeIndex::new(vertex.id as usize);
        let neighbors: Vec<_> = self
            .graph
            .neighbors(node_index)
            .map(|n| Vertex {
                id: n.index() as i64,
            })
            .collect();
        Ok(neighbors)
    }

    fn add_vertex(&mut self, vertex: Vertex) -> Result<(), GraphError> {
        self.graph.add_node("2");
        Ok(())
    }

    fn add_edge(&mut self, source: Vertex, target: Vertex, weight: f32) -> Result<(), GraphError> {
        let source_index = NodeIndex::new(source.id as usize);
        let target_index = NodeIndex::new(target.id as usize);
        self.graph.add_edge(source_index, target_index, weight);
        Ok(())
    }

    fn num_vertices(&self) -> usize {
        self.graph.node_count()
    }

    fn get_vertex_by_id(&self, id: i64) -> Option<Vertex> {
        self.graph
            .node_indices()
            .find(|i| i.index() as i64 == id)
            .map(|i| Vertex { id })
    }
}

impl From<MgpError> for GraphError {
    fn from(error: MgpError) -> Self {
        Self::MgpError(error)
    }
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

    fn neighbors(&self, vertex: Vertex) -> Result<Vec<Vertex>, GraphError> {
        let mut neighbors = vec![];
        let vertex_mgp = self.graph.vertex_by_id(vertex.id)?;
        let neighbors_iter = vertex_mgp.out_edges()?.map(|e| e.to_vertex());
        for neighbor_mgp in neighbors_iter {
            neighbors.push(Vertex {
                id: neighbor_mgp?.id(),
            });
        }
        Ok(neighbors)
    }

    fn add_vertex(&mut self, vertex: Vertex) -> Result<(), GraphError> {
        !unimplemented!()
    }

    fn add_edge(&mut self, source: Vertex, target: Vertex, weight: f32) -> Result<(), GraphError> {
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
            Ok(vertex_mgp) => Some(Vertex { id }),
            Err(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph;
    use petgraph::Graph as PetgraphGraph;

    #[test]
    fn test_petgraph_graph() {
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

        let vertices = petgraph.vertices_iter().unwrap();
        assert_eq!(vertices.len(), 4);

        let neighbors_a = petgraph
            .neighbors(Vertex {
                id: a.index() as i64,
            })
            .unwrap();
        assert_eq!(neighbors_a.len(), 2);
        assert!(neighbors_a.iter().any(|n| n.id == b.index() as i64));
        assert!(neighbors_a.iter().any(|n| n.id == c.index() as i64));

        let neighbors_b = petgraph
            .neighbors(Vertex {
                id: b.index() as i64,
            })
            .unwrap();
        assert_eq!(neighbors_b.len(), 1);
        assert!(neighbors_b.iter().any(|n| n.id == d.index() as i64));

        petgraph.add_vertex(Vertex { id: 4 }).unwrap();
        assert_eq!(petgraph.vertices_iter().unwrap().len(), 5);
    }
}
