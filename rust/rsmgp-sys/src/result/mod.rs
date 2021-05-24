use snafu::Snafu;

#[derive(Debug, PartialEq, Snafu)]
#[snafu(visibility = "pub")]
pub enum MgpError {
    #[snafu(display("An error inside Rust procedure."))]
    MgpDefaultError,
    #[snafu(display("Unable to allocate memory inside Rust procedure."))]
    MgpAllocationError,
    #[snafu(display("Unable to prepare result within Rust procedure."))]
    MgpPreparingResultError,
    #[snafu(display("Unable to add a type of procedure parameter in Rust Module."))]
    MgpAddProcedureParameterTypeError,
    #[snafu(display("Out of bound label index."))]
    MgpOutOfBoundLabelIndex,
    #[snafu(display("Unable to create Rust CString based on Memgraph value."))]
    MgpCreationOfCStringError,
    #[snafu(display("Unable to create Rust Vertex based on Memgraph value."))]
    MgpCreationOfVertexError,
    #[snafu(display("Unable to create Rust Edge based on Memgraph value."))]
    MgpCreationOfEdgeError,
    #[snafu(display("Unable to create result Memgraph Vertex (allocation error)."))]
    MgpResultVertexAllocationError,
}

pub type MgpResult<T, E = MgpError> = std::result::Result<T, E>;
