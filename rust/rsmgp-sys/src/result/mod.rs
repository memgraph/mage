use snafu::Snafu;

// TODO(gitbuda): Polish error messages.

#[derive(Debug, PartialEq, Snafu)]
#[snafu(visibility = "pub")]
pub enum MgpError {
    #[snafu(display("Unable to insert map value."))]
    UnableToInsertMapValue,

    #[snafu(display("Unable to access map value."))]
    UnableToAccessMapValue,

    #[snafu(display("Unable to create map iterator."))]
    UnableToCreateMapIterator,

    #[snafu(display("Unable to create map."))]
    UnableToCreateMap,

    #[snafu(display("Unable to create empty map."))]
    UnableToCreateEmptyMap,

    #[snafu(display("Unable to create empty list."))]
    UnableToCreateEmptyList,

    #[snafu(display("Unable to copy list."))]
    UnableToCopyList,

    #[snafu(display("Unable to append list value."))]
    UnableToAppendListValue,

    #[snafu(display("Unable to append extend list value."))]
    UnableToAppendExtendListValue,

    #[snafu(display("Unable to access list value by index."))]
    UnableToAccessListValueByIndex,

    #[snafu(display("Unable to create graph vertices iterator."))]
    UnableToCreateGraphVerticesIterator,

    #[snafu(display("Unable to make vertex copy."))]
    UnableToMakeVertexCopy,

    #[snafu(display("Unable to find vertex by id."))]
    UnableToFindVertexById,

    #[snafu(display("Unable to get vertex property."))]
    UnableToGetVertexProperty,

    #[snafu(display("Unable to return vertex property because of make name error."))]
    UnableToReturnVertexPropertyMakeNameEror,

    #[snafu(display("Unable to return vertex properties iterator."))]
    UnableToReturnVertexPropertiesIterator,

    #[snafu(display("Unable to make edge copy."))]
    UnableToMakeEdgeCopy,

    #[snafu(display("Unable to return vertex in_edges iterator."))]
    UnableToReturnVertexInEdgesIterator,

    #[snafu(display("Unable to return vertex out_edges iterator."))]
    UnableToReturnVertexOutEdgesIterator,

    #[snafu(display("Unable to return edge property because of value allocation error."))]
    UnableToReturnEdgePropertyValueAllocationError,

    #[snafu(display("Unable to return edge property because of value creation error."))]
    UnableToReturnEdgePropertyValueCreationError,

    #[snafu(display("Unable to return edge property because of name allocation error."))]
    UnableToReturnEdgePropertyNameAllocationError,

    #[snafu(display("Unable to return edge properties iterator."))]
    UnableToReturnEdgePropertiesIterator,

    #[snafu(display("Unable to make null value."))]
    UnableToMakeNullValue,

    #[snafu(display("Unable to make bool value."))]
    UnableToMakeBoolValue,

    #[snafu(display("Unable to make integer value."))]
    UnableToMakeIntegerValue,

    #[snafu(display("Unable to make double value."))]
    UnableToMakeDoubleValue,

    // TODO(gitbuda): Multiple string related error messages.
    #[snafu(display("Unable to make Memgraph compatible string value."))]
    UnableToMakeMemgraphStringValue,

    #[snafu(display("Unable to make list value."))]
    UnableToMakeListValue,

    #[snafu(display("Unable to make map value."))]
    UnableToMakeMapValue,

    #[snafu(display("Unable to make vertex value."))]
    UnableToMakeVertexValue,

    #[snafu(display("Unable to make edge value."))]
    UnableToMakeEdgeValue,

    #[snafu(display("Unable to make path value."))]
    UnableToMakePathValue,

    #[snafu(display("Unable to make new CString."))]
    UnableToMakeCString,

    #[snafu(display("Unable to make new Value::String."))]
    UnableToMakeValueString,

    #[snafu(display("Unable to create result record."))]
    UnableToCreateResultRecord,

    #[snafu(display("Unable to prepare result within Rust procedure."))]
    PreparingResultError,

    #[snafu(display("Unable to add a type of procedure parameter in Rust Module."))]
    AddProcedureParameterTypeError,

    #[snafu(display("Out of bound label index."))]
    OutOfBoundLabelIndexError,

    #[snafu(display("Unable to make a path copy."))]
    UnableToMakePathCopy,

    #[snafu(display("Out of bound path vertex index."))]
    OutOfBoundPathVertexIndex,

    #[snafu(display("Out of bound path edge index."))]
    OutOfBoundPathEdgeIndex,

    #[snafu(display("Unable to create path with start Vertex."))]
    UnableToCreatePathWithStartVertex,

    #[snafu(display(
        "Unable to expand Path because of not matching Vertex value or lack of memory."
    ))]
    UnableToExpandPath,
}

pub type MgpResult<T, E = MgpError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests;
