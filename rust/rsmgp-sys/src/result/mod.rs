use snafu::Snafu;

#[derive(Debug, PartialEq, Snafu)]
#[snafu(visibility = "pub")]
pub enum MgpError {
    #[snafu(display("Unable to insert map value."))]
    UnableToInsertMapValue,

    #[snafu(display("Unable to access map value."))]
    UnableToAccessMapValue,

    #[snafu(display("Unable to create map iterator."))]
    UnableToCreateMapIterator,

    #[snafu(display("Unable to create map value object."))]
    UnableToCreateMapValueObject,

    #[snafu(display("Unable to create map."))]
    UnableToCreateMap,

    #[snafu(display("Unable to create empty map."))]
    UnableToCreateEmptyMap,

    #[snafu(display("Unable to create empty list."))]
    UnableToCreateEmptyList,

    #[snafu(display("Unable to copy list."))]
    UnableToCopyList,

    #[snafu(display("Unable to create list."))]
    UnableToCreateList,

    #[snafu(display("Unable to append list value."))]
    UnableToAppendListValue,

    #[snafu(display("Unable to append extend list value."))]
    UnableToAppendExtendListValue,

    #[snafu(display("Unable to access list value by index."))]
    UnableToAccessListValueByIndex,

    #[snafu(display("Unable to create graph vertices iterator."))]
    UnableToCreateGraphVerticesIterator,

    #[snafu(display("Unable to find vertex by id."))]
    UnableToFindVertexById,

    #[snafu(display("Unable to return vertex property because of value allocation error."))]
    UnableToReturnVertexPropertyValueAllocationError,

    #[snafu(display("Unable to return vertex properties iterator."))]
    UnableToReturnVertexPropertiesIterator,

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

    #[snafu(display("Unable to allocate null value."))]
    UnableToAllocateNullValue,

    #[snafu(display("Unable to allocate bool value."))]
    UnableToAllocateBoolValue,

    #[snafu(display("Unable to allocate integer value."))]
    UnableToAllocateIntegerValue,

    #[snafu(display("Unable to allocate string value."))]
    UnableToAllocateStringValue,

    #[snafu(display("Unable to allocate double value."))]
    UnableToAllocateDoubleValue,

    #[snafu(display("Unable to allocate vertex value."))]
    UnableToAllocateVertexValue,

    #[snafu(display("Unable to allocate edge value."))]
    UnableToAllocateEdgeValue,

    #[snafu(display("Unable to create/allocate new CString."))]
    UnableToCreateCString,

    #[snafu(display("Unable to create result record."))]
    UnableToCreateResultRecord,

    #[snafu(display("Unable to prepare result within Rust procedure."))]
    PreparingResultError,

    #[snafu(display("Unable to add a type of procedure parameter in Rust Module."))]
    AddProcedureParameterTypeError,

    #[snafu(display("Out of bound label index."))]
    OutOfBoundLabelIndexError,

    #[snafu(display("Out of bound path vertex index."))]
    OutOfBoundPathVertexIndex,

    #[snafu(display("Out of bound path edge index."))]
    OutOfBoundPathEdgeIndex,

    #[snafu(display("Unable to allocate path value."))]
    UnableToAllocatePathValue,
}

pub type MgpResult<T, E = MgpError> = std::result::Result<T, E>;
