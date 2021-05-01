use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(visibility = "pub")]
pub enum MgpError {
    #[snafu(display("An error inside Rust procedure."))]
    MgpDefaultError,
    #[snafu(display("Unable to allocate memory inside Rust procedure."))]
    MgpAllocationError,
    #[snafu(display("Unable to prepare result within Rust procedure."))]
    MgpPreparingResultError,
    #[snafu(display("Unable to add a type of procedure paramater in Rust Module."))]
    MgpAddProcedureParameterTypeError,
}

pub type MgpResult<T, E = MgpError> = std::result::Result<T, E>;
