use cuda_async::error::DeviceError;
use cutile_compiler::error::JITError;
use cuda_core::DriverError;
use std::{
    error,
    fmt::{self, Display, Formatter},
};

#[derive(Debug)]
pub struct TensorError(pub String);

impl Display for TensorError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let message = &self.0;
        write!(formatter, "{message}")
    }
}

impl error::Error for TensorError {}

#[derive(Debug)]
pub struct KernelLaunchError(pub String);

impl Display for KernelLaunchError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let message = &self.0;
        write!(formatter, "{message}")
    }
}

impl error::Error for KernelLaunchError {}

pub enum Error {
    Tensor(TensorError),
    KernelLaunch(KernelLaunchError),
    JIT(JITError),
    Device(DeviceError),
    Driver(DriverError),
    Anyhow(anyhow::Error),
}

impl fmt::Debug for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        // Delegate to Display so that `.unwrap()` produces the same
        // user-facing output (including source locations from JITError).
        Display::fmt(self, formatter)
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::Tensor(error) => write!(formatter, "error: {error}"),
            Self::KernelLaunch(error) => write!(formatter, "error: {error}"),
            Self::JIT(error) => write!(formatter, "error: {error}"),
            Self::Device(error) => write!(formatter, "error: {error}"),
            Self::Driver(error) => write!(formatter, "error: {error}"),
            Self::Anyhow(error) => write!(formatter, "error: {error}"),
        }
    }
}

impl error::Error for Error {}

// Tensor

pub fn tensor_error(err_str: &str) -> Error {
    return Error::Tensor(TensorError(err_str.to_string()));
}

pub fn tensor_error_result<R>(err_str: &str) -> Result<R, Error> {
    return Err(tensor_error(err_str));
}

// Kernel Launch

pub fn kernel_launch_error(err_str: &str) -> Error {
    return Error::KernelLaunch(KernelLaunchError(err_str.to_string()));
}

pub fn kernel_launch_error_result<R>(err_str: &str) -> Result<R, Error> {
    return Err(kernel_launch_error(err_str));
}

// anyhow

impl From<anyhow::Error> for Error {
    fn from(error: anyhow::Error) -> Self {
        Self::Anyhow(error)
    }
}

// Device

impl From<Error> for DeviceError {
    fn from(error: Error) -> Self {
        DeviceError::Anyhow(format!("{:?}", error))
    }
}

impl From<DeviceError> for Error {
    fn from(error: DeviceError) -> Self {
        Self::Device(error)
    }
}

// DriverError

impl From<DriverError> for Error {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

// Syn

impl From<JITError> for Error {
    fn from(error: JITError) -> Self {
        Self::JIT(error)
    }
}
