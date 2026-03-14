use proc_macro2::{Span, TokenStream as TokenStream2};
use std::{
    error,
    fmt::{self, Display, Formatter},
};
use syn::spanned::Spanned;

#[derive(Debug)]
pub enum Error {
    Syn(syn::Error),
}

impl From<syn::Error> for Error {
    fn from(error: syn::Error) -> Self {
        Self::Syn(error)
    }
}

impl Error {
    pub fn to_compile_error(&self) -> TokenStream2 {
        match self {
            Self::Syn(err) => err.to_compile_error().into(),
        }
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::Syn(error) => write!(formatter, "Syntax error: {error}"),
        }
    }
}

impl error::Error for Error {}

/// Create an `Error` at the given span.
pub fn syn_err(span: Span, message: &str) -> Error {
    syn::Error::new(span, message).into()
}

/// Return `Err(Error)` at the given span with an arbitrary `Ok` type.
pub fn syn_error_at<T>(span: Span, message: &str) -> Result<T, Error> {
    Err(syn_err(span, message))
}

/// Return `Err(Error)` at `Span::call_site()` with an arbitrary `Ok` type.
pub fn call_site_error<T>(message: &str) -> Result<T, Error> {
    syn_error_at(Span::call_site(), message)
}

pub trait SpannedError {
    /// Return `Err(Error)` anchored to this item's span, with an arbitrary `Ok` type.
    fn err<T>(&self, message: &str) -> Result<T, Error>;
}

impl<S> SpannedError for S
where
    S: Spanned,
{
    fn err<T>(&self, message: &str) -> Result<T, Error> {
        syn_error_at(self.span(), message)
    }
}
