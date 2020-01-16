#![allow(unused)]
#[macro_use]
extern crate derive_builder;

extern crate nalgebra as na;

/// Every equation should hav a Problem type, a solution type, and the same solution handling setup.
pub mod error;
pub mod ode;
