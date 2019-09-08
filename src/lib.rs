#![allow(unused)]
#[macro_use]
extern crate derive_builder;

extern crate nalgebra as na;

/// Every equation has a Problem type, a solution type, and the same solution handling setup.
pub mod gaussian;
pub mod hessian;
pub mod jacobian;

pub mod error;

pub mod ode;

pub mod bvp;

pub mod sde;

pub mod alg;

#[cfg(feature = "problems")]
pub mod problems;

pub mod types;

pub mod algorithms;
