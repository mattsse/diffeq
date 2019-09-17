use crate::ode::runge_kutta::WeightType;
use snafu::{ensure, Backtrace, ErrorCompat, ResultExt, Snafu};
use std::str::FromStr;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Element not initialized {}", msg))]
    Uninitialized { msg: String },
    #[snafu(display("The user id {} is invalid", user_id))]
    UserIdInvalid { user_id: i32, backtrace: Backtrace },
    #[snafu(display("{}", err))]
    Ode { err: OdeError },
}

impl Error {
    pub(crate) fn uninitialized<T: Into<String>>(msg: T) -> Self {
        Error::Uninitialized { msg: msg.into() }
    }

    pub(crate) fn ode<T: Into<OdeError>>(err: T) -> Self {
        Error::Ode { err: err.into() }
    }
}

#[derive(Debug, Snafu)]
pub enum OdeError {
    #[snafu(display("Expected {:?} weights, got {:?} weights", expected, got))]
    InvalidButcherTableauWeightType {
        expected: WeightType,
        got: WeightType,
    },
    #[snafu(display(
        "Encountered NAN after {} computations while solving at timestamp {}",
        computation,
        timestamp
    ))]
    NAN { computation: usize, timestamp: f64 },
    #[snafu(display("Zero time span"))]
    ZeroTimeSpan,
    #[snafu(display("Initial step has wrong sign"))]
    InvalidInitstep,
}

impl Into<Error> for OdeError {
    fn into(self) -> Error {
        Error::Ode { err: self }
    }
}
