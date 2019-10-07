use crate::ode::runge_kutta::WeightType;
use snafu::{Backtrace, Snafu};

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
    pub fn uninitialized<T: Into<String>>(msg: T) -> Self {
        Error::Uninitialized { msg: msg.into() }
    }

    pub fn ode<T: Into<OdeError>>(err: T) -> Self {
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
    #[snafu(display("Unable to compute matrix operation"))]
    InvalidMatrix,
}

impl Into<Error> for OdeError {
    fn into(self) -> Error {
        Error::Ode { err: self }
    }
}

/// Enumeration of the errors that may arise during integration.
#[derive(Debug)]
pub enum IntegrationError {
    MaxNumStepReached { x: f64, n_step: u32 },
    StepSizeUnderflow { x: f64 },
    StiffnessDetected { x: f64 },
}
