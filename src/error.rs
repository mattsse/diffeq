use crate::ode::runge_kutta::WeightType;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OdeError {
    #[error("{msg}")]
    Uninitialized { msg: String },
    #[error("Expected {expected:?} weights, found {found:?} weights")]
    InvalidButcherTableauWeightType {
        expected: WeightType,
        found: WeightType,
    },
    #[error(
        "Encountered NAN after {computation} computations while solving at timestamp {timestamp}"
    )]
    NAN { computation: usize, timestamp: f64 },
    #[error("Zero time span is not allowed")]
    ZeroTimeSpan,
    #[error("Initial step has wrong sign")]
    InvalidInitstep,
    #[error("Unable to compute matrix operation")]
    InvalidMatrix,
}

impl OdeError {
    pub(crate) fn uninitialized<T: ToString>(s: T) -> Self {
        OdeError::Uninitialized { msg: s.to_string() }
    }
}

/// Enumeration of the errors that may arise during integration.
#[derive(Error, Debug)]
pub enum IntegrationError {
    #[error("Maximum steps reached at {at}, after {n_step} steps")]
    MaxNumStepReached { at: f64, n_step: u32 },
    #[error("Encountered step size underflow at {at}")]
    StepSizeUnderflow { at: f64 },
    #[error("Stiffness detected at {at}")]
    StiffnessDetected { at: f64 },
}
