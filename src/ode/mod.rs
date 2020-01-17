pub mod coeff;
pub mod options;
pub mod problem;
pub mod rosenbrock;
pub mod runge_kutta;
pub mod solution;
pub mod types;
#[cfg(feature = "serde0")]
use serde::{Deserialize, Serialize};

/// The available ODE solvers.
#[cfg_attr(feature = "serde0", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum Ode {
    Feuler,
    Heun,
    Midpoint,
    Ode23,
    Ode23s,
    Ode4,
    Ode45,
    Ode45fe,
    Ode4skr,
    Ode4ss,
    Ode78,
}

impl std::str::FromStr for Ode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "feuler" => Ok(Ode::Feuler),
            "heun" => Ok(Ode::Heun),
            "midpoint" => Ok(Ode::Midpoint),
            "ode23" => Ok(Ode::Ode23),
            "ode23s" => Ok(Ode::Ode23s),
            "ode4" => Ok(Ode::Ode4),
            "ode45" => Ok(Ode::Ode45),
            "ode4skr" => Ok(Ode::Ode4skr),
            "ode4s" => Ok(Ode::Ode4ss),
            "ode78" => Ok(Ode::Ode78),
            _ => Err(format!("{} is not a valid Ode identifier", s)),
        }
    }
}
