pub mod options;
pub mod problem;
pub mod runge_kutta;
pub mod types;

#[derive(Debug, Clone)]
pub enum Ode {
    Ode23,
    Ode23s,
    Ode4,
    Ode45,
    Ode4ms,
    Ode4s,
    Ode78,
}

impl std::str::FromStr for Ode {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ode23" => Ok(Ode::Ode23),
            "ode23s" => Ok(Ode::Ode23s),
            "ode4" => Ok(Ode::Ode4),
            "ode45" => Ok(Ode::Ode45),
            "ode4ms" => Ok(Ode::Ode4ms),
            "ode4s" => Ok(Ode::Ode4s),
            "ode78" => Ok(Ode::Ode78),
            _ => Err(()),
        }
    }
}
