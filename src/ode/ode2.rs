pub enum Ode {
    Ode23,
    Ode45,
    Ode23s,
    Ode78,
    Ode4,
    Ode4ms,
    Ode4s,
}

pub trait OdeAlgorithm {}

// http://docs.juliadiffeq.org/latest/features/performance_overloads.html
pub trait OdeFunction {
    type u0;
}

/// F: the RHS of the ODE dy/dt = F(t,y), which is a function of t and y(t)
/// and returns dy/dt::typeof(y/t)
/// y0: initial value for y. The type of y0, promoted as necessary according to the numeric type used
/// for the times, determines the element type of the yout vector (yout::Vector{typeof(y0*one(t))})
/// tspan: Any iterable of sorted t values at which the solution (y) is requested.
/// Most solvers will only consider tspan[1] and tspan[end], and intermediary points will be
/// interpolated. If tspan[1] > tspan[end] the integration is performed backwards. The times are
/// promoted as necessary to a common floating-point type.
pub trait OdeSolver {
    /// Vector of points at which solutions were obtained
    type Tout;
    /// solutions at times tout, stored as a vector yout as described above.
    /// Note that if y0 is a vector, you can get a matlab-like matrix with hcat(yout...).
    type Yout;
}

pub struct OdeProblem<Rhs, Start, Time> {
    rhs: Rhs,
    y0: Start,
    time: Vec<Time>,
}

pub struct OdeSolution<Tout, Yout> {
    tout: Vec<Tout>,
    yout: Vec<Yout>,
}

/// determine the maximal, minimal and initial integration step.
#[derive(Debug, Default)]
pub struct Steps {
    pub maxstep: usize,
    pub minstep: usize,
    pub initstep: usize,
}

/// defining your ODE in pseudocode
macro_rules! ode_def {
    () => {};
}

macro_rules! ode_def_bare {
    () => {};
}

pub trait Solver {
    type Problem;
    type Alg;
    type Args;
}

#[cfg(test)]
mod tests {

    #[test]
    fn ode() {}
}
