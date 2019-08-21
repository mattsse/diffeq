use crate::ode::ode2::OdeProblem;
use alga::linear::FiniteDimInnerSpace;
use na::allocator::Allocator;
use na::dimension::Dim;
use na::{
    DefaultAllocator, DimName, MatrixMN, MatrixN, RealField, Unit, Vector2, Vector3, VectorN, U1,
    U2,
};
use std::fmt;

#[derive(Debug, Clone)]
pub enum RKSymbol {
    Feuler,
    Midpoint,
    Heun,
    RK4,
    RK21,
    RK23,
    RK45,
    Dopri5,
    Feh78,
    /// (Name, Order)
    Other((String, usize)),
}

impl RKSymbol {
    pub fn order(&self) -> usize {
        unimplemented!()
    }
}

// T is the type of the coefficients
// S is the number of stages (an int)

/// Tableua of the form
///  c1  | a_11   ....   a_1s
///  .   | a_21 .          .
///  .   | a_31     .      .
///  .   | ....         .  .
///  c_s | a_s1  ....... a_ss
/// -----+--------------------
///      | b_1     ...   b_s   this is the one used for stepping
///      | b'_1    ...   b'_s  this is the one used for error-checking
#[derive(Debug, Clone)]
pub struct ButcherTableau<T: RealField, S: Dim, C: Dim = U1>
where
    DefaultAllocator: Allocator<T, U1, S>
        + Allocator<T, U2, S>
        + Allocator<T, S, S>
        + Allocator<T, S>
        + Allocator<T, S, C>,
{
    /// identifier for the rk method
    pub symbol: RKSymbol,
    /// coefficients - rk matrix
    pub a: MatrixN<T, S>,
    /// weights
    pub b: Step<T, S>,
    /// nodes
    pub c: MatrixMN<T, S, C>,
}

#[derive(Debug, Clone)]
pub enum Step<N: RealField, S: Dim>
where
    DefaultAllocator: Allocator<N, S> + Allocator<N, U2, S>,
{
    Fixed(VectorN<N, S>),
    /// Adaptive weights where
    /// row 1 is used for stepping
    /// row 2 is used for error-checking
    Adaptive(MatrixMN<N, U2, S>),
}

impl<T: RealField, S: Dim, C: Dim> ButcherTableau<T, S, C>
where
    DefaultAllocator: Allocator<T, U1, S>
        + Allocator<T, U2, S>
        + Allocator<T, S, S>
        + Allocator<T, S>
        + Allocator<T, S, C>,
{
    // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    /// the Butcher-Barrier says, that the amount of stages grows faster tha the order.
    /// For `nstages` ≥ 5, more than order 5 is required to solve the system
    #[inline]
    pub fn order(&self) -> usize {
        self.symbol.order()
    }

    /// the number of stages `S`
    pub fn nstages(&self) -> usize {
        self.c.nrows()
    }

    /// checks wether the rk method is consistent
    /// A Runge–Kutta method is consistent if:
    /// \sum _{j=1}^{i-1}a_{ij}=c_{i}{\text{ for }}i=2,\ldots ,s.
    /// checks if the diagonal entries of the lower triangle of the coefficents `a` correspond to the values of `c`
    pub fn is_consistent_rk(&self) -> bool {
        for (j, i) in (1..self.nstages()).enumerate() {
            if self.a[(i, j)] != self.c[(i, 0)] {
                return false;
            }
        }
        true
    }

    fn oderk_fixed(&self) {
        let ks: Vec<T> = Vec::with_capacity(self.order());
    }

    /// the midpoint method https://en.wikipedia.org/wiki/Midpoint_method
    // TODO move to second order only
    pub fn midpoint(&self) {}
}

// TODO impl access funs
// https://github.com/srenevey/ode-solvers/blob/master/src/butcher_tableau.rs#L388

//impl fmt::Display for ButcherTableau {
//    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        unimplemented!()
//    }
//}
