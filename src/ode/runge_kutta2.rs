#![allow(clippy::just_underscores_and_digits)]

use crate::ode::ode2::OdeProblem;
use alga::linear::FiniteDimInnerSpace;
use na::{
    allocator::Allocator, dimension::Dim, DefaultAllocator, DimName, Matrix1, Matrix1x4, Matrix4,
    MatrixMN, MatrixN, MatrixSlice, RealField, Unit, Vector1, Vector2, Vector3, Vector4, VectorN,
    U1, U2, U4,
};
use num_traits::identities::{One, Zero};
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

/// Tableua of the form
///  c1  | a_11   ....   a_1s
///  .   | a_21 .          .
///  .   | a_31     .      .
///  .   | ....         .  .
///  c_s | a_s1  ....... a_ss
/// -----+--------------------
///      | b_1     ...   b_s   this is the one used for stepping
///      | b'_1    ...   b'_s  this is the one used for error-checking
///
/// where `T` is the type of the coefficients
/// and `S` is the number of stages (an int)
#[derive(Debug, Clone)]
pub struct ButcherTableau<T: RealField, S: Dim>
where
    DefaultAllocator:
        Allocator<T, U1, S> + Allocator<T, U2, S> + Allocator<T, S, S> + Allocator<T, S>,
{
    /// identifier for the rk method
    pub symbol: RKSymbol,
    /// coefficients - rk matrix
    pub a: MatrixN<T, S>,
    /// weights
    pub b: Step<T, S>,
    /// nodes
    pub c: VectorN<T, S>,
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

impl<N: RealField, S: Dim> Step<N, S>
where
    DefaultAllocator: Allocator<N, S> + Allocator<N, U2, S>,
{
    // TODO refactor, find better solution to separate fixed and daptive, is every btab adaptable?
    pub fn as_slice(&self) -> &[N] {
        match self {
            Step::Fixed(f) => f.as_slice(),
            Step::Adaptive(a) => a.as_slice(),
        }
    }
}

// TODO refactor
//impl<'a, T: RealField, S: Dim> IntoIterator for &'a ButcherTableau<T, S>
//where
//    DefaultAllocator:
//        Allocator<T, U1, S> + Allocator<T, U2, S> + Allocator<T, S, S> + Allocator<T, S>,
//{
//    type Item = T;
//    type IntoIter = super::ode2::K<'a, T, S>;
//
//    fn into_iter(self) -> Self::IntoIter {
//        super::ode2::K {
//            order: 0,
//            btab: self
//        }
//    }
//}

impl<T: RealField, S: Dim> ButcherTableau<T, S>
where
    DefaultAllocator:
        Allocator<T, U1, S> + Allocator<T, U2, S> + Allocator<T, S, S> + Allocator<T, S>,
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
            if self.a[(i, j)] != self.c[i] {
                return false;
            }
        }
        true
    }

    pub fn is_fixed(&self) -> bool {
        match &self.b {
            Step::Fixed(_) => true,
            _ => false,
        }
    }

    pub fn is_adaptive(&self) -> bool {
        !self.is_fixed()
    }

    /// the midpoint method https://en.wikipedia.org/wiki/Midpoint_method
    // TODO move to second order only
    pub fn midpoint(&self) {}
}

// TODO impl access funs
// https://github.com/srenevey/ode-solvers/blob/master/src/butcher_tableau.rs#L388

impl<T: RealField, S: Dim> fmt::Display for ButcherTableau<T, S>
where
    DefaultAllocator:
        Allocator<T, U1, S> + Allocator<T, U2, S> + Allocator<T, S, S> + Allocator<T, S>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row in 0..self.nstages() {
            write!(f, " {:.3} |", self.c[row])?;
            for col in 0..self.nstages() {
                write!(f, " {:.3}", self.a[(row, col)])?;
            }
            writeln!(f)?;
        }
        write!(f, "-------+")?;
        write!(f, "{}", "------".repeat(self.nstages()))?;
        match &self.b {
            Step::Fixed(fixed) => {
                write!(f, "\n       |")?;
                for b in fixed.iter() {
                    write!(f, " {:.3}", b)?;
                }
            }
            Step::Adaptive(adapt) => {
                for row in 0..adapt.nrows() {
                    write!(f, "\n       |")?;
                    for col in 0..self.nstages() {
                        write!(f, " {:.3}", adapt[(row, col)])?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl ButcherTableau<f64, U1> {
    /// constructs the Butcher Tableau for the (forward) Euler method
    /// ```text
    ///   0.000 | 0.000
    ///  -------+------
    ///         | 1.000
    /// ```
    pub fn feuler() -> Self {
        let a = Matrix1::zero();
        let b = Step::Fixed(Vector1::one());
        let c = Vector1::zero();

        Self {
            symbol: RKSymbol::Feuler,
            a,
            b,
            c,
        }
    }
}

impl ButcherTableau<f64, U4> {
    /// constructs the Butcher Tableau for the Runge Kutta 4 method
    /// ```text
    ///    0.000 | 0.000 0.000 0.000 0.000
    ///    0.500 | 0.500 0.000 0.000 0.000
    ///    0.500 | 0.000 0.500 0.000 0.000
    ///    1.000 | 0.000 0.000 1.000 0.000
    ///    -------+------------------------
    ///    | 0.167 0.333 0.333 0.167
    /// ```
    pub fn rk4() -> Self {
        let __ = 0.0;
        let c = Vector4::new(__, 0.5, 0.5, 1.0);
        let b = Step::Fixed(Vector4::new(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0));
        let a = Matrix4::new(
            __, __, __, __, 0.5, __, __, __, __, 0.5, __, __, __, __, 1.0, __,
        );

        Self {
            symbol: RKSymbol::RK4,
            a,
            b,
            c,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_consistent() {
        assert!(ButcherTableau::feuler().is_consistent_rk());
        assert!(ButcherTableau::rk4().is_consistent_rk());
    }
}
