#![allow(clippy::just_underscores_and_digits)]

use alga::linear::FiniteDimInnerSpace;
use na::allocator::Allocator;
use na::*;
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

impl ButcherTableau<f64, U2> {
    /// the midpoint method https://en.wikipedia.org/wiki/Midpoint_method
    pub fn midpoint() -> Self {
        let a = Matrix2::new(0.0, 0.0, 0.5, 0.0);
        let b = Step::Fixed(Vector2::new(0.0, 1.0));
        let c = Vector2::new(0.0, 0.5);

        Self {
            symbol: RKSymbol::Midpoint,
            a,
            b,
            c,
        }
    }
    ///
    /// ```text
    ///  0.000 | 0.000 0.000
    ///  1.000 | 1.000 0.000
    /// -------+------------
    ///        | 0.500 0.500
    /// ```
    pub fn heun() -> Self {
        let a = Matrix2::new(0., 0., 1., 0.);
        let b = Step::Fixed(Vector2::new(0.5, 0.5));
        let c = Vector2::new(0., 1.);

        Self {
            symbol: RKSymbol::Heun,
            a,
            b,
            c,
        }
    }

    pub fn rk21() -> Self {
        let a = Matrix2::new(0.0, 0.0, 1.0, 0.0);
        let b = Step::Adaptive(Matrix2::new(0.5, 0.5, 1.0, 0.0));
        let c = Vector2::new(0.0, 1.0);

        Self {
            symbol: RKSymbol::RK21,
            a,
            b,
            c,
        }
    }
}

impl ButcherTableau<f64, U4> {
    pub fn rk23() -> Self {
        let a = Matrix4::new(
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.75,
            0.0,
            0.0,
            2.0 / 9.0,
            1.0 / 3.0,
            4.0 / 9.0,
            0.0,
        );
        let b = Step::Adaptive(Matrix2x4::new(
            7.0 / 24.0,
            0.25,
            1.0 / 3.0,
            0.125,
            1.0 / 9.0,
            1.0 / 3.0,
            4.0 / 9.0,
            0.0,
        ));
        let c = Vector4::new(0.0, 0.5, 0.75, 1.0);

        Self {
            symbol: RKSymbol::RK23,
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

impl ButcherTableau<f64, U6> {
    pub fn rk45() -> Self {
        let a = Matrix6::new(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.09375,
            0.28125,
            0.0,
            0.0,
            0.0,
            0.0,
            1932.0 / 2197.0,
            -7200.0 / 2197.0,
            7296.0 / 2197.0,
            0.0,
            0.0,
            0.0,
            439.0 / 216.0,
            -8.0,
            3680.0 / 513.0,
            -845.0 / 4104.0,
            0.0,
            0.0,
            -8.0 / 27.0,
            2.0,
            -3544.0 / 2565.0,
            1859.0 / 4104.0,
            -0.275,
            0.0,
        );
        let b = Step::Adaptive(Matrix2x6::new(
            25.0 / 216.0,
            0.0,
            1408.0 / 2565.0,
            2197.0 / 4104.0,
            -0.2,
            0.0,
            16.0 / 135.0,
            0.0,
            6656.0 / 12825.0,
            28561.0 / 56430.0,
            -0.18,
            2.0 / 55.0,
        ));

        let c = Vector6::new(0.0, 0.25, 0.375, 12.0 / 13.0, 1.0, 0.5);

        Self {
            symbol: RKSymbol::RK45,
            a,
            b,
            c,
        }
    }
}

impl ButcherTableau<f64, U7> {
    pub fn dopri5() -> Self {
        let a = MatrixMN::from_row_slice_generic(
            U7,
            U7,
            &[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.075,
                0.225,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                44.0 / 45.0,
                -56.0 / 15.0,
                32.0 / 9.0,
                0.0,
                0.0,
                0.0,
                0.0,
                19372.0 / 6561.0,
                -25360.0 / 2187.0,
                64448.0 / 6561.0,
                -212.0 / 729.0,
                0.0,
                0.0,
                0.0,
                9017.0 / 3168.0,
                -355.0 / 33.0,
                46732.0 / 5247.0,
                49.0 / 176.0,
                -5103.0 / 18656.0,
                0.0,
                0.0,
                35.0 / 384.0,
                0.0,
                500.0 / 1113.0,
                125.0 / 192.0,
                -2187.0 / 6784.0,
                11.0 / 84.0,
                0.0,
            ],
        );
        let b = Step::Adaptive(MatrixMN::from_row_slice_generic(
            U2,
            U7,
            &[
                35.0 / 384.0,
                0.0,
                500.0 / 1113.0,
                125.0 / 192.0,
                -2187.0 / 6784.0,
                11.0 / 84.0,
                0.0,
                5179.0 / 57600.0,
                0.0,
                7571.0 / 16695.0,
                393.0 / 640.0,
                -92097.0 / 339_200.0,
                187.0 / 2100.0,
                0.025,
            ],
        ));
        let c =
            VectorN::from_row_slice_generic(U7, U1, &[0.0, 0.2, 0.3, 0.75, 8.0 / 9.0, 1.0, 1.0]);

        Self {
            symbol: RKSymbol::Dopri5,
            a,
            b,
            c,
        }
    }
}

impl ButcherTableau<f64, U13> {
    pub fn feh78() -> Self {
        let a = MatrixMN::from_row_slice_generic(
            U13,
            U13,
            &[
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0 / 27.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0 / 36.0,
                1.0 / 12.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0 / 24.0,
                0.0,
                0.125,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                5.0 / 12.0,
                0.0,
                -25.0 / 16.0,
                25.0 / 16.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5,
                0.0,
                0.0,
                0.75,
                0.2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -25.0 / 108.0,
                0.0,
                0.0,
                125.0 / 108.0,
                -65.0 / 27.0,
                125.0 / 54.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                31.0 / 300.0,
                0.0,
                0.0,
                0.0,
                61.0 / 225.0,
                -2.0 / 9.0,
                13.0 / 900.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                -53.0 / 6.0,
                704.0 / 45.0,
                -107.0 / 9.0,
                67.0 / 90.0,
                3.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -91.0 / 108.0,
                0.0,
                0.0,
                23.0 / 108.0,
                -976.0 / 135.0,
                311.0 / 54.0,
                -19.0 / 60.0,
                17.0 / 6.0,
                -1.0 / 12.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2383.0 / 4100.0,
                0.0,
                0.0,
                -341.0 / 164.0,
                4496.0 / 1025.0,
                -301.0 / 82.0,
                2133.0 / 4100.0,
                45.0 / 82.0,
                45.0 / 164.0,
                18.0 / 41.0,
                0.0,
                0.0,
                0.0,
                3.0 / 205.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -6.0 / 41.0,
                -3.0 / 205.0,
                -3.0 / 41.0,
                3.0 / 41.0,
                6.0 / 41.0,
                0.0,
                0.0,
                0.0,
                -1777.0 / 4100.0,
                0.0,
                0.0,
                -341.0 / 164.0,
                4496.0 / 1025.0,
                -289.0 / 82.0,
                2193.0 / 4100.0,
                51.0 / 82.0,
                33.0 / 164.0,
                12.0 / 41.0,
                0.0,
                1.0,
                0.0,
            ],
        );
        let b = Step::Adaptive(MatrixMN::from_row_slice_generic(
            U2,
            U13,
            &[
                41.0 / 840.0,
                0.0,
                0.0,
                0.0,
                0.0,
                34.0 / 105.0,
                9.0 / 35.0,
                9.0 / 35.0,
                9.0 / 280.0,
                9.0 / 280.0,
                41.0 / 840.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                34.0 / 105.0,
                9.0 / 35.0,
                9.0 / 35.0,
                9.0 / 280.0,
                9.0 / 280.0,
                0.0,
                41.0 / 840.0,
                41.0 / 840.0,
            ],
        ));
        let c = VectorN::from_row_slice_generic(
            U13,
            U1,
            &[
                0.0,
                2.0 / 27.0,
                1.0 / 9.0,
                1.0 / 6.0,
                5.0 / 12.0,
                0.5,
                5.0 / 6.0,
                1.0 / 6.0,
                2.0 / 3.0,
                1.0 / 3.0,
                1.0,
                0.0,
                1.0,
            ],
        );

        Self {
            symbol: RKSymbol::Feh78,
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
        assert!(ButcherTableau::midpoint().is_consistent_rk());
        assert!(ButcherTableau::heun().is_consistent_rk());
        assert!(ButcherTableau::rk21().is_consistent_rk());
        assert!(ButcherTableau::rk4().is_consistent_rk());
    }
}
