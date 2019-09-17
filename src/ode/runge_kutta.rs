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
    pub b: Weights<T, S>,
    /// nodes
    pub c: VectorN<T, S>,
}

#[derive(Debug, Clone)]
pub enum WeightType {
    Fixed,
    Adaptive,
}

#[derive(Debug, Clone)]
pub enum Weights<N: RealField, S: Dim>
where
    DefaultAllocator: Allocator<N, S> + Allocator<N, U2, S>,
{
    // TODO use also MatrixMN<N, U2, S> for ease of use? or swap MN for Adaptive?
    Explicit(VectorN<N, S>),
    /// Adaptive weights where
    /// row 1 is used for stepping
    /// row 2 is used for error-checking
    Adaptive(MatrixMN<N, U2, S>),
}

impl<N: RealField, S: Dim> Weights<N, S>
where
    DefaultAllocator: Allocator<N, S> + Allocator<N, U2, S>,
{
    // TODO refactor, find better solution to separate fixed and daptive, is every btab adaptable?
    pub fn as_slice(&self) -> &[N] {
        match self {
            Weights::Explicit(e) => e.as_slice(),
            Weights::Adaptive(a) => a.as_slice(),
        }
    }

    pub fn weights(&self) {
        let r = match self {
            Weights::Explicit(e) => {}
            Weights::Adaptive(a) => {}
        };
    }
}

/// https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
impl<T: RealField, S: Dim> ButcherTableau<T, S>
where
    DefaultAllocator:
        Allocator<T, U1, S> + Allocator<T, U2, S> + Allocator<T, S, S> + Allocator<T, S>,
{
    /// the Butcher-Barrier says, that the amount of stages grows faster tha the order.
    /// For `nstages` ≥ 5, more than order 5 is required to solve the system
    #[inline]
    pub fn order(&self) -> usize {
        self.symbol.order()
    }

    /// the number of stages `S`
    #[inline]
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

    #[inline]
    pub fn is_fixed(&self) -> bool {
        match &self.b {
            Weights::Explicit(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_adaptive(&self) -> bool {
        !self.is_fixed()
    }

    #[inline]
    pub fn weight_type(&self) -> WeightType {
        match &self.b {
            Weights::Explicit(_) => WeightType::Fixed,
            Weights::Adaptive(_) => WeightType::Adaptive,
        }
    }

    /// First same as last. c.f. H&W p.167
    #[inline]
    pub fn is_first_same_as_last(&self) -> bool {
        let b = self.b.as_slice();
        let row_idx = self.nstages() - 1;
        println!("b {:?}", b);
        for c in 0..self.nstages() {
            println!("a: {}    b: {}", self.a[(row_idx, c)], b[c]);
            if self.a[(row_idx, c)] != b[c] {
                return false;
            }
        }
        self.c[row_idx] == T::one()
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
            Weights::Explicit(fixed) => {
                write!(f, "\n       |")?;
                for b in fixed.iter() {
                    write!(f, " {:.3}", b)?;
                }
            }
            Weights::Adaptive(adapt) => {
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
        let b = Weights::Explicit(Vector1::one());
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
        let a = Matrix2::new(0., 0., 0.5, 0.0);
        let b = Weights::Explicit(Vector2::new(0., 1.0));
        let c = Vector2::new(0., 0.5);

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
        let b = Weights::Explicit(Vector2::new(0.5, 0.5));
        let c = Vector2::new(0., 1.);

        Self {
            symbol: RKSymbol::Heun,
            a,
            b,
            c,
        }
    }

    pub fn rk21() -> Self {
        let a = Matrix2::new(0., 0., 1., 0.);
        let b = Weights::Adaptive(Matrix2::new(0.5, 0.5, 1., 0.));
        let c = Vector2::new(0., 1.);

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
            0.,
            0.,
            0.,
            0.,
            0.5,
            0.,
            0.,
            0.,
            0.,
            0.75,
            0.,
            0.,
            2. / 9.,
            1. / 3.,
            4. / 9.,
            0.,
        );
        let b = Weights::Adaptive(Matrix2x4::new(
            7. / 24.,
            0.25,
            1. / 3.,
            0.125,
            1. / 9.,
            1. / 3.,
            4. / 9.,
            0.,
        ));
        let c = Vector4::new(0., 0.5, 0.75, 1.);

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
        let c = Vector4::new(__, 0.5, 0.5, 1.);
        let b = Weights::Explicit(Vector4::new(1. / 6., 1. / 3., 1. / 3., 1. / 6.));
        let a = Matrix4::new(
            __, __, __, __, 0.5, __, __, __, __, 0.5, __, __, __, __, 1., __,
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
    /// Fehlberg https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    /// Order of 4 with an error estimator of order 5
    pub fn rk45() -> Self {
        let a = Matrix6::new(
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.25,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.09375,
            0.28125,
            0.,
            0.,
            0.,
            0.,
            1932. / 2197.,
            -7200. / 2197.,
            7296. / 2197.,
            0.,
            0.,
            0.,
            439. / 216.,
            -8.,
            3680. / 513.,
            -845. / 4104.,
            0.,
            0.,
            -8. / 27.,
            2.,
            -3544. / 2565.,
            1859. / 4104.,
            -0.275,
            0.,
        );
        let b = Weights::Adaptive(Matrix2x6::new(
            25. / 216.,
            0.,
            1408. / 2565.,
            2197. / 4104.,
            -0.2,
            0.,
            16. / 135.,
            0.,
            6656. / 12825.,
            28561. / 56430.,
            -0.18,
            2. / 55.,
        ));

        let c = Vector6::new(0., 0.25, 0.375, 12. / 13., 1., 0.5);

        Self {
            symbol: RKSymbol::RK45,
            a,
            b,
            c,
        }
    }
}
///  0.000 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000
///  0.200 | 0.200 0.000 0.000 0.000 0.000 0.000 0.000
///  0.300 | 0.075 0.225 0.000 0.000 0.000 0.000 0.000
///  0.750 | 0.978 -3.733 3.556 0.000 0.000 0.000 0.000
///  0.889 | 2.953 -11.596 9.823 -0.291 0.000 0.000 0.000
///  1.000 | 2.846 -10.758 8.906 0.278 -0.274 0.000 0.000
///  1.000 | 0.091 0.000 0.449 0.651 -0.322 0.131 0.000
/// -------+------------------------------------------
///        | 0.091 0.000 0.449 0.651 -0.322 0.131 0.000
///        | 0.090 0.000 0.453 0.614 -0.272 0.089 0.025
impl ButcherTableau<f64, U7> {
    pub fn dopri5() -> Self {
        let a = MatrixMN::from_row_slice_generic(
            U7,
            U7,
            &[
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.2,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.075,
                0.225,
                0.,
                0.,
                0.,
                0.,
                0.,
                44. / 45.,
                -56. / 15.,
                32. / 9.,
                0.,
                0.,
                0.,
                0.,
                19372. / 6561.,
                -25360. / 2187.,
                64448. / 6561.,
                -212. / 729.,
                0.,
                0.,
                0.,
                9017. / 3168.,
                -355. / 33.,
                46732. / 5247.,
                49. / 176.,
                -5103. / 18656.,
                0.,
                0.,
                35. / 384.,
                0.,
                500. / 1113.,
                125. / 192.,
                -2187. / 6784.,
                11. / 84.,
                0.,
            ],
        );
        let b = Weights::Adaptive(MatrixMN::from_row_slice_generic(
            U2,
            U7,
            &[
                35. / 384.,
                0.,
                500. / 1113.,
                125. / 192.,
                -2187. / 6784.,
                11. / 84.,
                0.,
                5179. / 57600.,
                0.,
                7571. / 16695.,
                393. / 640.,
                -92097. / 339_200.,
                187. / 2100.,
                0.025,
            ],
        ));
        let c = VectorN::from_row_slice_generic(U7, U1, &[0., 0.2, 0.3, 0.75, 8. / 9., 1., 1.]);

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
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                2. / 27.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                1. / 36.,
                1. / 12.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                1. / 24.,
                0.,
                0.125,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                5. / 12.,
                0.,
                -25. / 16.,
                25. / 16.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.5,
                0.,
                0.,
                0.75,
                0.2,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -25. / 108.,
                0.,
                0.,
                125. / 108.,
                -65. / 27.,
                125. / 54.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                31. / 300.,
                0.,
                0.,
                0.,
                61. / 225.,
                -2. / 9.,
                13. / 900.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                2.,
                0.,
                0.,
                -53. / 6.,
                704. / 45.,
                -107. / 9.,
                67. / 90.,
                3.,
                0.,
                0.,
                0.,
                0.,
                0.,
                -91. / 108.,
                0.,
                0.,
                23. / 108.,
                -976. / 135.,
                311. / 54.,
                -19. / 60.,
                17. / 6.,
                -1. / 12.,
                0.,
                0.,
                0.,
                0.,
                2383. / 4100.,
                0.,
                0.,
                -341. / 164.,
                4496. / 1025.,
                -301. / 82.,
                2133. / 4100.,
                45. / 82.,
                45. / 164.,
                18. / 41.,
                0.,
                0.,
                0.,
                3. / 205.,
                0.,
                0.,
                0.,
                0.,
                -6. / 41.,
                -3. / 205.,
                -3. / 41.,
                3. / 41.,
                6. / 41.,
                0.,
                0.,
                0.,
                -1777. / 4100.,
                0.,
                0.,
                -341. / 164.,
                4496. / 1025.,
                -289. / 82.,
                2193. / 4100.,
                51. / 82.,
                33. / 164.,
                12. / 41.,
                0.,
                1.,
                0.,
            ],
        );
        let b = Weights::Adaptive(MatrixMN::from_row_slice_generic(
            U2,
            U13,
            &[
                41. / 840.,
                0.,
                0.,
                0.,
                0.,
                34. / 105.,
                9. / 35.,
                9. / 35.,
                9. / 280.,
                9. / 280.,
                41. / 840.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                34. / 105.,
                9. / 35.,
                9. / 35.,
                9. / 280.,
                9. / 280.,
                0.,
                41. / 840.,
                41. / 840.,
            ],
        ));
        let c = VectorN::from_row_slice_generic(
            U13,
            U1,
            &[
                0.,
                2. / 27.,
                1. / 9.,
                1. / 6.,
                5. / 12.,
                0.5,
                5. / 6.,
                1. / 6.,
                2. / 3.,
                1. / 3.,
                1.,
                0.,
                1.,
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

    #[test]
    fn is_fsal() {
        assert!(!ButcherTableau::midpoint().is_first_same_as_last());
        println!("{}", ButcherTableau::dopri5());
        assert!(ButcherTableau::dopri5().is_first_same_as_last());
    }
}
