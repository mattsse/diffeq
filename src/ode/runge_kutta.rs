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
    Other((String, RKOrder)),
}

impl RKSymbol {
    pub fn order(&self) -> RKOrder {
        match self {
            RKSymbol::Feuler => RKOrder::Explicit(1),
            RKSymbol::Midpoint => RKOrder::Explicit(2),
            RKSymbol::Heun => RKOrder::Explicit(2),
            RKSymbol::RK4 => RKOrder::Explicit(4),
            RKSymbol::RK21 => RKOrder::Adaptive((2, 1)),
            RKSymbol::RK23 => RKOrder::Adaptive((2, 3)),
            RKSymbol::RK45 => RKOrder::Adaptive((4, 5)),
            RKSymbol::Dopri5 => RKOrder::Adaptive((5, 4)),
            RKSymbol::Feh78 => RKOrder::Adaptive((7, 8)),
            RKSymbol::Other((_, order)) => order.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RKOrder {
    Explicit(usize),
    Adaptive((usize, usize)),
}

impl RKOrder {
    pub fn min(&self) -> usize {
        match *self {
            RKOrder::Explicit(o) => o,
            RKOrder::Adaptive((a, b)) => {
                if a < b {
                    a
                } else {
                    b
                }
            }
        }
    }

    pub fn max(&self) -> usize {
        match *self {
            RKOrder::Explicit(o) => o,
            RKOrder::Adaptive((a, b)) => {
                if a < b {
                    b
                } else {
                    a
                }
            }
        }
    }
}

/// Tableua of the form
///
/// ```text
///  c1  | a_11   ....   a_1s
///  .   | a_21 .          .
///  .   | a_31     .      .
///  .   | ....         .  .
///  c_s | a_s1  ....... a_ss
/// -----+--------------------
///      | b_1     ...   b_s   this is the one used for stepping
///      | b'_1    ...   b'_s  this is the one used for error-checking
/// ```
///
/// where `T` is the type of the coefficients
/// and `S` is the number of stages (an int)
#[derive(Debug, Clone)]
pub struct ButcherTableau<S: Dim, T: RealField = f64>
where
    DefaultAllocator:
        Allocator<T, U1, S> + Allocator<T, S, U2> + Allocator<T, S, S> + Allocator<T, S>,
{
    /// identifier for the rk method
    pub symbol: RKSymbol,
    /// coefficients - rk matrix
    pub a: MatrixN<T, S>,
    /// weights, adaptive weights are column major, means 2 fixed columns and `S` rows
    pub b: Weights<S, T>,
    /// nodes
    pub c: VectorN<T, S>,
}

#[derive(Debug, Clone)]
pub enum WeightType {
    Explicit,
    Adaptive,
}

#[derive(Debug, Clone)]
pub enum Weights<S: Dim, N: RealField = f64>
where
    DefaultAllocator: Allocator<N, S> + Allocator<N, S, U2>,
{
    Explicit(VectorN<N, S>),
    /// Adaptive weights where
    /// column 1 is used for stepping
    /// column 2 is used for error-checking
    Adaptive(MatrixMN<N, S, U2>),
}

impl<S: Dim, N: RealField> Weights<S, N>
where
    DefaultAllocator: Allocator<N, S> + Allocator<N, S, U2>,
{
    pub fn as_slice(&self) -> &[N] {
        match self {
            Weights::Explicit(e) => e.as_slice(),
            Weights::Adaptive(a) => a.as_slice(),
        }
    }
}

/// https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
impl<S: Dim, T: RealField> ButcherTableau<S, T>
where
    DefaultAllocator:
        Allocator<T, U1, S> + Allocator<T, S, U2> + Allocator<T, S, S> + Allocator<T, S>,
{
    /// the Butcher-Barrier says, that the amount of stages grows faster tha the order.
    /// For `nstages` ≥ 5, more than order 5 is required to solve the system
    #[inline]
    pub fn order(&self) -> RKOrder {
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
    #[inline]
    pub fn is_consistent_rk(&self) -> bool {
        for (j, i) in (1..self.nstages()).enumerate() {
            if self.a[(i, j)] != self.c[i] {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn is_explicit(&self) -> bool {
        match &self.b {
            Weights::Explicit(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_adaptive(&self) -> bool {
        !self.is_explicit()
    }

    #[inline]
    pub fn weight_type(&self) -> WeightType {
        match &self.b {
            Weights::Explicit(_) => WeightType::Explicit,
            Weights::Adaptive(_) => WeightType::Adaptive,
        }
    }

    /// First same as last. c.f. H&W p.167
    #[inline]
    pub fn is_first_same_as_last(&self) -> bool {
        let row_idx = self.nstages() - 1;
        for (col, b) in self.b.as_slice().iter().enumerate().take(self.nstages()) {
            if self.a[(row_idx, col)] != *b {
                return false;
            }
        }
        self.c[row_idx] == T::one()
    }
}

impl<S: Dim, T: RealField> fmt::Display for ButcherTableau<S, T>
where
    DefaultAllocator:
        Allocator<T, U1, S> + Allocator<T, S, U2> + Allocator<T, S, S> + Allocator<T, S>,
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
            Weights::Explicit(explicit) => {
                write!(f, "\n       |")?;
                for b in explicit.iter() {
                    write!(f, " {:.3}", b)?;
                }
            }
            Weights::Adaptive(adapt) => {
                for col in 0..adapt.ncols() {
                    write!(f, "\n       |")?;
                    for row in 0..self.nstages() {
                        write!(f, " {:.3}", adapt[(row, col)])?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl ButcherTableau<U1> {
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

impl ButcherTableau<U2> {
    /// the midpoint method https://en.wikipedia.org/wiki/Midpoint_method
    ///
    /// ```text
    ///  0.000 | 0.000 0.000
    ///  0.500 | 0.500 0.000
    /// -------+------------
    ///        | 0.000 1.000
    /// ```
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

impl ButcherTableau<U4> {
    /// ```text
    ///  0.000 | 0.000 0.000 0.000 0.000
    ///  0.500 | 0.500 0.000 0.000 0.000
    ///  0.750 | 0.000 0.750 0.000 0.000
    ///  1.000 | 0.222 0.333 0.444 0.000
    /// -------+------------------------
    ///        | 0.292 0.250 0.333 0.125
    ///        | 0.111 0.333 0.444 0.000
    /// ```
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
        let b = Weights::Adaptive(Matrix4x2::new(
            7. / 24.,
            1. / 9.,
            0.25,
            1. / 3.,
            1. / 3.,
            4. / 9.,
            0.125,
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

impl ButcherTableau<U4> {
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
        let c = Vector4::new(0., 0.5, 0.5, 1.);
        let b = Weights::Explicit(Vector4::new(1. / 6., 1. / 3., 1. / 3., 1. / 6.));
        let a = Matrix4::new(
            0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 1., 0.,
        );

        Self {
            symbol: RKSymbol::RK4,
            a,
            b,
            c,
        }
    }
}

impl ButcherTableau<U6> {
    /// Fehlberg https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    /// Order of 4 with an error estimator of order 5
    ///
    /// ```text
    ///  0.000 | 0.000 0.000 0.000 0.000 0.000 0.000
    ///  0.250 | 0.250 0.000 0.000 0.000 0.000 0.000
    ///  0.375 | 0.094 0.281 0.000 0.000 0.000 0.000
    ///  0.923 | 0.879 -3.277 3.321 0.000 0.000 0.000
    ///  1.000 | 2.032 -8.000 7.173 -0.206 0.000 0.000
    ///  0.500 | -0.296 2.000 -1.382 0.453 -0.275 0.000
    /// -------+------------------------------------
    ///        | 0.116 0.000 0.549 0.535 -0.200 0.000
    ///        | 0.119 0.000 0.519 0.506 -0.180 0.036
    /// ```
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
        let b = Weights::Adaptive(Matrix6x2::new(
            25. / 216.,
            16. / 135.,
            0.,
            0.,
            1408. / 2565.,
            6656. / 12825.,
            2197. / 4104.,
            28561. / 56430.,
            -0.2,
            -0.18,
            0.,
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
impl ButcherTableau<U7> {
    /// ```text
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
    /// ```
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
            U7,
            U2,
            &[
                35. / 384.,
                5179. / 57600.,
                0.,
                0.,
                500. / 1113.,
                7571. / 16695.,
                125. / 192.,
                393. / 640.,
                -2187. / 6784.,
                -92097. / 339_200.,
                11. / 84.,
                187. / 2100.,
                0.,
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

impl ButcherTableau<U13> {
    ///  0.000 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    ///  0.074 | 0.074 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    ///  0.111 | 0.028 0.083 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    ///  0.167 | 0.042 0.000 0.125 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    ///  0.417 | 0.417 0.000 -1.562 1.562 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    ///  0.500 | 0.500 0.000 0.000 0.750 0.200 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    ///  0.833 | -0.231 0.000 0.000 1.157 -2.407 2.315 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    ///  0.167 | 0.103 0.000 0.000 0.000 0.271 -0.222 0.014 0.000 0.000 0.000 0.000 0.000 0.000
    ///  0.667 | 2.000 0.000 0.000 -8.833 15.644 -11.889 0.744 3.000 0.000 0.000 0.000 0.000 0.000
    ///  0.333 | -0.843 0.000 0.000 0.213 -7.230 5.759 -0.317 2.833 -0.083 0.000 0.000 0.000 0.000
    ///  1.000 | 0.581 0.000 0.000 -2.079 4.386 -3.671 0.520 0.549 0.274 0.439 0.000 0.000 0.000
    ///  0.000 | 0.015 0.000 0.000 0.000 0.000 -0.146 -0.015 -0.073 0.073 0.146 0.000 0.000 0.000
    ///  1.000 | -0.433 0.000 0.000 -2.079 4.386 -3.524 0.535 0.622 0.201 0.293 0.000 1.000 0.000
    /// -------+------------------------------------------------------------------------------
    ///        | 0.049 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.324 0.324 0.257
    ///        | 0.257 0.257 0.257 0.032 0.032 0.032 0.032 0.049 0.000 0.000 0.049 0.000 0.049
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
            U13,
            U2,
            &[
                41. / 840.,
                9. / 35.,
                0.,
                9. / 35.,
                0.,
                9. / 35.,
                0.,
                9. / 280.,
                0.,
                9. / 280.,
                0.,
                9. / 280.,
                0.,
                9. / 280.,
                0.,
                41. / 840.,
                0.,
                0.,
                0.,
                0.,
                34. / 105.,
                41. / 840.,
                34. / 105.,
                0.,
                9. / 35.,
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
        assert!(ButcherTableau::dopri5().is_first_same_as_last());
    }
}
