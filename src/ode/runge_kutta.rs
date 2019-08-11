use crate::types::DiffEquationSystem;
use na::allocator::*;
use na::*;
use num::{Num, Zero};

/// Hairer & Wanner  1996 Step Size Selection p.123

/// explicit:  (ERK), falls für alle i und j mit i ≤ j gilt, daÿ aij = 0 ist
/// implizit (IRK), falls ein aij 6= 0 für i ≤ j existiert.
/// Bei einem impliziten Runge-Kutta-Verfahren nden sich auch kj mit j ≥ i als Argumente
/// der Funktion f bei der Berechnung von ki
// TODO generic tableau factory methods or global constants?!
// TODO macro development worth it?

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
    Other(String),
}

/// see http://users.nphysics.org/t/using-nalgebra-in-generics/90/3
#[derive(Debug)]
pub struct RKTableau<R, C, T>
where
    R: Dim + DimName,
    C: Dim + DimName,
    DefaultAllocator: Allocator<T, R, C> + Allocator<T, C, C> + Allocator<T, C> + Allocator<i64, R>,
    T: Num + Scalar,
{
    pub symbol: RKSymbol,
    pub order: VectorN<i64, R>,
    pub a: MatrixN<T, C>,
    pub b: MatrixMN<T, R, C>,
    pub c: VectorN<T, C>,
}

impl<R, C, T> RKTableau<R, C, T>
where
    R: Dim + DimName,
    C: Dim + DimName,
    DefaultAllocator: Allocator<T, R, C> + Allocator<T, C, C> + Allocator<T, C> + Allocator<i64, R>,
    T: Num + Scalar,
{
    /// calculates the next k,
    // TODO closure as param,
    // TODO mutable vec or create new vec in place
    /// ks is matrix with columns = len(a), and rows len(dim(y0))
    pub fn calc_next_k<Time, Data>(
        &self,
        _dt: Time,
        _t: Time,
        dof: usize,
        _ks: &mut MatrixMN<Time, Data, C>,
    ) where
        Time: Num + Scalar,
        Data: DiffEquationSystem<Time>,
        DefaultAllocator: Allocator<Time, Data, C>,
    {
        let s = self.c.len();

        for _i in 1..s {
            for _d in 1..=dof {}
        }
    }

    pub fn is_valid(&self) -> bool {
        unimplemented!()
    }

    pub fn is_explicit(&self) -> bool {
        unimplemented!()
    }
    pub fn is_implicit(&self) -> bool {
        unimplemented!()
    }

    pub fn oderk_fixed<F, Data>(&self, _f: F, y0: Data, tspan: Vec<T>)
    where
        F: Fn(T, Data) -> Data,
        Data: DiffEquationSystem<T>,
    {
        let dof = y0.value();

        // rows for each time step, column size = dimension of the equation systems' type
        // TODO create uninitialized
        let _ys = MatrixMN::<T, _, _>::zeros_generic(Dynamic::new(tspan.len()), Dynamic::new(dof));
        //    let x = ys.get(1,1);

        // TODO see rosen p. 51
        let _s = self.a.ncols();
        // k_i =

        for i in 0..tspan.len() - 1 {
            // compute derivative
            let _dt = tspan[i + 1] - tspan[i];
        }

        for (_i, _t) in tspan.iter().enumerate() {}
    }
}

impl RKTableau<U1, U1, f64> {
    pub fn feuler() -> Self {
        let order = Vector1::new(1);
        let a = Matrix1::zero();
        let b = Matrix1::new(1.0);
        let c = Vector1::zero();

        RKTableau {
            symbol: RKSymbol::Feuler,
            order,
            a,
            b,
            c,
        }
    }
}
impl RKTableau<U1, U2, f64> {
    pub fn midpoint() -> Self {
        let order = Vector1::new(2);
        let a = Matrix2::new(0.0, 0.0, 0.5, 0.0);
        let b = Matrix1x2::new(0.0, 1.0);
        let c = Vector2::new(0.0, 0.5);

        RKTableau {
            symbol: RKSymbol::Midpoint,
            order,
            a,
            b,
            c,
        }
    }

    pub fn heun() -> Self {
        let order = Vector1::new(2);
        let a = Matrix2::new(0.0, 0.0, 1.0, 0.0);
        let b = Matrix1x2::new(0.5, 0.5);
        let c = Vector2::new(0.0, 0.1);

        RKTableau {
            symbol: RKSymbol::Heun,
            order,
            a,
            b,
            c,
        }
    }
}

impl RKTableau<U1, U4, f64> {
    pub fn rk4() -> Self {
        let order = Vector1::new(4);
        let a = Matrix4::new(
            0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        );
        let b = Matrix1x4::new(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0);
        let c = Vector4::new(0.0, 0.5, 0.5, 1.0);

        RKTableau {
            symbol: RKSymbol::RK4,
            order,
            a,
            b,
            c,
        }
    }
}

impl RKTableau<U2, U2, f64> {
    pub fn rk21() -> Self {
        let order = Vector2::new(2, 1);
        let a = Matrix2::new(0.0, 0.0, 1.0, 0.0);
        let b = Matrix2::new(0.5, 0.5, 1.0, 0.0);
        let c = Vector2::new(0.0, 1.0);

        RKTableau {
            symbol: RKSymbol::RK21,
            order,
            a,
            b,
            c,
        }
    }
}

impl RKTableau<U2, U4, f64> {
    pub fn rk23() -> Self {
        let order = Vector2::new(2, 3);
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
        let b = Matrix2x4::new(
            7.0 / 24.0,
            0.25,
            1.0 / 3.0,
            0.125,
            1.0 / 9.0,
            1.0 / 3.0,
            4.0 / 9.0,
            0.0,
        );
        let c = Vector4::new(0.0, 0.5, 0.75, 1.0);

        RKTableau {
            symbol: RKSymbol::RK23,
            order,
            a,
            b,
            c,
        }
    }
}

impl RKTableau<U2, U6, f64> {
    pub fn rk45() -> Self {
        let order = Vector2::new(4, 5);
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
        let b = Matrix2x6::new(
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
        );
        let c = Vector6::new(0.0, 0.25, 0.375, 12.0 / 13.0, 1.0, 0.5);

        RKTableau {
            symbol: RKSymbol::RK23,
            order,
            a,
            b,
            c,
        }
    }
}

impl RKTableau<U2, U7, f64> {
    pub fn dopri5() -> Self {
        let order = Vector2::new(5, 4);
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
        let b = MatrixMN::from_row_slice_generic(
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
        );
        let c =
            VectorN::from_row_slice_generic(U7, U1, &[0.0, 0.2, 0.3, 0.75, 8.0 / 9.0, 1.0, 1.0]);

        RKTableau {
            symbol: RKSymbol::Dopri5,
            order,
            a,
            b,
            c,
        }
    }
}

impl RKTableau<U2, U13, f64> {
    pub fn feh78() -> Self {
        let order = Vector2::new(2, 1);
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
        let b = MatrixMN::from_row_slice_generic(
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
        );
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

        RKTableau {
            symbol: RKSymbol::Feh78,
            order,
            a,
            b,
            c,
        }
    }
}
// TODO impl refactor
pub fn oderk_fixed<F, T, Data, R, C>(_f: F, y0: Data, tspan: Vec<T>, btab_: RKTableau<R, C, T>)
where
    F: Fn(T, Data) -> Data,
    Data: DiffEquationSystem<T>,
    T: Num + Scalar,
    R: Dim + DimName,
    C: Dim + DimName,
    DefaultAllocator: Allocator<T, R, C> + Allocator<T, C, C> + Allocator<T, C> + Allocator<i64, R>,
{
    let dof = y0.value();

    // rows for each time step, column size = dimension of the equation systems' type
    // TODO create uninitialized
    let _ys = MatrixMN::<T, _, _>::zeros_generic(Dynamic::new(tspan.len()), Dynamic::new(dof));
    //    let x = ys.get(1,1);

    // TODO see rosen p. 51
    let _s = btab_.a.ncols();
    // k_i =

    for (_i, _t) in tspan.iter().enumerate() {}
}

pub fn oderk_adapt<F, T, Data, R, C>(_f: F, y0: Data, tspan: Vec<T>, _btab_: RKTableau<R, C, T>)
where
    F: Fn(T, Data) -> Data,
    Data: DiffEquationSystem<T>,
    T: Num + Scalar,
    R: Dim + DimName,
    C: Dim + DimName,
    DefaultAllocator: Allocator<T, R, C> + Allocator<T, C, C> + Allocator<T, C> + Allocator<i64, R>,
{
    let dof = y0.value();
    let _ys = MatrixMN::<T, _, _>::zeros_generic(Dynamic::new(tspan.len()), Dynamic::new(dof));
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn feuler() {
        let _feuler = RKTableau::feuler();
    }
}
