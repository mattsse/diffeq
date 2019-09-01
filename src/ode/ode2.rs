use crate::ode::options::{AdaptiveOptions, OdeOptionMap};
use crate::ode::runge_kutta::ButcherTableau;
use alga::general::RealField;
use na::{allocator::Allocator, DefaultAllocator, Dim, U1, U2};
use num_traits::abs;
use std::iter::FromIterator;
use std::ops::{Add, Index, IndexMut, Mul};
use std::str::FromStr;

pub trait OdeType: Clone {
    type Item: RealField + Add<f64, Output = Self::Item> + Mul<f64, Output = Self::Item>;

    /// degree of freedom
    fn dof(&self) -> usize;

    fn get(&self, index: usize) -> Self::Item;

    fn get_mut(&mut self, index: usize) -> &mut Self::Item;

    fn insert(&mut self, index: usize, item: Self::Item);

    #[inline]
    fn ode_iter(&self) -> OdeTypeIterator<Self> {
        OdeTypeIterator {
            index: 0,
            ode_ty: self,
        }
    }
}

pub struct OdeTypeIterator<'a, T: OdeType> {
    index: usize,
    ode_ty: &'a T,
}

impl<'a, T: OdeType> Iterator for OdeTypeIterator<'a, T> {
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.ode_ty.dof() {
            let next = self.ode_ty.get(self.index);
            self.index += 1;
            Some(next)
        } else {
            None
        }
    }
}

impl<T: RealField + Add<f64, Output = T> + Mul<f64, Output = T>> OdeType for Vec<T> {
    type Item = T;

    #[inline]
    fn dof(&self) -> usize {
        self.len()
    }

    #[inline]
    fn get(&self, index: usize) -> Self::Item {
        self[index]
    }

    #[inline]
    fn get_mut(&mut self, index: usize) -> &mut Self::Item {
        &mut self[index]
    }

    #[inline]
    fn insert(&mut self, index: usize, item: Self::Item) {
        self[index] = item;
    }
}

macro_rules! impl_ode_ty {
    ($($ty:ident),*) => {
        $(impl OdeType for $ty {
            type Item = $ty;

            #[inline]
            fn dof(&self) -> usize {
                1
            }

            #[inline]
            fn get(&self, index: usize) -> Self::Item {
                *self
            }

            #[inline]
            fn get_mut(&mut self, index: usize) -> &mut Self::Item {
                self
            }

            #[inline]
            fn insert(&mut self, index: usize, item: Self::Item) {
                *self = item;
            }
        })*
    };
}

macro_rules! impl_ode_tuple {
    ( [($( $ty:ident),+) => $dof:expr;$item:ident;$($idx:tt),+]) => {
        impl OdeType for ( $($ty),*) {
            type Item = $item;

            #[inline]
            fn dof(&self) -> usize {
                $dof
            }

            fn get(&self, index: usize) -> Self::Item {
                match index {
                    $(
                     _ if index == $idx => self.$idx,
                    )*
                    _=> panic!("index out of bounds: the len is {} but the index is {}", $dof, index)
                }
            }

            fn get_mut(&mut self, index: usize) -> &mut Self::Item {
                match index {
                    $(
                     _ if index == $idx => &mut self.$idx,
                    )*
                    _=> panic!("index out of bounds: the len is {} but the index is {}", $dof, index)
                }
            }

            fn insert(&mut self, index: usize, item: Self::Item) {
                match index {
                    $(
                     _ if index == $idx => { self.$idx = item },
                    )*
                    _=> panic!("index out of bounds: the len is {} but the index is {}", $dof, index)
                }
            }
        }
    };
}

impl_ode_ty!(f64);
//impl_ode_ty!(f64, f32);
impl_ode_tuple!([(f64, f64) => 2;f64;0,1]);
//impl_ode_tuple!([(f32, f32) => 2;f32;0,1]);
impl_ode_tuple!([(f64, f64, f64) => 3;f64;0,1,2]);
//impl_ode_tuple!([(f32, f32, f32) => 3;f32;0,1,2]);
impl_ode_tuple!([(f64, f64, f64, f64) => 4;f64;0,1,2,3]);
//impl_ode_tuple!([(f32, f32, f32, f32) => 4;f32;0,1,2,3]);
impl_ode_tuple!([(f64, f64, f64, f64, f64) => 5;f64;0,1,2,3,4]);
//impl_ode_tuple!([(f32, f32, f32, f32, f32) => 5;f32;0,1,2,3,4]);
impl_ode_tuple!([(f64, f64, f64, f64, f64, f64) => 6;f64;0,1,2,3,4,5]);
//impl_ode_tuple!([(f32, f32, f32, f32, f32, f32) => 6;f32;0,1,2,3,4,5]);
impl_ode_tuple!([(f64, f64, f64, f64, f64, f64, f64) => 7;f64;0,1,2,3,4,5,6]);
//impl_ode_tuple!([(f32, f32, f32, f32, f32, f32, f32) => 7;f32;0,1,2,3,4,5,6]);
impl_ode_tuple!([(f64, f64, f64, f64, f64, f64, f64, f64) => 8;f64;0,1,2,3,4,5,6,7]);
//impl_ode_tuple!([(f32, f32, f32, f32, f32, f32, f32, f32) => 8;f32;0,1,2,3,4,5,6,7]);
impl_ode_tuple!([(f64, f64, f64, f64, f64, f64, f64, f64, f64) => 9;f64;0,1,2,3,4,5,6,7,8]);
//impl_ode_tuple!([(f32, f32, f32, f32, f32, f32, f32, f32, f32) => 9;f32;0,1,2,3,4,5,6,7,8]);

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

impl FromStr for Ode {
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

// http://docs.juliadiffeq.org/latest/features/performance_overloads.html

/// F: the RHS of the ODE dy/dt = F(t,y), which is a function of t and y(t)
/// and returns dy/dt::typeof(y/t)
/// y0: initial value for y. The type of y0, promoted as necessary according to the numeric type used
/// for the times, determines the element type of the yout vector (yout::Vector{typeof(y0*one(t))})
/// tspan: Any iterable of sorted t values at which the solution (y) is requested.
/// Most solvers will only consider tspan\[0\] and tspan\[end\], and intermediary points will be
/// interpolated. If tspan\[0\] > tspan\[end\] the integration is performed backwards. The times are
/// promoted as necessary to a common floating-point type.
pub trait OdeSolver {
    /// Vector of points at which solutions were obtained
    type Tout;
    /// solutions at times tout, stored as a vector yout as described above.
    /// Note that if y0 is a vector, you can get a matlab-like matrix with hcat(yout...).
    type Yout;
}

pub struct OdeProblem<Rhs, Y>
where
    Rhs: Fn(f64, &Y) -> Y,
    Y: OdeType,
{
    /// the RHS of the ODE dy/dt = F(t,y), which is a function of t and y(t) and returns the derivatives of y
    f: Rhs,
    /// initial value for `Rhs` input
    /// determines the element type of the `yout` vector of the solutions
    y0: Y,
    /// sorted t values at which the solution (y) is requested
    tspan: Vec<f64>,
}

// TODO OdeOptions as field or solve parameter

impl<Rhs, Y, T> OdeProblem<Rhs, Y>
where
    Rhs: Fn(f64, &Y) -> Y,
    T: RealField + Add<f64, Output = T> + Mul<f64, Output = T>,
    Y: OdeType<Item = T>,
{
    pub fn ode45<S: Dim>(&self, opts: &OdeOptionMap) {
        self.oderk_adapt(&ButcherTableau::rk45(), opts)
    }

    /// solve with adaptive Runge-Kutta methods
    fn oderk_adapt<S: Dim, Ops: Into<AdaptiveOptions>>(
        &self,
        btab: &ButcherTableau<f64, S>,
        opts: Ops,
    ) where
        DefaultAllocator: Allocator<f64, U1, S>
            + Allocator<f64, U2, S>
            + Allocator<f64, S, S>
            + Allocator<f64, S>,
    {
        let mut ops = opts.into();
        let minstep = ops.minstep.map_or_else(
            || abs(self.tspan[self.tspan.len() - 1] - self.tspan[0]) / 1e18,
            |step| step.0,
        );

        let maxstep = ops.maxstep.map_or_else(
            || abs(self.tspan[self.tspan.len() - 1] - self.tspan[0]) / 2.5,
            |step| step.0,
        );
    }

    /// solve the problem using the Feuler Butchertableau
    pub fn ode1(&self, ops: &OdeOptionMap) -> OdeSolution<f64, Y> {
        self.oderk_fixed(&ButcherTableau::feuler())
    }

    fn oderk_fixed<S: Dim>(&self, btab: &ButcherTableau<f64, S>) -> OdeSolution<f64, Y>
    where
        DefaultAllocator: Allocator<f64, U1, S>
            + Allocator<f64, U2, S>
            + Allocator<f64, S, S>
            + Allocator<f64, S>,
    {
        // TODO apply point filter if set
        // stores the computed values
        let mut ys: Vec<Y> = Vec::with_capacity(self.tspan.len());

        // insert y0 as initial point
        ys.push(self.y0.clone());

        // the dimension of the solution type, eg. Vec3
        let dof = self.y0.dof();

        for i in 0..self.tspan.len() - 1 {
            let dt = self.tspan[i + 1] - self.tspan[i];
            let mut yi = ys[i].clone();

            // all weights
            let b = btab.b.as_slice();
            // loop over all stages and k values of the butcher tableau
            for (s, k) in self
                .calc_ks(btab, self.tspan[i], &ys[i], dt)
                .iter()
                .enumerate()
            {
                // adapt in all dimensions
                for d in 0..dof {
                    *yi.get_mut(d) += k.get(d) * b[s] * dt;
                }
            }

            ys.push(yi);
        }

        OdeSolution {
            tout: self.tspan.clone(),
            yout: ys,
        }
    }

    /// calculates all `k` values for a given value `yn` at a specific time `t`
    fn calc_ks<S: Dim>(&self, btab: &ButcherTableau<f64, S>, t: f64, yn: &Y, dt: f64) -> Vec<Y>
    where
        DefaultAllocator: Allocator<f64, U1, S>
            + Allocator<f64, U2, S>
            + Allocator<f64, S, S>
            + Allocator<f64, S>,
    {
        let mut ks: Vec<Y> = Vec::with_capacity(btab.nstages());

        // k1 is just the function call
        ks.push((self.f)(t, yn));

        // the dimensions of the solution type
        let dof = yn.dof();

        for s in 1..btab.nstages() {
            let tn = t + btab.c[s] * dt;

            // need a fresh yn
            let mut yi = yn.clone();

            // loop over all previous computed ks
            for k in &ks {
                // loop over a coefficients in row s
                for j in 0..btab.nstages() - 1 {
                    let a = btab.a[(s, j)];
                    // adapt in all dimensions
                    for d in 0..dof {
                        *yi.get_mut(d) += k.get(d) * dt * a;
                    }
                }
            }
            ks.push((self.f)(tn, &yi));
        }

        ks
    }
}
//
//impl<Rhs, Time, Y> OdeProblem<Rhs, Time, Y>
//where
//    Rhs: Fn(Time, &[Y]) -> Vec<Y>,
//    Y: RealField,
//    Time: RealField + std::cmp::Ord,
//{
//    #[inline]
//    pub fn sort_tspan(&mut self) {
//        self.tspan.sort()
//    }
//}

#[derive(Debug)]
pub struct OdeSolution<Tout: RealField, Yout: OdeType> {
    /// Vector of points at which solutions were obtained
    tout: Vec<Tout>,
    /// solutions at times `tout`, stored as a vector `yout`
    yout: Vec<Yout>,
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

pub trait OdeSystem {}

#[cfg(test)]
mod tests {
    use super::*;

    const dt: f64 = 0.001;
    const tf: f64 = 100.0;

    // Initial position in space
    const y0: [f64; 3] = [0.1, 0.0, 0.0];

    // Constants sigma, rho and beta
    const sigma: f64 = 10.0;
    const rho: f64 = 28.0;
    const bet: f64 = 8.0 / 3.0;

    #[test]
    fn lorenz() {
        fn f(t: f64, r: &[f64]) -> Vec<f64> {
            let (x, y, z) = (r[0], r[1], r[2]);

            let dx_dt = sigma * (y - x);
            let dy_dt = x * (rho - z) - y;
            let dz_dt = x * y - bet * z;

            vec![dx_dt, dy_dt, dz_dt]
        }

        let tspan = itertools_num::linspace(0., tf, 100).collect();

        let problem = OdeProblem {
            f,
            y0: y0.to_vec(),
            tspan,
        };

        println!("{:?}", problem.ode1(&OdeOptionMap::default()));
    }

    #[test]
    fn ode_tuple() {
        let mut t3 = (0., 1., 2.);
        assert_eq!(3, t3.dof());
        assert_eq!(0., t3.get(0));
        t3.insert(0, 2.);
        assert_eq!(2., t3.get(0));
        assert_eq!(vec![2., 1., 2.], t3.ode_iter().collect::<Vec<_>>());

        let mut t6 = (1., 2., 3., 4., 5., 6.);
        assert_eq!(6, t6.dof());
        t6.insert(5, 7.);
        assert_eq!(
            vec![1., 2., 3., 4., 5., 7.],
            t6.ode_iter().collect::<Vec<_>>()
        );
    }
}
