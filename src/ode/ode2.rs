use crate::ode::options::OdeOptionMap;
use crate::ode::runge_kutta::ButcherTableau;
use alga::general::RealField;
use na::{allocator::Allocator, DefaultAllocator, Dim, U1, U2};
use std::iter::FromIterator;
use std::ops::{Add, Index, IndexMut, Mul};

// TODO figure out if this is useful as type alias
//pub type OdeFunction<T: RealField, Y: RealField> = dyn Fn(T, &[Y]) -> Vec<Y>;

// TODO create trait for common params, scalar, Vec, Tuple
// TODO or struct?
pub trait OdeType<T: RealField>: Clone {
    /// degree of freedom
    fn dof(&self) -> usize;

    fn get(&self, index: usize) -> T;

    fn set(&mut self, index: usize, item: T);
}

impl<T: RealField> OdeType<T> for Vec<T> {
    fn dof(&self) -> usize {
        self.len()
    }

    fn get(&self, index: usize) -> T {
        self[index]
    }

    fn set(&mut self, index: usize, item: T) {
        self[index] = item;
    }
}

macro_rules! impl_ode_real {
    ($($ty:ty),*) => {

    $(
    impl OdeType<$ty> for $ty {
        fn dof(&self) -> usize {
            1
        }

        fn get(&self, index: usize) -> $ty {
            *self
        }

        fn set(&mut self, index: usize, item: $ty) {
            *self = item;
        }
    })*

    };
}

impl_ode_real!(f64, f32);

#[derive(Debug, Clone)]
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

#[derive(Default)]
pub struct KTable<Y: RealField> {
    pub ks: Vec<Vec<Y>>,
}

impl<Y: RealField> KTable<Y> {
    pub fn with_capacity(stages: usize) -> Self {
        Self {
            ks: Vec::with_capacity(stages),
        }
    }
}

impl<Y: RealField> FromIterator<Vec<Y>> for KTable<Y> {
    fn from_iter<T: IntoIterator<Item = Vec<Y>>>(iter: T) -> Self {
        unimplemented!()
    }
}

// TODO iterator that returns a new ktable in each session, need tspan as input

pub struct K<'a, T: RealField, Y, S: Dim, Rhs>
where
    Y: RealField + Add<T, Output = Y> + Mul<T, Output = Y>,
    Rhs: Fn(T, &[Y]) -> Vec<Y>,
    DefaultAllocator:
        Allocator<T, U1, S> + Allocator<T, U2, S> + Allocator<T, S, S> + Allocator<T, S>,
{
    idx: usize,
    f: &'a Rhs,
    yn: &'a Vec<Vec<Y>>,
    btab: &'a ButcherTableau<T, S>,
    /// the time to approximate
    tspan: &'a Vec<T>,
    // also called h
    //    step_size: T,
}

impl<'a, T: RealField, Y, S: Dim, Rhs> K<'a, T, Y, S, Rhs>
where
    Rhs: Fn(T, &[Y]) -> Vec<Y>,
    Y: RealField + Add<T, Output = Y> + Mul<T, Output = Y>,
    DefaultAllocator:
        Allocator<T, U1, S> + Allocator<T, U2, S> + Allocator<T, S, S> + Allocator<T, S>,
{
    pub fn new(
        f: &'a Rhs,
        yn: &'a Vec<Vec<Y>>,
        tspan: &'a Vec<T>,
        btab: &'a ButcherTableau<T, S>,
    ) -> Self {
        Self {
            idx: 0,
            tspan,
            yn,
            f,
            btab,
        }
    }
}

// streaming iterator or redesign
impl<'a, T: RealField, Y, S: Dim, Rhs> Iterator for K<'a, T, Y, S, Rhs>
where
    Rhs: Fn(T, &[Y]) -> Vec<Y>,
    Y: RealField + Add<T, Output = Y> + Mul<T, Output = Y>,
    DefaultAllocator:
        Allocator<T, U1, S> + Allocator<T, U2, S> + Allocator<T, S, S> + Allocator<T, S>,
{
    type Item = Vec<Vec<Y>>;

    /// ```latex
    /// k_{s}&=f(t_{n}+c_{s}h,y_{n}+h(a_{s1}k_{1}+a_{s2}k_{2}+\cdots +a_{s,s-1}k_{s-1}))}
    /// ```
    /// returns all ks values for each step
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.tspan.len() {
            return None;
        }

        let ks: Vec<Vec<Y>> = Vec::with_capacity(self.btab.nstages());

        //        let dof = self.yn.len();

        if self.idx == 0 {
            //            self.ks[0] = (self.f)(self.t, self.yn);
            //            return Some(self.ks[0].iter().collect());
        }

        for s in 0..self.btab.nstages() - 1 {
            //            let ks = &self.ks[s];

            //            for d in 0..dof {}
            // a_s,1 .. a_s,s-1
        }

        self.idx += 1;
        None
    }
}

// TODO enforce same dimensions Vec<Y> D: Dim? or leave it to the caller to construct the f properly
// or abandon Vec entirely and rely on the caller that the type impls the necessary types

pub struct OdeProblem<Rhs, Y>
where
    Rhs: Fn(f64, &[Y]) -> Vec<Y>,
    Y: RealField + Add<f64, Output = Y> + Mul<f64, Output = Y>,
{
    /// the RHS of the ODE dy/dt = F(t,y), which is a function of t and y(t) and returns the derivatives of y
    f: Rhs,
    /// initial value for `Rhs` input
    /// determines the element type of the `yout` vector of the solutions
    y0: Vec<Y>,
    /// sorted t values at which the solution (y) is requested
    tspan: Vec<f64>,
}

// TODO OdeOptions as field or solve parameter

impl<Rhs, Y> OdeProblem<Rhs, Y>
where
    Rhs: Fn(f64, &[Y]) -> Vec<Y>,
    Y: RealField + Add<f64, Output = Y> + Mul<f64, Output = Y>,
{
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
        let mut ys: Vec<Vec<Y>> = Vec::with_capacity(self.tspan.len());

        // insert y0 as initial point
        ys.push(self.y0.clone());

        // the dimension of the solution type, eg. Vec3
        let dof = self.y0.len();

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
                    yi[d] += k[d] * b[s] * dt;
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
    fn calc_ks<S: Dim>(
        &self,
        btab: &ButcherTableau<f64, S>,
        t: f64,
        yn: &Vec<Y>,
        dt: f64,
    ) -> Vec<Vec<Y>>
    where
        DefaultAllocator: Allocator<f64, U1, S>
            + Allocator<f64, U2, S>
            + Allocator<f64, S, S>
            + Allocator<f64, S>,
    {
        let mut ks = Vec::with_capacity(btab.nstages());

        // k1 is just the function call
        ks.push((self.f)(t, yn));

        // the dimensions of the solution type
        let dof = yn.len();

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
                        yi[d] += k[d] * dt * a;
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
pub struct OdeSolution<Tout: RealField, Yout: RealField> {
    /// Vector of points at which solutions were obtained
    tout: Vec<Tout>,
    /// solutions at times `tout`, stored as a vector `yout`
    yout: Vec<Vec<Yout>>,
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
}
