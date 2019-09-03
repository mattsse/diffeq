use crate::ode::options::{AdaptiveOptions, OdeOptionMap};
use crate::ode::runge_kutta::{ButcherTableau, Step};
use crate::ode::types::{OdeType, OdeTypeIterator};
use alga::general::RealField;
use na::{allocator::Allocator, DefaultAllocator, Dim, VectorN, U1, U2};
use num_traits::{abs, signum};
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Add, Mul};

/// F: the RHS of the ODE dy/dt = F(t,y), which is a function of t and y(t)
/// and returns dy/dt::typeof(y/t)
/// y0: initial value for y. The type of y0, promoted as necessary according to the numeric type used
/// for the times, determines the element type of the yout vector (yout::Vector{typeof(y0*one(t))})
/// tspan: Any iterable of sorted t values at which the solution (y) is requested.
/// Most solvers will only consider tspan\[0\] and tspan\[end\], and intermediary points will be
/// interpolated. If tspan\[0\] > tspan\[end\] the integration is performed backwards. The times are
/// promoted as necessary to a common floating-point type.
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
        // store for the computed values
        let mut ys: Vec<Y> = Vec::with_capacity(self.tspan.len());

        let mut ops = opts.into();
        let minstep = ops.minstep.map_or_else(
            || abs(self.tspan[self.tspan.len() - 1] - self.tspan[0]) / 1e18,
            |step| step.0,
        );

        let maxstep = ops.maxstep.map_or_else(
            || abs(self.tspan[self.tspan.len() - 1] - self.tspan[0]) / 2.5,
            |step| step.0,
        );

        // integration loop

        //        // loop over all stages and k values of the butcher tableau
        //        for (s, k) in self
        //            .calc_ks(btab, self.tspan[i], &ys[i], dt)
        //            .iter()
        //            .enumerate()
        //            {
        //
        //            }
    }

    /// solve the problem using the Feuler Butchertableau
    pub fn ode1(&self, ops: &OdeOptionMap) -> OdeSolution<f64, Y> {
        self.oderk_fixed(&ButcherTableau::feuler())
    }

    // TODO should this return an error on f64::NAN?
    fn oderk_fixed<S: Dim>(&self, btab: &ButcherTableau<f64, S>) -> OdeSolution<f64, Y>
    where
        DefaultAllocator: Allocator<f64, U1, S>
            + Allocator<f64, U2, S>
            + Allocator<f64, S, S>
            + Allocator<f64, S>,
    {
        // store for the computed values
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

    /// ```latex
    /// e_{n+1}=h\sum _{i=1}^{s}(b_{i}-b_{i}^{*})k_{i}
    /// ```
    fn calc_error<S: Dim>(&self, ks: &[Y], btab: &ButcherTableau<f64, S>, dt: f64) -> Y
    where
        DefaultAllocator: Allocator<f64, U1, S>
            + Allocator<f64, U2, S>
            + Allocator<f64, S, S>
            + Allocator<f64, S>,
    {
        assert_eq!(btab.nstages(), ks.len());

        // get copy of the Odetype and ensure default values
        let mut err = ks[0].clone();
        err.set_zero();

        if let Step::Adaptive(b) = &btab.b {
            for (s, k) in ks.iter().enumerate() {
                // adapt in every dimension
                for d in 0..err.dof() {
                    // subtract b_1s from b_0s
                    let weight_err = b[(0, s)] - b[(1, s)];
                    *err.get_mut(d) += k.get(d) * weight_err;
                }
            }
        }

        // multiply with stepsize
        for d in 0..err.dof() {
            let sum = err.get(d);
            *err.get_mut(d) = sum * dt;
        }

        err
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

    /// estimator for initial step based on book
    /// "Solving Ordinary Differential Equations I" by Hairer et al., p.169
    /// Returns first step, direction of integration and F evaluated at t0
    fn hinit(&self, x0: &Y, t0: f64, tend: f64, order: usize, reltol: f64, abstol: f64) {
        let tdir = signum(tend - t0);
        assert_ne!(0., tdir);
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
pub struct OdeSolution<T: RealField, Y: OdeType> {
    /// Vector of points at which solutions were obtained
    tout: Vec<T>,
    /// solutions at times `tout`, stored as a vector `yout`
    yout: Vec<Y>,
}

impl<T: RealField, Y: OdeType> fmt::Display for OdeSolution<T, Y>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(")?;

        fn slice_print<T: fmt::Debug>(f: &mut fmt::Formatter, items: &[T]) -> fmt::Result {
            write!(f, "[")?;
            let mut i = 0;
            while i < items.len() {
                if i == items.len() - 1 {
                    write!(f, "{:?}", items[i])?;
                } else {
                    write!(f, "{:?}, ", items[i])?;
                }
                if i > 8 && i < items.len() - 10 {
                    write!(f, "... ")?;
                    i = items.len() - 11;
                }
                i += 1;
            }
            write!(f, "]")
        }

        slice_print(f, &self.tout)?;
        write!(f, ", Vec{{{}}}", self.yout.len())?;
        slice_print(f, &self.yout)?;

        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.01;
    const TF: f64 = 10.0;

    // Initial position in space
    const Y0: [f64; 3] = [0.1, 0.0, 0.0];

    // Constants SIGMA, RHO and beta
    const SIGMA: f64 = 10.0;
    const RHO: f64 = 28.0;
    const BET: f64 = 8.0 / 3.0;

    #[test]
    fn lorenz() {
        fn f(t: f64, v: &Vec<f64>) -> Vec<f64> {
            let (x, y, z) = (v[0], v[1], v[2]);
            let u = BET * z;
            let dx_dt = SIGMA * (y - x);
            let dy_dt = x * (RHO - z) - y;
            let dz_dt = x * y - BET * z;

            vec![dx_dt, dy_dt, dz_dt]
        }

        let tspan: Vec<_> = itertools_num::linspace(0., TF, (TF / DT) as usize).collect();

        let problem = OdeProblem {
            f,
            y0: vec![0.1, 0., 0.],
            tspan,
        };

        println!("{}", problem.ode1(&OdeOptionMap::default()));
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
