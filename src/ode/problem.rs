use crate::error::{Error, Result};
use crate::ode::options::{AdaptiveOptions, OdeOptionMap};
use crate::ode::runge_kutta::{ButcherTableau, Step};
use crate::ode::types::{OdeType, OdeTypeIterator, PNorm};
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
pub struct OdeProblem<F, Y>
where
    F: Fn(f64, &Y) -> Y,
    Y: OdeType,
{
    /// the RHS of the ODE dy/dt = F(t,y), which is a function of t and y(t) and returns the derivatives of y
    f: F,
    /// initial value for `Rhs` input
    /// determines the element type of the `yout` vector of the solutions
    y0: Y,
    /// sorted t values at which the solution (y) is requested
    tspan: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct OdeBuilder<F, Y>
where
    F: Fn(f64, &Y) -> Y,
    Y: OdeType,
{
    f: Option<F>,
    y0: Option<Y>,
    tspan: Option<Vec<f64>>,
}

impl<F, Y> OdeBuilder<F, Y>
where
    F: Fn(f64, &Y) -> Y,
    Y: OdeType,
{
    /// set the problem function
    pub fn fun(mut self, f: F) -> Self {
        self.f = Some(f);
        self
    }

    /// set the initial starting point
    pub fn init<T: Into<Y>>(mut self, y0: T) -> Self {
        self.y0 = Some(y0.into());
        self
    }

    /// set the time span for the problem
    pub fn tspan(mut self, tspan: Vec<f64>) -> Self {
        self.tspan = Some(tspan);
        self
    }

    /// creates a new tspan with `n` items from `from` to `to`
    pub fn tspan_linspace(mut self, from: f64, to: f64, n: usize) -> Self {
        self.tspan = Some(itertools_num::linspace(from, to, n).collect());
        self
    }

    fn build(self) -> Result<OdeProblem<F, Y>> {
        let f = self
            .f
            .ok_or(Error::uninitialized("Required problem must be initialized"))?;
        let y0 = self.y0.ok_or(Error::uninitialized(
            "Initial starting point must be initialized",
        ))?;
        let tspan = self
            .tspan
            .ok_or(Error::uninitialized("Time span must be initialized"))?;

        Ok(OdeProblem { f, y0, tspan })
    }
}

impl<F, Y> Default for OdeBuilder<F, Y>
where
    F: Fn(f64, &Y) -> Y,
    Y: OdeType,
{
    fn default() -> Self {
        Self {
            f: None,
            y0: None,
            tspan: None,
        }
    }
}

// TODO fix Into<f64>
impl<F, Y, T> OdeProblem<F, Y>
where
    F: Fn(f64, &Y) -> Y,
    T: RealField + Add<f64, Output = T> + Mul<f64, Output = T> + Into<f64>,
    Y: OdeType<Item = T>,
{
    /// convenience method to create a new builder
    /// same as `OdeBuilder::default()`
    pub fn builder() -> OdeBuilder<F, Y> {
        OdeBuilder::default()
    }

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
        println!("{:?}", ks[0]);
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
                    for d in 0..yn.dof() {
                        *yi.get_mut(d) += k.get(d) * dt * a;
                    }
                }
            }
            // compute the next k value
            ks.push((self.f)(tn, &yi));
        }

        ks
    }

    /// estimator for initial step based on book
    /// "Solving Ordinary Differential Equations I" by Hairer et al., p.169
    /// Returns first step, direction of integration and F evaluated at t0
    pub fn hinit(
        &self,
        x0: &Y,
        t0: f64,
        tend: f64,
        order: usize,
        reltol: f64,
        abstol: f64,
    ) -> InitialHint<Y> {
        let tdir = signum(tend - t0);
        assert_ne!(0., tdir);

        let norm = x0.pnorm(PNorm::InfPos);
        let one = Y::Item::one();

        let tau = (norm * reltol).max(one * abstol);

        let d0 = norm / tau;
        let f0 = (self.f)(t0, x0);

        let d1 = f0.pnorm(PNorm::InfPos) / tau;

        let h0: f64 = if d0 < one * 1e-5 || d1 < one * 1e-5 {
            1.0e-6
        } else {
            0.001 * (d0 / d1).into()
        };

        // perform Euler step, in every dimension
        let mut x1 = x0.clone();
        for d in 0..x1.dof() {
            *x1.get_mut(d) += (f0.get(d) * h0 * tdir);
        }
        let f1 = (self.f)(t0 + tdir * h0, &x1);

        // estimate second derivative
        let mut f1_0 = f1.clone();
        for d in 0..f1_0.dof() {
            *f1_0.get_mut(d) -= f0.get(d);
        }

        let d2 = f1_0.pnorm(PNorm::InfPos) / (tau * h0);

        let h1: f64 = if d1.max(d2) < one * 1e15 {
            1.0e-6.max(1.0e-3 * h0)
        } else {
            let pow = (2. + d1.max(d2).log10().into()) / (order as f64 + 1.);
            10f64.powf(pow)
        };

        let h = tdir * h1.min(100. * h0).min(tdir * (tend - t0));

        InitialHint { h, tdir, f0 }
    }
}

#[derive(Debug)]
pub struct InitialHint<Y> {
    /// step size hint
    h: f64,
    /// signum(tend - t0)
    tdir: f64,
    /// initial evaluation of the problem function
    f0: Y,
}

// TODO rm `T`, use f64 instead
#[derive(Debug)]
pub struct OdeSolution<T: RealField, Y: OdeType> {
    /// Vector of points at which solutions were obtained
    tout: Vec<T>,
    /// solutions at times `tout`, stored as a vector `yout`
    yout: Vec<Y>,
}

impl<T: RealField, Y: OdeType> fmt::Display for OdeSolution<T, Y> {
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
    const TF: f64 = 100.0;

    // Initial position in space
    const Y0: [f64; 3] = [0.1, 0.0, 0.0];

    // Constants SIGMA, RHO and beta
    const SIGMA: f64 = 10.0;
    const RHO: f64 = 28.0;
    const BET: f64 = 8.0 / 3.0;

    fn lorenz_attractor(t: f64, v: &Vec<f64>) -> Vec<f64> {
        let (x, y, z) = (v[0], v[1], v[2]);

        // Lorenz equations
        let dx_dt = SIGMA * (y - x);
        let dy_dt = x * (RHO - z) - y;
        let dz_dt = x * y - BET * z;

        // derivatives as vec
        vec![dx_dt, dy_dt, dz_dt]
    }

    #[test]
    fn lorenz_test() {
        let tspan: Vec<_> = itertools_num::linspace(0., TF, (TF / DT) as usize).collect();

        let problem = OdeProblem {
            f: lorenz_attractor,
            y0: vec![0.1, 0., 0.],
            tspan,
        };
    }

    #[test]
    fn hinit_test() {
        let problem = OdeProblem::builder().tspan_linspace(
            0., TF, (TF / DT) as usize
        ).fun(lorenz_attractor).init(vec![0.1, 0., 0.]).build().unwrap();

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
