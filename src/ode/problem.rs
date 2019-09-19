use crate::error::{Error, OdeError, Result};
use crate::ode::coeff::{CoefficientMap, CoefficientPoint};
use crate::ode::options::{AdaptiveOptions, OdeOptionMap, StepTimeout};
use crate::ode::runge_kutta::{ButcherTableau, WeightType, Weights};
use crate::ode::types::{OdeType, OdeTypeIterator, PNorm};
use alga::general::{RealField, SupersetOf};
use itertools::Itertools;
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
#[derive(Debug, Clone)]
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

    /// creates a new OdeProblem
    /// returns an error if any field is None
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

    pub fn ode45<S: Dim>(&self, opts: &OdeOptionMap) -> Result<OdeSolution<f64, Y>, OdeError> {
        self.oderk_adapt(&ButcherTableau::rk45(), opts)
    }

    /// solve with adaptive Runge-Kutta methods
    fn oderk_adapt<S: Dim, Ops: Into<AdaptiveOptions>>(
        &self,
        btab: &ButcherTableau<f64, S>,
        opts: Ops,
    ) -> Result<OdeSolution<f64, Y>, OdeError>
    where
        DefaultAllocator: Allocator<f64, U1, S>
            + Allocator<f64, S, U2>
            + Allocator<f64, S, S>
            + Allocator<f64, S>,
    {
        if !btab.is_adaptive() {
            return Err(OdeError::InvalidButcherTableauWeightType {
                expected: WeightType::Adaptive,
                got: WeightType::Explicit,
            });
        }

        if self.tspan.is_empty() {
            // nothing to solve
            return Ok(OdeSolution::default());
        }

        // store for the computed values
        let mut ys: Vec<Y> = Vec::with_capacity(self.tspan.len());

        let mut t = self.tspan[0];
        let tend = self.tspan[self.tspan.len() - 1];
        let mut opts = opts.into();
        let minstep = opts
            .minstep
            .map_or_else(|| abs(tend - t) / 1e18, |step| step.0);

        let maxstep = opts
            .maxstep
            .map_or_else(|| abs(tend - t) / 2.5, |step| step.0);

        let reltol = opts.reltol.0;
        let abstol = opts.abstol.0;

        let init = self.hinit(&self.y0, t, tend, btab.symbol.order().min(), reltol, abstol)?;
        let mut dt = if signum(opts.initstep.0) == init.tdir {
            opts.initstep.0
        } else {
            return Err(OdeError::InvalidInitstep);
        };

        // integration loop
        let mut timeout = 0usize;
        let order = btab.symbol.order().min();
        let mut diagnostics = Diagnostics::default();
        let norm = opts.norm.0;

        // TODO filtering if not every point is required
        let mut tspan: Vec<f64> = Vec::with_capacity(self.tspan.len());

        let mut coeff = CoefficientPoint::new(init.f0.clone(), self.y0.clone());
        loop {
            // k0 is just the function call
            let coeffs = self.calc_coefficients(btab, t, coeff.clone(), dt);

            let y = &self.y0;
            let (ytrial, mut yerr) = self.embedded_step(y, &coeffs, t, dt, btab)?;

            // check error and find a new step size

            let step = self.stepsize_hw92(
                dt, init.tdir, y, &ytrial, &mut yerr, order, timeout, abstol, reltol, maxstep, norm,
            );

            if step.err < 1. {
                // accept step
                diagnostics.accepted_steps += 1;

                // TODO
                let f0 = &coeffs[0].k;
                let f1 = if btab.is_first_same_as_last() {
                    coeffs[btab.nstages() - 1].k.clone()
                } else {
                    (self.f)(t + dt, &ytrial)
                };
                // interpolate onto given output points

                // store at all new times which are < t+dt
                for t_iter in self.tspan.iter().take_while_ref(|t_iter| {
                    let tt = init.tdir * **t_iter;
                    init.tdir * t < tt && tt < init.tdir * (t + dt)
                }) {
                    let yout = self.hermite_interp(*t_iter, t, dt, y, &ytrial, f0, &f1);
                    ys.push(yout);
                    tspan.push(*t_iter);
                }
                // also store every step taken
                ys.push(ytrial);
                tspan.push(t + dt);

                coeff.k = f1;

                // Break if this was the last step:

                // Update t to the time at the end of current step:
                t += dt;
                dt = step.dt;

                // TODO
            } else if step.dt.abs() > minstep {
                // minimum step size reached
                break;
            } else {
                // redo step with smaller dt
                diagnostics.rejected_steps += 1;
            }
        }

        unimplemented!()
    }

    /// solve the problem using the Feuler Butchertableau
    pub fn ode1(self, ops: &OdeOptionMap) -> OdeSolution<f64, Y> {
        self.oderk_fixed(&ButcherTableau::feuler())
    }

    // TODO should this return an error on f64::NAN? how for `Realfield`?
    // TODO is providing an optionmap beneficial?
    fn oderk_fixed<S: Dim>(self, btab: &ButcherTableau<f64, S>) -> OdeSolution<f64, Y>
    where
        DefaultAllocator: Allocator<f64, U1, S>
            + Allocator<f64, S, U2>
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
                .calc_coefficients(
                    btab,
                    self.tspan[i],
                    CoefficientPoint::new((self.f)(self.tspan[i], &yi), yi.clone()),
                    dt,
                )
                .ks()
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
            tout: self.tspan,
            yout: ys,
        }
    }

    /// ```latex
    /// e_{n+1}=h\sum _{i=1}^{s}(b_{i}-b_{i}^{*})k_{i}
    /// ```
    fn calc_error<S: Dim>(
        &self,
        coeffs: &CoefficientMap<Y>,
        btab: &ButcherTableau<f64, S>,
        dt: f64,
    ) -> Result<Y, OdeError>
    where
        DefaultAllocator: Allocator<f64, U1, S>
            + Allocator<f64, S, U2>
            + Allocator<f64, S, S>
            + Allocator<f64, S>,
    {
        assert_eq!(btab.nstages(), coeffs.len());

        // get copy of the Odetype and ensure default values
        let mut err = coeffs[0].k.clone();
        err.set_zero();

        if let Weights::Adaptive(b) = &btab.b {
            for (s, k) in coeffs.ks().enumerate() {
                // adapt in every dimension
                for d in 0..err.dof() {
                    // subtract b_1s from b_0s
                    let weight_err = b[(s, 0)] - b[(s, 1)];
                    *err.get_mut(d) += k.get(d) * weight_err;
                }
            }
            // multiply with stepsize
            for d in 0..err.dof() {
                err.insert(d, err.get(d) * dt);
            }

            Ok(err)
        } else {
            Err(OdeError::InvalidButcherTableauWeightType {
                expected: WeightType::Adaptive,
                got: WeightType::Explicit,
            })
        }
    }

    /// calculates all coefficients values for a given value `yn` at a specific time `t`
    /// creates an `CoefficientMap` with the calculated coefficient `k` and their
    /// approximations `y` of size `S`, the number of stages of the butcher tableau
    pub fn calc_coefficients<S: Dim>(
        &self,
        btab: &ButcherTableau<f64, S>,
        t: f64,
        init: CoefficientPoint<Y>,
        dt: f64,
    ) -> CoefficientMap<Y>
    where
        DefaultAllocator: Allocator<f64, U1, S>
            + Allocator<f64, S, U2>
            + Allocator<f64, S, S>
            + Allocator<f64, S>,
    {
        let mut coeffs = CoefficientMap::with_capacity(btab.nstages());

        coeffs.push(init);

        for s in 1..btab.nstages() {
            let tn = t + btab.c[s] * dt;
            // need a fresh y
            let mut yi = coeffs[0].y.clone();

            // loop over all previous computed ks
            for k in coeffs.ks() {
                // loop over a coefficients in row s
                for j in 0..btab.nstages() - 1 {
                    let a = btab.a[(s, j)];
                    // adapt in all dimensions
                    for d in 0..yi.dof() {
                        *yi.get_mut(d) += k.get(d) * dt * a;
                    }
                }
            }
            // compute the next k value
            coeffs.push(CoefficientPoint::new((self.f)(tn, &yi), yi));
        }
        coeffs
    }

    /// Does one embedded R-K step updating ytrial, yerr and ks.
    pub fn embedded_step<S: Dim>(
        &self,
        yn: &Y,
        coeffs: &CoefficientMap<Y>,
        t: f64,
        dt: f64,
        btab: &ButcherTableau<f64, S>,
    ) -> Result<(Y, Y), OdeError>
    where
        DefaultAllocator: Allocator<f64, U1, S>
            + Allocator<f64, S, U2>
            + Allocator<f64, S, S>
            + Allocator<f64, S>,
    {
        // trial solution at time t+dt
        let mut ytrial = yn.clone();
        ytrial.set_zero();

        // error of trial solution
        let mut yerr = ytrial.clone();

        if let Weights::Adaptive(b) = &btab.b {
            for (s, k) in coeffs.ks().take(btab.nstages()).enumerate() {
                for d in 0..yn.dof() {
                    *ytrial.get_mut(d) += k.get(d) * b[(s, 0)];
                    *yerr.get_mut(d) += k.get(d) * b[(s, 1)];
                }
            }

            for d in 0..yn.dof() {
                ytrial.insert(d, yn.get(d) + ytrial.get(d) * dt);
                yerr.insert(d, (ytrial.get(d) - yerr.get(d)) * dt);
            }

            Ok((ytrial, yerr))
        } else {
            Err(OdeError::InvalidButcherTableauWeightType {
                expected: WeightType::Adaptive,
                got: WeightType::Explicit,
            })
        }
    }

    /// For dense output see Hairer & Wanner p.190 using Hermite
    ///  interpolation. Updates y in-place.
    /// f_0 = f(x_0 , y_0) , f_1 = f(x_0 + h, y_1 )
    pub fn hermite_interp(
        &self,
        tquery: f64,
        t: f64,
        dt: f64,
        y0: &Y,
        y1: &Y,
        f0: &Y,
        f1: &Y,
    ) -> Y {
        let mut y = y0.clone();
        let theta = (tquery - t) / dt;

        for i in 0..y0.dof() {
            let val = (y0.get(i) * (1. - theta)
                + y1.get(i) * theta
                + (y1.get(i) - y0.get(i))
                    * theta
                    * (theta - 1.)
                    * (f0.get(i) * (theta - 1.) * dt + f1.get(i) * theta * dt + (1. - 2. * theta)));
            y.insert(i, val);
        }
        y
    }

    /// Estimates the error and a new step size following Hairer & Wanner 1992, p167
    ///
    // TODO pass optionmap instead
    pub fn stepsize_hw92(
        &self,
        dt: f64,
        tdir: f64,
        x0: &Y,
        xtrial: &Y,
        xerr: &mut Y,
        order: usize,
        mut timeout: usize,
        abstol: f64,
        reltol: f64,
        maxstep: f64,
        norm: PNorm,
    ) -> StepHW92 {
        let fac = 0.8;
        let facmax = 5.;
        let facmin = 0.2;

        for d in 0..x0.dof() {
            if std::f64::NAN == xtrial.get(d).into() {
                return StepHW92 {
                    err: 10.,
                    dt: facmin * dt,
                    timeout_ctn: *StepTimeout::default(),
                };
            }

            *xerr.get_mut(d) /= (x0.get(d).norm1().max(xtrial.get(d).norm1()) * reltol + abstol);
        }

        let err = xerr.pnorm(PNorm::default()).into();

        let pow = (1 / (order + 1)) as i32;

        let mut new_dt = maxstep.min(facmin.max(err.powi(-1).powi(pow).into()) * tdir * dt);

        if timeout > 0 {
            new_dt = new_dt.min(dt);
            timeout -= 1;
        }

        StepHW92 {
            err,
            dt: tdir * new_dt,
            timeout_ctn: timeout,
        }
    }

    /// estimator for initial step based on book
    /// "Solving Ordinary Differential Equations I" by Hairer et al., p.169
    /// Returns first step, direction of integration and F evaluated at t0
    // TODO rm pub
    // TODO t0 and tend can prbly be removed as parameter
    pub fn hinit(
        &self,
        x0: &Y,
        t0: f64,
        tend: f64,
        order: usize,
        reltol: f64,
        abstol: f64,
    ) -> Result<InitialHint<Y>, OdeError> {
        let tdir = signum(tend - t0);
        if tdir == 0. {
            return Err(OdeError::ZeroTimeSpan);
        }

        let norm = x0.pnorm(PNorm::InfPos);
        let one = Y::Item::one();
        let tau = (norm * reltol).max(one * abstol);
        let d0 = norm / tau;
        let f0 = (self.f)(t0, x0);
        let d1 = f0.pnorm(PNorm::InfPos) / tau;

        let h0: f64 = if d0 < one * 1e-5 || d1 < one * 1e-5 {
            1.0e-6
        } else {
            0.01 * (d0 / d1).into()
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

        let h1: f64 = if d1.max(d2) < one * 1e-15 {
            1.0e-6.max(1.0e-3 * h0)
        } else {
            let pow = -(2. + d1.max(d2).log10().into()) / (order as f64 + 1.);
            10f64.powf(pow)
        };

        let h = tdir * h1.min(100. * h0).min(tdir * (tend - t0));

        Ok(InitialHint { h, tdir, f0 })
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

#[derive(Debug)]
pub struct StepHW92 {
    // TODO change to T RealField?
    err: f64,
    dt: f64,
    timeout_ctn: usize,
}

/// Contains some diagnostics of the integration.
#[derive(Clone, Copy, Debug, Default)]
pub struct Diagnostics {
    pub num_eval: u32,
    pub accepted_steps: u32,
    pub rejected_steps: u32,
}

impl fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Number of function evaluations: {}", self.num_eval)?;
        writeln!(f, "Number of accepted steps: {}", self.accepted_steps)?;
        write!(f, "Number of rejected steps: {}", self.rejected_steps)
    }
}

// TODO rm `T`, use f64 instead
#[derive(Debug)]
pub struct OdeSolution<T: RealField, Y: OdeType> {
    /// Vector of points at which solutions were obtained
    tout: Vec<T>,
    /// solutions at times `tout`, stored as a vector `yout`
    yout: Vec<Y>,
}

impl<T: RealField, Y: OdeType> Default for OdeSolution<T, Y> {
    fn default() -> Self {
        OdeSolution {
            tout: Vec::new(),
            yout: Vec::new(),
        }
    }
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

    const DT: f64 = 0.001;
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
    fn lorenz_attractor_test() {
        let s1 = lorenz_attractor(0., &vec![0.1, 0.0, 0.0]);
    }

    #[test]
    fn ode1_test() {
        let problem = OdeProblem::builder()
            .tspan_linspace(0., TF, 100001)
            .fun(lorenz_attractor)
            .init(vec![0.1, 0., 0.])
            .build()
            .unwrap();

        problem.ode1(&OdeOptionMap::default());
    }

    #[test]
    fn hinit_test() {
        let problem = OdeProblem::builder()
            .tspan_linspace(0., TF, 100001)
            .fun(lorenz_attractor)
            .init(vec![0.1, 0., 0.])
            .build()
            .unwrap();

        let y0 = vec![0.1, 0., 0.];

        let init = problem.hinit(&y0, 0.0, 100., 4, 1e-5 as f64, 1e-8 as f64);
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
