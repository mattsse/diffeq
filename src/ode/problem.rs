use crate::error::{Error, OdeError, Result};
use crate::ode::coeff::{CoefficientMap, CoefficientPoint};
use crate::ode::options::{AdaptiveOptions, OdeOp, OdeOptionMap, Points, StepTimeout};
use crate::ode::runge_kutta::{ButcherTableau, WeightType, Weights};
use crate::ode::solution::OdeSolution;
use crate::ode::types::{OdeType, PNorm};
use alga::general::RealField;
use na::{allocator::Allocator, DMatrix, DVector, DefaultAllocator, Dim, Dynamic, Matrix, U1, U2};
use num_traits::{abs, signum};
use std::fmt;
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
    pub fn build(self) -> Result<OdeProblem<F, Y>> {
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

    pub fn ode21(&self, opts: &OdeOptionMap) -> Result<OdeSolution<f64, Y>, OdeError> {
        self.oderk_adapt(&ButcherTableau::rk21(), opts)
    }

    pub fn ode23(&self, opts: &OdeOptionMap) -> Result<OdeSolution<f64, Y>, OdeError> {
        self.oderk_adapt(&ButcherTableau::rk23(), opts)
    }

    pub fn ode45(&self, opts: &OdeOptionMap) -> Result<OdeSolution<f64, Y>, OdeError> {
        self.ode45_dp(opts)
    }

    pub fn ode45_dp(&self, opts: &OdeOptionMap) -> Result<OdeSolution<f64, Y>, OdeError> {
        self.oderk_adapt(&ButcherTableau::dopri5(), opts)
    }

    pub fn ode45_fe(&self, opts: &OdeOptionMap) -> Result<OdeSolution<f64, Y>, OdeError> {
        self.oderk_adapt(&ButcherTableau::rk45(), opts)
    }

    pub fn ode78(&self, opts: &OdeOptionMap) -> Result<OdeSolution<f64, Y>, OdeError> {
        self.oderk_adapt(&ButcherTableau::feh78(), opts)
    }

    /// solve with adaptive Runge-Kutta methods
    fn oderk_adapt<S: Dim, Ops: Into<AdaptiveOptions>>(
        &self,
        btab: &ButcherTableau<S>,
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

        let mut t = self.tspan[0];
        let tend = self.tspan[self.tspan.len() - 1];
        let opts = opts.into();
        let minstep = opts
            .minstep
            .map_or_else(|| abs(tend - t) / 1e18, |step| step.0);

        let maxstep = opts
            .maxstep
            .map_or_else(|| abs(tend - t) / 2.5, |step| step.0);

        let reltol = opts.reltol.0;
        let abstol = opts.abstol.0;

        let init = self.hinit(&self.y0, t, tend, btab.symbol.order().min(), reltol, abstol)?;

        let mut dt = if opts.initstep.0 != 0. {
            if signum(opts.initstep.0) == init.tdir {
                opts.initstep.0
            } else {
                return Err(OdeError::InvalidInitstep);
            }
        } else {
            init.h
        };

        let mut timeout = 0usize;
        let order = btab.symbol.order().min();
        let mut diagnostics = Diagnostics::default();
        let norm = opts.norm.0;
        let mut last_step = false;

        let mut tspan: Vec<f64> = Vec::with_capacity(self.tspan.len());
        tspan.push(t);

        // store for the computed values
        let mut ys = Vec::with_capacity(self.tspan.len());
        ys.push(self.y0.clone());

        let mut coeff = CoefficientPoint::new(init.f0.clone(), self.y0.clone());

        let mut iter_fixed = 1usize;
        // integration loop
        loop {
            let coeffs = self.calc_coefficients(btab, t, coeff.clone(), dt);
            let y = ys[ys.len() - 1].clone();

            let (ytrial, yerr) = self.embedded_step(&y, &coeffs, t, dt, btab)?;

            // check error and find a new step size
            let step = self.stepsize_hw92(
                dt, init.tdir, &y, &ytrial, yerr, order, timeout, abstol, reltol, maxstep, norm,
            );
            timeout = step.timeout_ctn;

            if step.err < 1. {
                // accept step
                diagnostics.accepted_steps += 1;

                let f0 = &coeffs[0].k;
                let f1 = if btab.is_first_same_as_last() {
                    coeffs[btab.nstages() - 1].k.clone()
                } else {
                    (self.f)(t + dt, &ytrial)
                };

                // interpolate onto given output points
                if Points::Specified == opts.points {
                    while iter_fixed < self.tspan.len()
                        && (init.tdir * self.tspan[iter_fixed] < init.tdir * (t + dt) || last_step)
                    {
                        let yout = self.hermite_interp(
                            self.tspan[iter_fixed],
                            t,
                            dt,
                            &y,
                            &ytrial,
                            f0,
                            &f1,
                        );
                        ys.push(yout);
                        tspan.push(self.tspan[iter_fixed]);
                        iter_fixed += 1;
                    }
                } else {
                    // store at all new times which are < t+dt
                    while iter_fixed < self.tspan.len()
                        && init.tdir * t < init.tdir * self.tspan[iter_fixed]
                        && init.tdir * self.tspan[iter_fixed] < init.tdir * (t + dt)
                    {
                        let yout = self.hermite_interp(
                            self.tspan[iter_fixed],
                            t,
                            dt,
                            &y,
                            &ytrial,
                            f0,
                            &f1,
                        );
                        ys.push(yout);
                        tspan.push(self.tspan[iter_fixed]);
                        iter_fixed += 1;
                    }
                    // also store every step taken
                    ys.push(ytrial.clone());
                    tspan.push(t + dt);
                }

                coeff = CoefficientPoint::new(f1, ytrial);

                // break if this was the last step
                if last_step {
                    break;
                }

                // update t to the time at the end of current step:
                t += dt;
                dt = step.dt;

                // Hit end point exactly if next step within 1% of end
                if init.tdir * (t + dt + dt / 100.) >= init.tdir * tend {
                    dt = tend - t;
                    // next step is the last, if it succeeds
                    last_step = true;
                }
            } else if step.dt.abs() < minstep {
                // minimum step size reached
                break;
            } else {
                // redo step with smaller dt
                diagnostics.rejected_steps += 1;
                last_step = false;
                dt = step.dt;
                timeout = *StepTimeout::default();
            }
        }

        Ok(OdeSolution {
            yout: ys,
            tout: tspan,
        })
    }

    /// solve the problem using the Feuler Butchertableau
    pub fn ode1(self) -> OdeSolution<f64, Y> {
        self.oderk_fixed(&ButcherTableau::feuler())
    }

    fn oderk_fixed<S: Dim>(self, btab: &ButcherTableau<S>) -> OdeSolution<f64, Y>
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

    /// Solve stiff systems based on a modified Rosenbrock triple
    pub fn ode23s<S: Dim, Ops: Into<AdaptiveOptions>>(
        &self,
        opts: Ops,
    ) -> Result<OdeSolution<f64, Y>, OdeError> {
        if self.tspan.is_empty() {
            // nothing to solve
            return Ok(OdeSolution::default());
        }
        let mut t = self.tspan[0];
        let tend = self.tspan[self.tspan.len() - 1];
        let opts = opts.into();
        let reltol = opts.reltol.0;
        let abstol = opts.abstol.0;
        let minstep = opts
            .minstep
            .map_or_else(|| abs(tend - t) / 1e18, |step| step.0);
        let maxstep = opts
            .maxstep
            .map_or_else(|| abs(tend - t) / 2.5, |step| step.0);

        let two_sqrt = 2f64.sqrt();
        let d = 0.5 + two_sqrt;
        let e32 = 6. + two_sqrt;

        let mut init = if opts.initstep.0 == 0. {
            // initial guess at a step size
            self.hinit(&self.y0, t, tend, 3, reltol, abstol)?
        } else {
            InitialHint {
                h: opts.initstep.0,
                tdir: (tend - t).signum(),
                f0: (self.f)(t, &self.y0),
            }
        };

        init.h = init.tdir * init.h.abs().min(maxstep);

        let mut tout: Vec<f64> = Vec::with_capacity(self.tspan.len());
        // first output time
        tout.push(t);

        let mut yout = Vec::with_capacity(self.tspan.len());
        // first output solution
        yout.push(self.y0.clone());

        // get Jacobian of F wrt y0
        let mut jac = self.fdjacobian(&self.y0, t);

        let (m, n) = jac.shape();
        let identity = DMatrix::<T>::identity(m, n);

        let mut y = self.y0.clone();
        let mut f0 = DVector::from_iterator(y.dof(), init.f0.ode_iter());

        while (t - tend).abs() > 0. && minstep < init.h.abs() {
            if (t - tend).abs() < init.h.abs() {
                init.h = tend - t;
            }

            let mut w = identity.clone() - jac.clone() * (T::one() * (init.h * d));
            if jac.len() != 1 {
                //                W = lu( I - h*d*J )
                // TODO how rm clone?
                w = w.lu().lu_internal().clone();
            };
            // TODO try_inverse?

            // approximate time-derivative of f

            let mut fdt =
                DVector::from_iterator(y.dof(), (self.f)(t + init.h / 100., &y).ode_iter());

            for i in 0..fdt.dof() {
                let fdti = fdt[i] - init.f0.get(i);
                fdt[i] = fdti * ((init.h * d) / (init.h / 100.));
            }

            // modified Rosenbrock formula
            // inv(W) * (F0 + T)
            // TODO handle NoneError
            let w_inv = w.try_inverse().unwrap();
            let k1 = w_inv.clone() * (f0.clone() * fdt.clone());

            let mut f1y = y.clone();
            for i in 0..y.dof() {
                *f1y.get_mut(i) += (k1[i] * 0.5 * init.h);
            }

            let f1 = DVector::from_iterator(y.dof(), (self.f)(t + 0.5 * init.h, &f1y).ode_iter());
            let k2 = w_inv.clone() * (f1.clone() - k1.clone()) + k1.clone();

            let mut ynew = y.clone();
            for i in 0..ynew.dof() {
                *ynew.get_mut(i) += (k2[i] * init.h);
            }

            let f2 = DVector::from_iterator(y.dof(), (self.f)(t + init.h, &ynew).ode_iter());

            let k3 = w_inv
                * (f2.clone()
                    - ((k2.clone() - f1) * (T::one() * e32))
                    - ((k1.clone() - f0.clone()) * (T::one() * 2.))
                    + fdt);

            // error estimate
            let kerr = (k1.clone() - (k2.clone() * (T::one() * 2.)) + k3);
            // TODO impl Pnorm for Iterator type
            let mut etmp = y.clone();
            for i in 0..etmp.dof() {
                etmp.insert(i, kerr[i]);
            }
            let err = etmp.pnorm(PNorm::default()) * (init.h.abs() / 6.);

            // allowable error
            let delta = (y.pnorm(PNorm::default()).max(ynew.pnorm(PNorm::default())) * reltol)
                .max(T::one() * abstol);

            if err <= delta {
                // only points in tspan are requested
                // -> find relevant points in (t,t+h]
                for toi in &self.tspan {
                    if *toi > t && *toi <= t + init.h {
                        // rescale to (0,1]
                        let s = (*toi - t) / init.h;
                        // use interpolation formula to get solutions at t=toi
                        tout.push(*toi);

                        let ktmp = k1.clone() * (T::one() * (s * (1. - s) / (1. - 2. * d)))
                            + k2.clone() * (T::one() * (s * (s - 2. * d) / (1. - 2. * d)));
                        let mut ytmp = y.clone();
                        for i in 0..ytmp.dof() {
                            *ytmp.get_mut(i) += (ktmp[i] * init.h);
                        }
                        yout.push(ytmp);
                    }
                }

                if Points::All == opts.points && (tout[tout.len() - 1] != t + init.h) {
                    // add the intermediate points
                    tout.push(t + init.h);
                    yout.push(ynew.clone());
                }

                t = t + init.h;
                y = ynew;
                // use FSAL property
                f0 = f2;
                // get Jacobian of F wrt y for new solution
                jac = self.fdjacobian(&y, t);
            }

            let r: f64 = (delta / err).into();
            init.h = maxstep.min(r.powf(1. / 3.) * init.h.abs() * 0.8) * init.tdir;
        }

        Ok(OdeSolution { yout, tout })
    }

    /// ```latex
    /// e_{n+1}=h\sum _{i=1}^{s}(b_{i}-b_{i}^{*})k_{i}
    /// ```
    fn calc_error<S: Dim>(
        &self,
        coeffs: &CoefficientMap<Y>,
        btab: &ButcherTableau<S>,
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
        btab: &ButcherTableau<S>,
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

        // a coeffs in first row are zero
        for row in 1..btab.nstages() {
            // need a fresh y clone
            let mut yi = coeffs[0].y.clone();

            for (col, k) in coeffs.ks().enumerate() {
                // adapt in all dimensions
                for d in 0..yi.dof() {
                    *yi.get_mut(d) += k.get(d) * (btab.a[(row, col)] * dt);
                }
            }

            let tn = t + btab.c[row] * dt;
            // compute the next k value
            coeffs.push(CoefficientPoint::new((self.f)(tn, &yi), yi));
        }

        coeffs
    }

    /// Does one embedded R-K step updating ytrial, yerr and ks.
    fn embedded_step<S: Dim>(
        &self,
        yn: &Y,
        coeffs: &CoefficientMap<Y>,
        _t: f64,
        dt: f64,
        btab: &ButcherTableau<S>,
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
            for (s, k) in coeffs.ks().enumerate() {
                for d in 0..yn.dof() {
                    *ytrial.get_mut(d) += k.get(d) * b[(s, 0)];
                    *yerr.get_mut(d) += k.get(d) * b[(s, 1)];
                }
            }
            for d in 0..yn.dof() {
                yerr.insert(d, (ytrial.get(d) - yerr.get(d)) * dt);
                ytrial.insert(d, yn.get(d) + (ytrial.get(d) * dt));
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
    fn hermite_interp(&self, tquery: f64, t: f64, dt: f64, y0: &Y, y1: &Y, f0: &Y, f1: &Y) -> Y {
        let mut y = y0.clone();
        let theta = (tquery - t) / dt;

        for i in 0..y0.dof() {
            let val = (y0.get(i) * (1. - theta) + y1.get(i) * theta)
                + ((y1.get(i) - y0.get(i)) * (1. - 2. * theta)
                    + f0.get(i) * (theta - 1.) * dt
                    + f1.get(i) * theta * dt)
                    * theta
                    * (theta - 1.);

            y.insert(i, val);
        }
        y
    }

    /// Estimates the error and a new step size following Hairer & Wanner 1992, p167
    ///
    fn stepsize_hw92(
        &self,
        dt: f64,
        tdir: f64,
        x0: &Y,
        xtrial: &Y,
        mut xerr: Y,
        order: usize,
        mut timeout: usize,
        abstol: f64,
        reltol: f64,
        maxstep: f64,
        _norm: PNorm,
    ) -> StepHW92 {
        let fac = 0.8;
        let _facmax = 5.;
        let facmin = 0.2;

        for d in 0..x0.dof() {
            if std::f64::NAN == xtrial.get(d).into() {
                return StepHW92 {
                    err: 10.,
                    dt: facmin * dt,
                    timeout_ctn: *StepTimeout::default(),
                };
            }

            *xerr.get_mut(d) /= x0.get(d).norm1().max(xtrial.get(d).norm1()) * reltol + abstol;
        }

        let err = xerr.pnorm(PNorm::default()).into();

        let pow = 1. / (order + 1) as f64;
        let mut new_dt = maxstep.min(facmin.max((err.powi(-1).powf(pow) * fac).into()) * tdir * dt);

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
    fn hinit(
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
            *x1.get_mut(d) += f0.get(d) * h0 * tdir;
        }
        let f1 = (self.f)(t0 + tdir * h0, &x1);
        // estimate second derivative
        let mut f1_0 = f1.clone();
        for d in 0..f1_0.dof() {
            *f1_0.get_mut(d) -= f0.get(d);
        }
        let d2 = f1_0.pnorm(PNorm::InfPos) / (tau * h0);

        let h1: f64 = if d1.max(d2) < one * 1e-15f64 {
            1.0e-6f64.max(1.0e-3f64 * h0)
        } else {
            let pow = -(2. + d1.max(d2).log10().into()) / ((order + 1) as f64);
            10f64.powf(pow)
        };

        let h = tdir * h1.min(100. * h0).min(tdir * (tend - t0));

        Ok(InitialHint { h, tdir, f0 })
    }

    /// Crude forward finite differences estimator of Jacobian as fallback
    /// returns a NxN Matrix where N is the degree of freedom of the `OdeType` `y`
    pub fn fdjacobian(&self, x: &Y, t: f64) -> DMatrix<T> {
        let ftx = (self.f)(t, x);
        let lx = ftx.dof();

        let mut dfdx = DMatrix::<T>::zeros(lx, lx);
        for n in 0..lx {
            let mut xj = x.get(n);
            if xj == T::zero() {
                xj += T::one();
            }
            // The / 100. is heuristic
            let dxj = xj * 0.01;
            let mut tmp = x.clone();
            *tmp.get_mut(n) += dxj;
            let mut yj = (self.f)(t, &tmp);
            for m in 0..lx {
                let mut yi = yj.get(m);
                yi -= ftx.get(m);
                yi /= dxj;
                dfdx[(m, n)] = yi;
            }
        }
        dfdx
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

    fn lorenz_attractor(_t: f64, v: &Vec<f64>) -> Vec<f64> {
        let (x, y, z) = (v[0], v[1], v[2]);

        // Lorenz equations
        let dx_dt = SIGMA * (y - x);
        let dy_dt = x * (RHO - z) - y;
        let dz_dt = x * y - BET * z;

        // derivatives as vec
        vec![dx_dt, dy_dt, dz_dt]
    }

    fn lorenz_problem() -> OdeProblem<impl Fn(f64, &Vec<f64>) -> Vec<f64>, Vec<f64>> {
        OdeProblem::builder()
            .tspan_linspace(0., TF, 100_001)
            .fun(lorenz_attractor)
            .init(vec![0.1, 0., 0.])
            .build()
            .unwrap()
    }

    #[test]
    fn ode1_test() {
        lorenz_problem().ode1();
    }

    #[test]
    fn ode45_test() {
        let mut ops = OdeOptionMap::default();
        ops.insert(Points::option_name(), Points::Specified.into());
        let _solution = lorenz_problem().ode45(&OdeOptionMap::default()).unwrap();
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

    #[test]
    fn fdjacobian_test() {
        let problem = lorenz_problem();

        // dFdx[-10.0 10.0 0.0; 28.0 -1.0 -0.1; 0.0 0.1 -2.66667]
        let x = vec![0.1, 0.0, 0.0];

        let jac = problem.fdjacobian(&x, 0.0);

        println!("{:?}", jac);
    }
}
