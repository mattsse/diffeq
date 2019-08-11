pub mod ode2;
pub mod ops;
pub mod runge_kutta2;

// impl trait for Fn(dy, y, μ, t) and so on
use crate::types::*;

use num::Num;

mod runge_kutta;

/// root element for a OdeProblem
#[derive(Debug, Clone)]
pub struct OdeProblem<F, Data, Time>
where
    F: Fn(usize) -> usize,
{
    fun: F,
    time: Vec<Time>,
    data: Vec<Data>,
}

#[derive(Debug)]
pub struct ODE23 {}
#[derive(Debug)]
pub struct ODE23s {}

#[derive(Debug)]
pub struct ODE78 {}

#[derive(Debug)]
pub struct ODE4<T: Num, Data: DiffEquationSystem<T>> {
    pub tspan: Vec<T>,
    pub x0: Data,
    pub order: Order,
}

impl<T, Data> Solver<T, Data> for ODE4<T, Data>
where
    T: Num,
    Data: DiffEquationSystem<T>,
{
    fn solve<F>(&self, _f: F) -> Result<SolveSolution<T, Data>>
    where
        F: Fn(T, Data) -> Data,
    {
        unimplemented!()
    }
}

/// ODE ROSENBROCK Solve stiff differential equations, Rosenbrock method
fn ode_rosenbrock() {}
