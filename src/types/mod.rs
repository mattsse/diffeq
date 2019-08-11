use na::{Dim, DimName};
use num::Num;

/// api
/// pub fn solve<T:Solver(solver:T, fn() ->)

pub type Result<T> = ::std::result::Result<T, SolveError>;

pub trait Equation<T: Num, Data: DiffEquationSystem<T>>: Fn(T, Data) -> Data {}

#[derive(Debug)]
pub enum SolveError {
    // TODO impl
}

pub trait Solver<T, Data>
where
    Data: DiffEquationSystem<T>,
    T: Num,
{
    fn solve<F>(&self, f: F) -> Result<SolveSolution<T, Data>>
    where
        F: Fn(T, Data) -> Data;
}

#[derive(Debug)]
pub struct SolveSolution<T, Data>
where
    Data: DiffEquationSystem<T>,
    T: Num,
{
    pub size: usize,
    time: Vec<T>,
    data: Vec<Data>,
}

impl<T, Data> SolveSolution<T, Data>
where
    Data: DiffEquationSystem<T>,
    T: Num,
{
    pub fn data_points(&self) -> Vec<(&T, &Data)> {
        self.time.iter().zip(self.data.iter()).collect()
    }
}

// TODO rename this properly
// TODO impl for MatrixNM and VectorN from na crate
// TODO replace with MatrixNM instead?!! prio
// TODO needs to be indexable
pub trait DiffEquationSystem<Inner: Num>: Dim + DimName {
    // TODO rename; should accept a closure to apply
    fn apply(&self) -> Self;
}

#[derive(Debug)]
pub struct Point2D<T: Num> {
    pub x: T,
    pub y: T,
}

#[derive(Debug)]
pub struct Point3D<T: Num> {
    pub x: T,
    pub y: T,
    pub z: T,
}

// TODO write macro to impl DiffEquSystem for all primitive types

#[derive(Debug, Clone, PartialEq)]
pub enum Order {
    None,
    First,
    Second,
    Third,
    Fourth,
    Fifth,
    Higher(u8),
    Dim2(u8, u8),
}
