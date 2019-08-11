use alga::linear::FiniteDimInnerSpace;
use na::allocator::Allocator;
use na::dimension::Dim;
use na::{DefaultAllocator, RealField, Unit, Vector2, Vector3, VectorN};
use num::Num;

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

// T is the type of the coefficients
// S is the number of stages (an int)
//pub struct ButcherTableau<T: Scalar, S: Dim + DimName> {
//    pub a: MatrixN<T, S>,
//    pub b: Step<T, S>,
//    pub c: VectorN<T, S>,
//}

//pub enum Step<T: Scalar, S: Dim + DimName> {
//    Fixed(MatrixMN<T, U1, S>),
//    Adaptive(MatrixMN<T, U2, S>),
//}
