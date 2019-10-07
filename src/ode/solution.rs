#[cfg(feature = "serde0")]
use serde::{Deserialize, Serialize};
use crate::ode::types::OdeType;
use alga::general::RealField;
use std::fmt;

/// pairs the timestamp with the corresponding calculated value`
#[cfg_attr(feature = "serde0", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct SolutionPoint<Y: OdeType, T: RealField = f64> {
    pub t: T,
    pub y: Y,
}

impl<Y: OdeType, T: RealField> SolutionPoint<Y, T> {
    #[inline]
    pub fn new(t: T, y: Y) -> Self {
        Self { t, y }
    }
}

#[cfg_attr(feature = "serde0", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct OdeSolution<T: RealField, Y: OdeType> {
    /// Vector of points at which solutions were obtained
    pub tout: Vec<T>,
    /// solutions at times `tout`, stored as a vector `yout`
    pub yout: Vec<Y>,
}

impl <T: RealField, Y: OdeType> OdeSolution<T,Y> {


    /// pair each timestep with the corresponding output
    #[inline]
    pub fn zipped(self) -> Vec<(T, Y)>{
        self.tout.into_iter().zip(self.yout).collect()
    }

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
