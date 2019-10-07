use alga::general::RealField;
use na::{allocator::Allocator, ComplexField, DefaultAllocator, Dim, VectorN};
use num_traits::identities::{One, Zero};
use std::fmt;

use std::ops::{Add, Mul};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PNorm {
    P(usize),
    InfPos,
    InfNeg,
}

impl Default for PNorm {
    fn default() -> Self {
        PNorm::P(2)
    }
}

impl fmt::Display for PNorm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "norm(A, p=")?;
        match self {
            PNorm::P(p) => write!(f, "{})", p),
            PNorm::InfPos => write!(f, "Inf)"),
            PNorm::InfNeg => write!(f, "-Inf)"),
        }
    }
}

// TODO refactor api to support errors and options

// add default to item
pub trait OdeType: Clone + std::fmt::Debug {
    type Item: RealField + Add<f64, Output = Self::Item> + Mul<f64, Output = Self::Item>;

    // TODO rm this fn and Default bound

    fn set_zero(&mut self) {
        for i in 0..self.dof() {
            self.insert(i, Self::Item::zero());
        }
    }

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

    // TODO look up norm (4.11) of http://www.hds.bme.hu/~fhegedus/00%20-%20Numerics/B1993%20Solving%20Ordinary%20Differential%20Equations%20I%20-%20Nonstiff%20Problems.pdf
    // page 169 a)
    /// compute the p-norm of the OdeIterable
    fn pnorm(&self, p: PNorm) -> Self::Item {
        match p {
            PNorm::InfPos => self.ode_iter().fold(Self::Item::zero(), |norm, item| {
                let abs = item.abs();
                if abs > norm {
                    abs
                } else {
                    norm
                }
            }),
            PNorm::InfNeg => self.ode_iter().fold(Self::Item::zero(), |norm, item| {
                let abs = item.abs();
                if abs < norm {
                    abs
                } else {
                    norm
                }
            }),
            PNorm::P(p) => self
                .ode_iter()
                .fold(Self::Item::zero(), |norm, item| {
                    norm + item.abs().powi(p as i32)
                })
                .powf(Self::Item::one() * (1. / p as f64)),
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

impl<T, D: Dim> OdeType for VectorN<T, D>
where
    T: RealField + Add<f64, Output = T> + Mul<f64, Output = T>,
    DefaultAllocator: Allocator<T, D>,
{
    type Item = T;

    #[inline]
    fn dof(&self) -> usize {
        self.nrows()
    }

    fn get(&self, index: usize) -> Self::Item {
        self[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut Self::Item {
        &mut self[index]
    }

    fn insert(&mut self, index: usize, item: Self::Item) {
        self[index] = item;
    }
}

impl<T> OdeType for Vec<T>
where
    T: RealField + Add<f64, Output = T> + Mul<f64, Output = T>,
{
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
            fn get(&self, _index: usize) -> Self::Item {
                *self
            }

            #[inline]
            fn get_mut(&mut self, _index: usize) -> &mut Self::Item {
                self
            }

            #[inline]
            fn insert(&mut self, _index: usize, item: Self::Item) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
