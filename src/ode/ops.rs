pub trait Zero: Sized {
    fn zero() -> Self;
}

pub trait Normed {
    fn norm(&self) -> f64;
}

pub trait Norm<Rhs = Self> {
    type Output;

    fn norm(self, rhs: Rhs) -> Self::Output;
}
