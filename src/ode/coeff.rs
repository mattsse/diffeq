use crate::ode::types::OdeType;

#[derive(Debug)]
pub struct CoefficientMap<Y: OdeType> {
    inner: Vec<CoefficientPoint<Y>>,
}

impl<Y: OdeType> CoefficientMap<Y> {
    #[inline]
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn ks(&self) -> Ks<Y> {
        Ks {
            inner: self.inner.iter(),
        }
    }

    #[inline]
    pub fn ys(&self) -> Ks<Y> {
        Ks {
            inner: self.inner.iter(),
        }
    }
}

impl<Y: OdeType> std::ops::Deref for CoefficientMap<Y> {
    type Target = Vec<CoefficientPoint<Y>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<Y: OdeType> std::ops::DerefMut for CoefficientMap<Y> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<Y: OdeType> IntoIterator for CoefficientMap<Y> {
    type Item = CoefficientPoint<Y>;
    type IntoIter = std::vec::IntoIter<CoefficientPoint<Y>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a, Y: OdeType> IntoIterator for &'a CoefficientMap<Y> {
    type Item = &'a CoefficientPoint<Y>;
    type IntoIter = std::slice::Iter<'a, CoefficientPoint<Y>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

pub struct Ks<'a, Y: OdeType> {
    inner: std::slice::Iter<'a, CoefficientPoint<Y>>,
}

impl<'a, Y: OdeType> Iterator for Ks<'a, Y> {
    type Item = &'a Y;

    #[inline]
    fn next(&mut self) -> Option<&'a Y> {
        self.inner.next().map(|coeff| &coeff.k)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

pub struct Ys<'a, Y: OdeType> {
    inner: std::slice::Iter<'a, CoefficientPoint<Y>>,
}

impl<'a, Y: OdeType> Iterator for Ys<'a, Y> {
    type Item = &'a Y;

    #[inline]
    fn next(&mut self) -> Option<&'a Y> {
        self.inner.next().map(|coeff| &coeff.y)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// pairs the coefficient `k` with it's approximation `y`
#[derive(Debug, Clone)]
pub struct CoefficientPoint<Y: OdeType> {
    pub k: Y,
    pub y: Y,
}

impl<Y: OdeType> CoefficientPoint<Y> {
    #[inline]
    pub fn new(k: Y, y: Y) -> Self {
        Self { k, y }
    }
}
