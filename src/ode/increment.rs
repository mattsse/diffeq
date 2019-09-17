use crate::ode::types::OdeType;

pub struct IncrementMap<Y: OdeType> {
    inner: Vec<IncrementValue<Y>>,
}

impl<Y: OdeType> IncrementMap<Y> {

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

impl<Y: OdeType> std::ops::Deref for IncrementMap<Y> {
    type Target = Vec<IncrementValue<Y>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<Y: OdeType> std::ops::DerefMut for IncrementMap<Y> {

    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<Y: OdeType> IntoIterator for IncrementMap<Y> {
    type Item = IncrementValue<Y>;
    type IntoIter = std::vec::IntoIter<IncrementValue<Y>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a, Y: OdeType> IntoIterator for &'a IncrementMap<Y> {
    type Item = &'a IncrementValue<Y>;
    type IntoIter = std::slice::Iter<'a, IncrementValue<Y>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

pub struct Ks<'a, Y: OdeType> {
    inner: std::slice::Iter<'a, IncrementValue<Y>>,
}

impl<'a, Y: OdeType> Iterator for Ks<'a, Y> {
    type Item = &'a Y;

    #[inline]
    fn next(&mut self) -> Option<&'a Y> {
        self.inner.next().map(|inc| &inc.k)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

pub struct Ys<'a, Y: OdeType> {
    inner: std::slice::Iter<'a, IncrementValue<Y>>,
}

impl<'a, Y: OdeType> Iterator for Ys<'a, Y> {
    type Item = &'a Y;

    #[inline]
    fn next(&mut self) -> Option<&'a Y> {
        self.inner.next().map(|inc| &inc.y)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// pairs the increment `k` with it's approximation `y`
#[derive(Debug, Clone)]
pub struct IncrementValue<Y: OdeType> {
    pub k: Y,
    pub y: Y,
}

impl<Y: OdeType> IncrementValue<Y> {

    #[inline]
    pub fn new(k: Y, y: Y) -> Self {
        Self { k, y }
    }
}
