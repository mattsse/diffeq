use na::allocator::Allocator;
use na::*;

#[derive(Clone, Debug)]
pub struct RosenbrockCoeffs<S: Dim>
where
    DefaultAllocator: Allocator<f64, S, S> + Allocator<f64, S>,
{
    pub gamma: f64,
    pub a: MatrixN<f64, S>,
    pub b: VectorN<f64, S>,
    pub c: MatrixN<f64, S>,
}

impl RosenbrockCoeffs<U4> {

    /// Kaps-Rentrop coefficients
    pub fn kr4() -> Self {
        let a = Matrix4::new(
            0.,
            0.,
            0.,
            0.,
            2.,
            0.,
            0.,
            0.,
            4.452470820736,
            4.16352878860,
            0.,
            0.,
            4.452470820736,
            4.16352878860,
            0.,
            0.,
        );

        let b = Vector4::new(3.95750374663, 4.62489238836, 0.617477263873, 1.28261294568);

        let c = Matrix4::new(
            0.,
            0.,
            0.,
            0.,
            -5.07167533877,
            0.,
            0.,
            0.,
            6.02015272865,
            0.1597500684673,
            0.,
            0.,
            -1.856343618677,
            -8.50538085819,
            -2.08407513602,
            0.0,
        );

        Self {
            gamma: 0.231,
            a,
            b,
            c,
        }
    }

    /// Shampine coefficients
    pub fn s4() -> Self {
        let a = Matrix4::new(
            0.,
            0.,
            0.,
            0.,
            2.,
            0.,
            0.,
            0.,
            48. / 25.,
            6. / 250.,
            0.,
            0.,
            48. / 25.,
            6. / 250.,
            0.,
            0.,
        );

        let b = Vector4::new(19. / 9., 1. / 2., 25. / 108., 125. / 108.);
        let c = Matrix4::new(
            0.,
            0.,
            0.,
            0.,
            -8.,
            0.,
            0.,
            0.,
            372. / 25.,
            12. / 5.,
            0.,
            0.,
            -112. / 125.,
            -54. / 125.,
            -2. / 5.,
            0.,
        );

        Self {
            gamma: 0.5,
            a,
            b,
            c,
        }
    }
}
