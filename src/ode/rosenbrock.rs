use na::{allocator::Allocator, *};

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
            4.452_470_820_736,
            4.163_528_788_60,
            0.,
            0.,
            4.452_470_820_736,
            4.163_528_788_60,
            0.,
            0.,
        );

        let b = Vector4::new(
            3.957_503_746_63,
            4.624_892_388_36,
            0.617_477_263_873,
            1.282_612_945_6,
        );

        let c = Matrix4::new(
            0.,
            0.,
            0.,
            0.,
            -5.071_675_338_77,
            0.,
            0.,
            0.,
            6.020_152_728_65,
            0.159_750_068_467_3,
            0.,
            0.,
            -1.856_343_618_677,
            -8.505_380_858_19,
            -2.084_075_136_02,
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
