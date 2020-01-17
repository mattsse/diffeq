mod utils;

use diffeq::ode::problem::OdeProblem;
use diffeq::ode::solution::SolutionPoint;
use diffeq::ode::Ode;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, diffeq-example-wasm!");
}

//#[wasm_bindgen]
//#[derive(Clone)]
//pub struct Point3D {
//    x : f64,
//    y : f64,
//    z : f64,
//}
//
//#[wasm_bindgen]
//pub struct SolutionPoint {
//    time: f64,
//    y: Point3D
//}

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub ode: String,
    pub init: Vec<f64>,
    pub start_time: f64,
    pub end_time: f64,
    pub num_times: usize,
}

fn lorenz_attractor(_: f64, v: &Vec<f64>) -> Vec<f64> {
    let (x, y, z) = (v[0], v[1], v[2]);

    // Constants SIGMA, RHO and beta
    let sigma = 10.0;
    let rho = 28.0;
    let bet = 8.0 / 3.0;

    // Lorenz equations
    let dx_dt = sigma * (y - x);
    let dy_dt = x * (rho - z) - y;
    let dz_dt = x * y - bet * z;

    // derivatives as vec
    vec![dx_dt, dy_dt, dz_dt]
}

#[wasm_bindgen]
pub fn solve_lorenz_attractor(config: &JsValue) -> Result<JsValue, JsValue> {
    let config = config
        .into_serde::<Config>()
        .map_err(|_| JsValue::from_str("Failed to serialize solution"))?;

    let ode = Ode::from_str(&config.ode).map_err(|s| JsValue::from_str(&s))?;

    if config.init.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Can't solve for {} dimensional type",
            config.init.len()
        )));
    }

    let problem = OdeProblem::builder()
        .tspan_linspace(config.start_time, config.end_time, config.num_times)
        .fun(lorenz_attractor)
        .init(config.init)
        .build()
        .unwrap();

    let solution = problem
        .solve(ode, Default::default())
        .map_err(|e| JsValue::from_str(&format!("{}", e)))?;

    JsValue::from_serde(&solution).map_err(|_| JsValue::from_str("Failed to serialize solution"))
}
