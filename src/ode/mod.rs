// http://docs.juliadiffeq.org/latest/features/performance_overloads.html
pub trait OdeFunction {
    type u0;
}

// impl trait for Fn(dy, y, μ, t) and so on
