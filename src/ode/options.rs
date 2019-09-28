//#![allow(unused)]
use crate::ode::types::PNorm;
use std::collections::HashMap;
use std::fmt;

pub type OdeOptionMap = HashMap<&'static str, OdeOption>;

macro_rules! option_val {
    ($ops:ident rm $id:ident) => {
        $ops.remove($id::option_name()).and_then(|op| {
            if let OdeOption::$id(el) = op {
                Some(el)
            } else {
                None
            }
        })
    };

    ($ops:ident get $id:ident) => {
        $ops.get($id::option_name()).and_then(|op| {
            if let OdeOption::$id(el) = op {
                Some(el.clone())
            } else {
                None
            }
        })
    };
}

#[derive(Clone, Debug, Default, Builder)]
#[builder(setter(strip_option, into))]
pub struct AdaptiveOptions {
    pub minstep: Option<Minstep>,
    pub maxstep: Option<Maxstep>,
    pub initstep: Initstep,
    pub points: Points,
    pub reltol: Reltol,
    pub abstol: Abstol,
    pub norm: Norm,
    pub step_timeout: StepTimeout,
}

impl AdaptiveOptions {
    /// convenience method to create a new builder
    #[inline]
    pub fn builder() -> AdaptiveOptionsBuilder {
        AdaptiveOptionsBuilder::default()
    }
}

impl From<OdeOptionMap> for AdaptiveOptions {
    fn from(mut ops: OdeOptionMap) -> Self {
        Self {
            minstep: option_val!(ops rm Minstep),
            maxstep: option_val!(ops rm Maxstep),
            initstep: option_val!(ops rm Initstep).unwrap_or_default(),
            points: option_val!(ops rm Points).unwrap_or_default(),
            reltol: option_val!(ops rm Reltol).unwrap_or_default(),
            abstol: option_val!(ops rm Abstol).unwrap_or_default(),
            norm: option_val!(ops rm Norm).unwrap_or_default(),
            step_timeout: option_val!(ops rm StepTimeout).unwrap_or_default(),
        }
    }
}

impl From<&OdeOptionMap> for AdaptiveOptions {
    fn from(ops: &OdeOptionMap) -> Self {
        Self {
            minstep: option_val!(ops get Minstep),
            maxstep: option_val!(ops get Maxstep),
            initstep: option_val!(ops get Initstep).unwrap_or_default(),
            points: option_val!(ops get Points).unwrap_or_default(),
            reltol: option_val!(ops get Reltol).unwrap_or_default(),
            abstol: option_val!(ops get Abstol).unwrap_or_default(),
            norm: option_val!(ops get Norm).unwrap_or_default(),
            step_timeout: option_val!(ops get StepTimeout).unwrap_or_default(),
        }
    }
}

// constants for each option name
//or struct?
//pub struct OdeName;
// impls for struct like Headername http https://docs.rs/http/0.1.18/src/http/header/name.rs.html#32-34

pub trait OdeOp {
    fn option_name() -> &'static str;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Points {
    /// output is given for each value in `tspan`,
    /// as well as for each intermediate point the solver used
    All,
    /// output is given only for the supplied time stamps,
    /// without additional calculated time stamps
    Specified,
}

impl OdeOp for Points {
    fn option_name() -> &'static str {
        static NAME: &str = "Points";
        NAME
    }
}

impl Default for Points {
    fn default() -> Self {
        Points::All
    }
}

impl Into<OdeOption> for Points {
    fn into(self) -> OdeOption {
        OdeOption::Points(self)
    }
}

impl fmt::Display for Points {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Points: ")?;
        match self {
            Points::All => write!(f, "All"),
            Points::Specified => write!(f, "Specified"),
        }
    }
}

/// either single or mult value
macro_rules! options {
    ($($(#[$a:meta])*($id:ident, $n:expr) => $value:tt),*) => {

        $(
            option! {
                $(#[$a])*
                ($id, $n) => $value
            }
        )*

        /// All available Ode options
        #[derive(Debug, Clone, PartialEq)]
        pub enum OdeOption {
            Points(Points),
            $(
                $id($id),
            )*
        }
    };
}

macro_rules! option {
    // Single value option
    ($(#[$a:meta])*($id:ident, $n:expr) => [$value:ty]) => {
        $(#[$a])*
        #[derive(Clone, Debug, PartialEq)]
        pub struct $id(pub $value);
        __ode__deref!($id => $value);
        impl $crate::ode::options::OdeOp for $id {
            #[inline]
            fn option_name() -> &'static str {
                static NAME: &'static str = $n;
                NAME
            }
        }
        impl ::std::fmt::Display for $id {
            #[inline]
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                ::std::fmt::Display::fmt(&self.0, f)
            }
        }
        impl From<$value> for $id {

            fn from(item: $value) -> Self {
                $id(item)
            }
        }

        impl Into<OdeOption> for $id {

            fn into(self) -> OdeOption {
                OdeOption::$id(self)
            }
        }

    };

    // List option, multiple items
    ($(#[$a:meta])*($id:ident, $n:expr) => ($item:ty)) => {
        $(#[$a])*
        #[derive(Clone, Debug, PartialEq)]
        pub struct $id(pub Vec<$item>);
        __ode__deref!($id => Vec<$item>);

        impl ::std::fmt::Display for $id {
            #[inline]
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                fmt_comma_delimited(f, &self.0)
            }
        }

        impl Into<OdeOption> for $id {

            fn into(self) -> OdeOption {
                OdeOption::$id(self)
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __ode__deref {
    ($from:ty => $to:ty) => {
        impl ::std::ops::Deref for $from {
            type Target = $to;

            #[inline]
            fn deref(&self) -> &$to {
                &self.0
            }
        }

        impl ::std::ops::DerefMut for $from {
            #[inline]
            fn deref_mut(&mut self) -> &mut $to {
                &mut self.0
            }
        }
    };
}

options! {
    /// an integration step is accepted if E <= reltol*abs(y)
    (Reltol, "Reltol") => [f64],
    /// an integration step is accepted if E <= abstol
    (Abstol, "Abstol") => [f64],
    /// minimal integration step
    (Minstep, "Minstep") => [f64],
    /// maximal integration step
    (Maxstep, "Maxstep") => [f64],
    /// initial integration step
    #[derive(Default)]
    (Initstep, "Initstep") => [f64],
    /// Sometimes an integration step takes you out of the region where F(t,y) has a valid solution
    /// and F might result in an error.
    /// retries sets a limit to the number of times the solver might try with a smaller step.
    #[derive(Default)]
    (Retries, "Retries") => [usize],
    /// user defined norm for determining the error
    #[derive(Default)]
    (Norm, "Norm") => [PNorm],
    /// user defined timeout after which step reduction should not
    /// increase step for timeout controlled steps
    (StepTimeout, "StepTimeout") => [usize]
}

impl Default for Reltol {
    fn default() -> Self {
        Reltol(1e-5)
    }
}

impl Default for Abstol {
    fn default() -> Self {
        Abstol(1e-8)
    }
}

impl Default for StepTimeout {
    fn default() -> Self {
        StepTimeout(5)
    }
}

/// formats a list type separated by commas
#[inline]
fn fmt_comma_delimited<T: fmt::Display>(f: &mut ::std::fmt::Formatter, parts: &[T]) -> fmt::Result {
    let mut iter = parts.iter();
    if let Some(part) = iter.next() {
        ::std::fmt::Display::fmt(part, f)?;
    }
    for part in iter {
        f.write_str(", ")?;
        ::std::fmt::Display::fmt(part, f)?;
    }
    Ok(())
}
