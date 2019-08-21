use std::collections::HashMap;
use std::fmt;

// http://docs.juliadiffeq.org/latest/basics/common_solver_opts.html
// http://docs.juliadiffeq.org/latest/basics/compatibility_chart.html#Solver-Compatibility-Chart-1

// matlab: https://www.mathworks.com/help/matlab/math/summary-of-ode-options.html

pub struct OdeOptionMap(pub HashMap<&'static str, OdeOption>);
// for each solver a subset of the `OdeOption`
// Into / From impls to convert

// constants for each option name
//or struct?
//pub struct OdeName;
// impls for struct like Headername http https://docs.rs/http/0.1.18/src/http/header/name.rs.html#32-34

pub trait OdeOp {
    fn option_name() -> &'static str;
}

#[derive(Debug, Clone, PartialEq)]
pub enum Points {
    /// output is given for each value in `tspan`,
    /// as well as for each intermediate point the solver used
    All,
    /// output is given only for each value in `tspan`.
    Specified(Vec<usize>),
}

impl Default for Points {
    fn default() -> Self {
        Points::All
    }
}

impl fmt::Display for Points {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Points: ")?;
        match self {
            Points::All => write!(f, "All"),
            Points::Specified(idx) => {
                write!(f, "[")?;
                fmt_comma_delimited(f, idx)?;
                write!(f, "]")
            }
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
        #[derive(Debug, Clone)]
        pub enum OdeOption {
            $(
                $id(opt_val!($value)),
            )*
        }
    };
}

macro_rules! opt_val {
    ([$value:ty]) => {$value};
    (($item:ty)) => {Vec<$item>};
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
    };
}

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

macro_rules! impl_ode_ops {
    ( $(#[$a:meta])* @common $id:ident {
        $($(#[$fa:meta])* @$name:ident $f:ident: $ty:ty),*
    }) => {
        $(#[$a])*
        #[derive(Debug, Clone, Builder)]
        pub struct $id {
            #[builder(setter(strip_option), default)]
            pub reltol : Option<f64>,
            pub abstol : Option<f64>,
            pub minstep : Option<usize>,
            pub maxstep : Option<usize>,
            pub initstep : Option<usize>,
            $(
               $(#[$fa])*
               #[builder(setter(into))]
               pub $f : $ty,
            )*
        }

        impl From<crate::ode::options::OdeOptionMap> for $id {

            fn from(mut ops: OdeOptionMap) -> Self {

//                Self {
//                 reltol : ops.0.remove(crate::ode::options::Reltol::option_name()),
//                 abstol : ops.0.remove(crate::ode::options::Abstol::option_name()),
//                 minstep : ops.0.remove(crate::ode::options::Minstep::option_name()),
//                 maxstep : ops.0.remove(crate::ode::options::Maxstep::option_name()),
//                 initstep : ops.0.remove(crate::ode::options::Initstep::option_name()),
//                $(
//                    $f : ops.0.remove($name::option_name()),
//                )*
//                }
                unimplemented!()
            }
        }

        impl Into<crate::ode::options::OdeOptionMap> for $id {

            fn into(self) -> crate::ode::options::OdeOptionMap {
//             reltol : ops.0.remove(crate::ode::options::Reltol::option_name())
//                 .map(crate::ode::options::OdeOption::Reltol),
//                 abstol : ops.0.remove(crate::ode::options::Abstol::option_name())
//                  .map(crate::ode::options::OdeOption::Abstol),
//                 minstep : ops.0.remove(crate::ode::options::Minstep::option_name())
//                  .map(crate::ode::options::OdeOption::Minstep),
//                 maxstep : ops.0.remove(crate::ode::options::Maxstep::option_name())
//                  .map(crate::ode::options::OdeOption::Maxstep),
//                 initstep : ops.0.remove(crate::ode::options::Initstep::option_name())
//                  .map(crate::ode::options::OdeOption::Initstep),
                unimplemented!()
            }

        }
    };
}

options! {
    /// single
    (Reltol, "Reltol") => [f64],
    (Abstol, "Abstol") => [f64],
    (Minstep, "Minstep") => [usize],
    (Maxstep, "Maxstep") => [usize],
    (Initstep, "Initstep") => [usize],
    (OutputPoints, "Points") => [Points],

    /// multi
    (DummyMult, "Points") => (String)
}

impl_ode_ops!(
    /// docs
   @common Demo {
   /// docs
   @Reltol dummy : Option<f64> }
);
