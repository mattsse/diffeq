use std::collections::HashMap;
use std::fmt;

// http://docs.juliadiffeq.org/latest/basics/common_solver_opts.html
// http://docs.juliadiffeq.org/latest/basics/compatibility_chart.html#Solver-Compatibility-Chart-1

// matlab: https://www.mathworks.com/help/matlab/math/summary-of-ode-options.html

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

#[derive(Debug, Clone, PartialEq)]
pub enum Points {
    /// output is given for each value in `tspan`,
    /// as well as for each intermediate point the solver used
    All,
    /// output is given only for each value in `tspan`.
    /// where the inner vector contains the indexes of the requested `tspan` values
    Specified(Vec<usize>),
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
        #[derive(Debug, Clone, PartialEq)]
        pub enum OdeOption {
            Points(Points),
            $(
                $id($id),
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
        impl From<$value> for $id {

            fn from(item: $value) -> Self {
                $id(item)
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

macro_rules! get_opt {
    ($ops:expr => $s:ident {$($name:ident : $id:ident,)*}) => {
       $s {
           $(
                $name : $ops.0.remove($id::option_name()).map(|op|{
                    if let crate::ode::options::OdeOption::$name(val) = op {
                        Some(val.0)
                    } else {
                        None
                    }
                }),
           )*
        }
    };
}

macro_rules! impl_ode_ops {
    ( $(#[$a:meta])* @common $id:ident {
        $($(#[$fa:meta])* $fname:ident: $fty:ident),*
    }) => {

        $(#[$a])*
        #[derive(Debug, Clone, Builder)]
        pub struct $id {
            #[builder(setter(strip_option), default)]
            pub reltol : Option<Reltol>,
            pub abstol : Option<Abstol>,
            pub minstep : Option<Minstep>,
            pub maxstep : Option<Maxstep>,
            pub initstep : Option<Initstep>,
            $(
               $(#[$fa])*
               #[builder(setter(into))]
               pub $fname : $fty,
            )*
        }

        impl From<crate::ode::options::OdeOptionMap> for $id {

            fn from(mut ops: OdeOptionMap) -> Self {
                unimplemented!()
//                get_opt!{
//                    ops => $id {
//                        reltol : Reltol,
//                        abstol : Abstol,
//                        minstep : Minstep,
//                        maxstep : Maxstep,
//                        initstep : Initstep,
//                         $(
//                            $f : $name,
//                         )*
//                    }
//                }
            }
        }

        impl Into<crate::ode::options::OdeOptionMap> for $id {

            fn into(self) -> crate::ode::options::OdeOptionMap {
                unimplemented!()
            }

        }
    };
}

options! {
    /// user defined norm for determining the error
    (Norm, "Norm") => [f64],
    /// an integration step is accepted if E <= reltol*abs(y)
    (Reltol, "Reltol") => [f64],
    /// an integration step is accepted if E <= abstol
    (Abstol, "Abstol") => [f64],
    /// minimal integration step
    (Minstep, "Minstep") => [f64],
    /// maximal integration step
    (Maxstep, "Maxstep") => [f64],
    /// initial integration step
    (Initstep, "Initstep") => [f64],
    /// Sometimes an integration step takes you out of the region where F(t,y) has a valid solution
    /// and F might result in an error.
    /// retries sets a limit to the number of times the solver might try with a smaller step.
    (Retries, "Retries") => [usize]

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

impl Default for Initstep {
    fn default() -> Self {
        Initstep(0.)
    }
}

impl Default for Retries {
    fn default() -> Self {
        Retries(0)
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
