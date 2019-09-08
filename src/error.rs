use snafu::{ensure, Backtrace, ErrorCompat, ResultExt, Snafu};
use std::str::FromStr;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Element not initialized {}", msg))]
    Uninitialized { msg: String },
    #[snafu(display("The user id {} is invalid", user_id))]
    UserIdInvalid { user_id: i32, backtrace: Backtrace },
}

impl Error {
    pub fn uninitialized<T: Into<String>>(msg: T) -> Self {
        Error::Uninitialized { msg: msg.into() }
    }
}
