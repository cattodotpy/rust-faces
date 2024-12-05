use ort::Error as OrtError;

use crate::detection::RustFacesError;

impl From<OrtError> for RustFacesError {
    fn from(err: OrtError) -> Self {
        RustFacesError::InferenceError(err.to_string())
    }
}
