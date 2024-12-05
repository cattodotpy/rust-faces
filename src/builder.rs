use ort::{
    execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider},
    session::Session,
};

use crate::{
    blazeface::BlazeFaceParams,
    detection::{FaceDetector, RustFacesResult},
    model_repository::{GitHubRepository, ModelRepository},
    BlazeFace, MtCnn, MtCnnParams,
};

pub enum FaceDetection {
    BlazeFace640(BlazeFaceParams),
    BlazeFace320(BlazeFaceParams),
    MtCnn(MtCnnParams),
}

#[derive(Clone, Debug)]
enum OpenMode {
    File(String),
    Download,
}

/// Runtime inference provider. Some may not be available depending of your Onnx runtime installation.
#[derive(Clone, Copy, Debug)]
pub enum Provider {
    /// Uses the, default, CPU inference
    OrtCpu,
    /// Uses the Cuda inference.
    OrtCuda(i32),
    /// Uses Intel's OpenVINO inference.
    OrtVino(i32),
    /// Apple's Core ML inference.
    OrtCoreMl,
}

/// Inference parameters.
pub struct InferParams {
    /// Chooses the ONNX runtime provider.
    pub provider: Provider,
    /// Sets the number of intra-op threads.
    pub intra_threads: Option<usize>,
    /// Sets the number of inter-op threads.
    pub inter_threads: Option<usize>,
}

impl Default for InferParams {
    /// Default provider is `OrtCpu` (Onnx CPU).
    fn default() -> Self {
        Self {
            provider: Provider::OrtCpu,
            intra_threads: None,
            inter_threads: None,
        }
    }
}

/// Builder for loading or downloading, configuring, and creating face detectors.
pub struct FaceDetectorBuilder {
    detector: FaceDetection,
    open_mode: OpenMode,
    infer_params: InferParams,
}

impl FaceDetectorBuilder {
    /// Create a new builder for the given face detector.
    ///
    /// # Arguments
    ///
    /// * `detector` - The face detector to build.
    pub fn new(detector: FaceDetection) -> Self {
        Self {
            detector,
            open_mode: OpenMode::Download,
            infer_params: InferParams::default(),
        }
    }

    /// Load the model from the given file path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the model file.
    pub fn from_file(mut self, path: String) -> Self {
        self.open_mode = OpenMode::File(path);
        self
    }

    /// Sets the model to be downloaded from the model repository.
    pub fn download(mut self) -> Self {
        self.open_mode = OpenMode::Download;
        self
    }

    /// Sets the inference parameters.
    pub fn infer_params(mut self, params: InferParams) -> Self {
        self.infer_params = params;
        self
    }

    /// Instantiates a new detector.
    ///
    /// # Errors
    ///
    /// Returns an error if the model can't be loaded.
    ///
    /// # Returns
    ///
    /// A new face detector.
    pub fn build(&self) -> RustFacesResult<Box<dyn FaceDetector>> {
        let mut ort_builder = Session::builder()
            .unwrap()
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .unwrap()
            .with_approximate_gelu()
            .unwrap()
            .with_intra_threads(self.infer_params.intra_threads.unwrap_or(1))
            .unwrap()
            .with_inter_threads(self.infer_params.inter_threads.unwrap_or(1))
            .unwrap();

        ort_builder = match self.infer_params.provider {
            Provider::OrtCuda(device_id) => {
                let provider = CUDAExecutionProvider::default().with_device_id(device_id);

                if !provider.is_available().unwrap() {
                    eprintln!("Warning: CUDA is not available. It'll likely use CPU inference.");
                }

                println!("Using CUDA inference with device id: {}", device_id);
                ort_builder
                    .with_execution_providers([provider.build()])
                    .unwrap()
            }
            Provider::OrtVino(_device_id) => {
                return Err(crate::RustFacesError::Other(
                    "OpenVINO is not supported yet.".to_string(),
                ));
            }
            Provider::OrtCoreMl => {
                let provider = CoreMLExecutionProvider::default();

                if !provider.is_available().unwrap() {
                    eprintln!("Warning: CoreML is not available. It'll likely use CPU inference.");
                }
                ort_builder
                    .with_execution_providers([provider.build()])
                    .unwrap()
            }
            _ => ort_builder,
        };

        let repository = GitHubRepository::new();

        let model_paths = match &self.open_mode {
            OpenMode::Download => repository
                .get_model(&self.detector)?
                .iter()
                .map(|path| path.to_str().unwrap().to_string())
                .collect(),
            OpenMode::File(path) => vec![path.clone()],
        };

        match &self.detector {
            FaceDetection::BlazeFace640(params) => Ok(Box::new(BlazeFace::from_session(
                ort_builder.commit_from_file(&model_paths[0]).unwrap(),
                params.clone(),
            ))),
            FaceDetection::BlazeFace320(params) => Ok(Box::new(BlazeFace::from_session(
                ort_builder.commit_from_file(&model_paths[0]).unwrap(),
                params.clone(),
            ))),
            FaceDetection::MtCnn(params) => Ok(Box::new(
                MtCnn::from_session(
                    ort_builder
                        .clone()
                        .commit_from_file(&model_paths[0])
                        .unwrap(),
                    ort_builder
                        .clone()
                        .commit_from_file(&model_paths[1])
                        .unwrap(),
                    ort_builder.commit_from_file(&model_paths[2]).unwrap(),
                    params.clone(),
                )
                .unwrap(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {}
