use rust_faces::{BlazeFaceParams, FaceDetection, FaceDetectorBuilder, InferParams, Provider};

#[test]
pub fn main() {
    let face_detector =
        FaceDetectorBuilder::new(FaceDetection::BlazeFace640(BlazeFaceParams::default()))
            .download()
            .infer_params(InferParams {
                provider: Provider::OrtCpu,
                intra_threads: Some(5),
                ..Default::default()
            })
            .build()
            .expect("Fail to load the face detector.");

    let image = image::open("tests/data/images/faces.jpg")
        .expect("Can't open test image.")
        .into_rgb8();
    let _ = face_detector.detect(image.clone()).unwrap();

    std::fs::create_dir_all("tests/output").expect("Can't create test output dir.");
    image
        .save("tests/output/test_design.jpg")
        .expect("Can't save test image.");
}
