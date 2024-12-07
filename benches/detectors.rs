use criterion::{criterion_group, criterion_main, Criterion};
use rust_faces::{
    BlazeFaceParams, FaceDetection, FaceDetectorBuilder, InferParams, MtCnnParams, Provider,
};

fn criterion_benchmark(c: &mut Criterion) {
    let image = image::open("tests/data/images/faces.jpg")
        .expect("Can't open test image.")
        .into_rgb8();

    for (name, detection, provider) in vec![
        (
            "blazeface640_cpu",
            FaceDetection::BlazeFace640(BlazeFaceParams::default()),
            Provider::OrtCpu,
        ),
        (
            "blazeface640_gpu",
            FaceDetection::BlazeFace640(BlazeFaceParams::default()),
            Provider::OrtCuda(0),
        ),
        (
            "blazeface320_cpu",
            FaceDetection::BlazeFace320(BlazeFaceParams::default()),
            Provider::OrtCpu,
        ),
        (
            "blazeface320_gpu",
            FaceDetection::BlazeFace320(BlazeFaceParams::default()),
            Provider::OrtCuda(0),
        ),
        (
            "mtcnn_cpu",
            FaceDetection::MtCnn(MtCnnParams::default()),
            Provider::OrtCpu,
        ),
        (
            "mtcnn_gpu",
            FaceDetection::MtCnn(MtCnnParams::default()),
            Provider::OrtCuda(0),
        ),
    ] {
        let face_detector = FaceDetectorBuilder::new(detection)
            .download()
            .infer_params(InferParams {
                provider,
                ..Default::default()
            })
            .build()
            .expect("Fail to load the face detector.");
        c.bench_function(name, |b| {
            b.iter(|| face_detector.detect(image.clone()).unwrap())
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
