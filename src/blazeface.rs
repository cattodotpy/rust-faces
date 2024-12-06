use image::{
    imageops::{self, FilterType},
    GenericImageView, ImageBuffer, Pixel, Rgb,
};
use ndarray::{Array3, Axis};
use ort::{memory::Allocator, session::Session, value::Value};
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    detection::{FaceDetector, RustFacesResult},
    imaging::make_border,
    priorboxes::{PriorBoxes, PriorBoxesParams},
    Face, Nms,
};

pub type Image<P> = ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>;

fn resize_and_border<I: GenericImageView>(
    image: &I,
    output_size: (u32, u32),
    border_color: I::Pixel,
) -> (Image<I::Pixel>, f32)
where
    I::Pixel: 'static,
    <I::Pixel as Pixel>::Subpixel: 'static,
{
    let (input_width, input_height) = image.dimensions();
    let (output_width, output_height) = output_size;
    let ratio = (output_width as f32 / input_width as f32)
        .min(output_height as f32 / input_height as f32)
        .min(1.0); // avoid scaling up.

    let (resize_width, resize_height) = (
        (input_width as f32 * ratio).round() as i32,
        (input_height as f32 * ratio).round() as i32,
    );
    let resized = imageops::resize(
        image,
        resize_width as u32,
        resize_height as u32,
        FilterType::Nearest,
    );

    let (left, right, top, bottom) = {
        let (x_pad, y_pad) = (
            ((output_width as i32 - resize_width) % 16) as f32 / 2.0,
            ((output_height as i32 - resize_height) % 16) as f32 / 2.0,
        );
        (
            (x_pad - 0.1).round() as u32,
            (x_pad + 0.1).round() as u32,
            (y_pad - 0.1).round() as u32,
            (y_pad + 0.1).round() as u32,
        )
    };

    (
        make_border(&resized, top, bottom, left, right, border_color),
        ratio,
    )
}

#[derive(Debug, Clone)]
pub struct BlazeFaceParams {
    pub score_threshold: f32,
    pub nms: Nms,
    pub target_size: usize,
    pub prior_boxes: PriorBoxesParams,
}

impl Default for BlazeFaceParams {
    fn default() -> Self {
        Self {
            score_threshold: 0.95,
            nms: Nms::default(),
            target_size: 1280,
            prior_boxes: PriorBoxesParams::default(),
        }
    }
}

pub struct BlazeFace {
    session: Session,
    params: BlazeFaceParams,
}

impl BlazeFace {
    pub fn from_session(session: Session, params: BlazeFaceParams) -> Self {
        Self { session, params }
    }
}

impl FaceDetector for BlazeFace {
    fn detect(&self, image: ImageBuffer<Rgb<u8>, Vec<u8>>) -> RustFacesResult<Vec<Face>> {
        let (image, ratio) = resize_and_border(
            &image,
            (
                self.params.target_size as u32,
                self.params.target_size as u32,
            ),
            Rgb([104, 117, 123]),
        );
        let (input_width, input_height) = image.dimensions();
        let image = Array3::<f32>::from_shape_fn(
            (3, input_height as usize, input_width as usize),
            |(c, y, x)| {
                match c {
                    // https://github.com/zineos/blazeface/blob/main/tools/test.py seems to use OpenCV's BGR
                    0 => image.get_pixel(x as u32, y as u32)[2] as f32 - 104.0,
                    1 => image.get_pixel(x as u32, y as u32)[1] as f32 - 117.0,
                    2 => image.get_pixel(x as u32, y as u32)[0] as f32 - 123.0,
                    _ => unreachable!(),
                }
            },
        )
        .insert_axis(Axis(0));

        let mut binding = self.session.create_binding()?;
        let box_allocator = Allocator::default();
        let scores_allocator = Allocator::default();
        let landmark_allocator = Allocator::default();

        binding.bind_output_to_device("boxes", &box_allocator.memory_info())?;
        binding.bind_output_to_device("scores", &scores_allocator.memory_info())?;
        binding.bind_output_to_device("landmark", &landmark_allocator.memory_info())?;

        let input = Value::from_array(image)?;

        binding.bind_input("input", &input)?;

        let mut outputs = binding.run()?;

        // Boxes regressions: N box with the format [start x, start y, end x, end y].
        let binding = outputs.remove("boxes").unwrap();
        let boxes = binding.try_extract_tensor().unwrap();
        let binding = outputs.remove("scores").unwrap();
        let scores = binding.try_extract_tensor().unwrap();
        let binding = outputs.remove("landmark").unwrap();
        let landmarks = binding.try_extract_tensor().unwrap();

        let num_boxes = boxes.view().shape()[1];

        let priors = PriorBoxes::new(
            &self.params.prior_boxes,
            (input_width as usize, input_height as usize),
        );

        let scale_ratios = (input_width as f32 / ratio, input_height as f32 / ratio);

        let faces: Vec<_> = boxes
            .view()
            .to_shape((num_boxes, 4))
            .unwrap()
            .axis_iter(Axis(0))
            .zip(
                landmarks
                    .view()
                    .to_shape((num_boxes, 10))
                    .unwrap()
                    .axis_iter(Axis(0)),
            )
            .zip(priors.anchors.iter())
            .zip(
                scores
                    .view()
                    .to_shape((num_boxes, 2))
                    .unwrap()
                    .axis_iter(Axis(0)),
            )
            .par_bridge()
            .filter_map(|(((rect, landmarks), prior), score)| {
                let score = score[1];

                if score > self.params.score_threshold {
                    let rect = priors.decode_box(prior, &(rect[0], rect[1], rect[2], rect[3]));
                    let rect = rect.scale(scale_ratios.0, scale_ratios.1);

                    let landmarks = landmarks
                        .to_vec()
                        .chunks(2)
                        .map(|point| {
                            let point = priors.decode_landmark(prior, (point[0], point[1]));
                            (point.0 * scale_ratios.0, point.1 * scale_ratios.1)
                        })
                        .collect::<Vec<_>>();

                    Some(Face {
                        rect,
                        landmarks: Some(landmarks),
                        confidence: score,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(self.params.nms.suppress_non_maxima(faces))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        imaging::ToRgb8,
        model_repository::{GitHubRepository, ModelRepository},
        testing::{output_dir, sample_array_image, sample_image},
    };

    use super::*;
    use image::RgbImage;
    use rstest::rstest;

    use std::path::PathBuf;

    #[rstest]
    pub fn test_resize_and_border(sample_image: RgbImage, output_dir: PathBuf) {
        let (resized, _) = resize_and_border(&sample_image, (1280, 1280), Rgb([0, 255, 0]));

        resized.save(output_dir.join("test_resized.jpg")).unwrap();
        assert!(resized.width() == 896);
        assert!(resized.height() == 1280);
    }

    #[cfg(feature = "viz")]
    fn should_detect_impl(
        blaze_model: crate::FaceDetection,
        sample_array_image: Array3<u8>,
        output_dir: PathBuf,
    ) {
        use crate::viz;
        let environment = Arc::new(
            ort::Environment::builder()
                .with_name("BlazeFace")
                .build()
                .unwrap(),
        );

        let params = match &blaze_model {
            crate::FaceDetection::BlazeFace640(params) => params.clone(),
            crate::FaceDetection::BlazeFace320(params) => params.clone(),
            _ => unreachable!(),
        };

        let drive = GitHubRepository::new();
        let model_path = drive.get_model(&blaze_model).expect("Can't download model")[0].clone();

        let face_detector = BlazeFace::from_file(model_path.to_str().unwrap(), params);
        let mut canvas = sample_array_image.to_rgb8();
        let faces = face_detector
            .detect(sample_array_image.into_dyn().view())
            .unwrap();

        viz::draw_faces(&mut canvas, faces);

        canvas
            .save(output_dir.join("blazefaces.png"))
            .expect("Can't save image");
    }

    #[rstest]
    #[cfg(feature = "viz")]
    fn should_detect_640(sample_array_image: Array3<u8>, output_dir: PathBuf) {
        should_detect_impl(
            crate::FaceDetection::BlazeFace640(BlazeFaceParams::default()),
            sample_array_image,
            output_dir,
        );
    }

    #[rstest]
    #[cfg(feature = "viz")]
    fn should_detect_320(sample_array_image: Array3<u8>, output_dir: PathBuf) {
        should_detect_impl(
            crate::FaceDetection::BlazeFace320(BlazeFaceParams::default()),
            sample_array_image,
            output_dir,
        );
    }
}
