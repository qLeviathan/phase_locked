//! CORDIC Rotation Operations
//!
//! Specialized rotation functions for efficient geometric transformations

use fixed::types::I32F32;
use serde::{Deserialize, Serialize};

use super::{Cordic, CORDIC_GAIN_INV};

/// 2D point in fixed-point coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Point2D {
    pub x: I32F32,
    pub y: I32F32,
}

impl Point2D {
    pub fn new(x: I32F32, y: I32F32) -> Self {
        Self { x, y }
    }

    pub fn from_floats(x: f64, y: f64) -> Self {
        Self {
            x: I32F32::from_num(x),
            y: I32F32::from_num(y),
        }
    }

    pub fn to_floats(&self) -> (f64, f64) {
        (self.x.to_num::<f64>(), self.y.to_num::<f64>())
    }
}

/// Rotate point by angle using CORDIC
pub fn rotate_point(cordic: &Cordic, point: Point2D, angle: I32F32) -> Point2D {
    let (sin, cos) = cordic.sin_cos(angle);

    // Rotation matrix:
    // [cos -sin] [x]
    // [sin  cos] [y]

    let x_new = cos * point.x - sin * point.y;
    let y_new = sin * point.x + cos * point.y;

    Point2D::new(x_new, y_new)
}

/// Rotate multiple points efficiently (batch operation)
pub fn rotate_points_batch(cordic: &Cordic, points: &[Point2D], angle: I32F32) -> Vec<Point2D> {
    // Compute sin/cos once for all points
    let (sin, cos) = cordic.sin_cos(angle);

    points
        .iter()
        .map(|p| {
            let x_new = cos * p.x - sin * p.y;
            let y_new = sin * p.x + cos * p.y;
            Point2D::new(x_new, y_new)
        })
        .collect()
}

/// Complex number multiplication using CORDIC
/// (a + bi) × (c + di) = (ac - bd) + (ad + bc)i
pub fn complex_multiply(
    cordic: &Cordic,
    a: Point2D,
    b: Point2D,
) -> Point2D {
    // Get magnitude and phase of each complex number
    let (mag_a, phase_a) = cordic.magnitude_phase(a.x, a.y);
    let (mag_b, phase_b) = cordic.magnitude_phase(b.x, b.y);

    // Multiply magnitudes and add phases
    let mag_result = mag_a * mag_b;
    let phase_result = phase_a + phase_b;

    // Convert back to Cartesian
    let (sin, cos) = cordic.sin_cos(phase_result);

    Point2D::new(mag_result * cos, mag_result * sin)
}

/// Compute rotation to align vector with x-axis
/// Returns angle needed to rotate (x, y) to (|v|, 0)
pub fn align_to_x_axis(cordic: &Cordic, point: Point2D) -> I32F32 {
    cordic.atan2(point.y, point.x)
}

/// Rotate coordinate system (change basis)
pub fn change_basis(cordic: &Cordic, point: Point2D, basis_angle: I32F32) -> Point2D {
    // Rotate by negative angle to change basis
    rotate_point(cordic, point, -basis_angle)
}

/// Interpolate between two angles (shortest path on circle)
pub fn angle_interpolate(cordic: &Cordic, angle_a: I32F32, angle_b: I32F32, t: I32F32) -> I32F32 {
    // Normalize angles
    let a = super::fixed_point::normalize_angle(angle_a);
    let b = super::fixed_point::normalize_angle(angle_b);

    // Compute difference
    let mut diff = b - a;

    // Take shortest path
    let pi = I32F32::from_num(std::f64::consts::PI);
    if diff > pi {
        diff -= I32F32::from_num(2.0 * std::f64::consts::PI);
    } else if diff < -pi {
        diff += I32F32::from_num(2.0 * std::f64::consts::PI);
    }

    // Interpolate
    a + diff * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotate_point_90_degrees() {
        let cordic = Cordic::default();
        let point = Point2D::from_floats(1.0, 0.0);
        let angle = I32F32::from_num(std::f64::consts::FRAC_PI_2); // 90 degrees

        let rotated = rotate_point(&cordic, point, angle);
        let (x, y) = rotated.to_floats();

        // After 90° rotation, (1,0) -> (0,1)
        assert!(x.abs() < 0.01);
        assert!((y - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rotate_points_batch() {
        let cordic = Cordic::default();
        let points = vec![
            Point2D::from_floats(1.0, 0.0),
            Point2D::from_floats(0.0, 1.0),
            Point2D::from_floats(-1.0, 0.0),
        ];

        let angle = I32F32::from_num(std::f64::consts::FRAC_PI_4); // 45 degrees
        let rotated = rotate_points_batch(&cordic, &points, angle);

        assert_eq!(rotated.len(), 3);
    }

    #[test]
    fn test_complex_multiply() {
        let cordic = Cordic::default();

        // (1 + i) × (1 + i) = 2i
        let a = Point2D::from_floats(1.0, 1.0);
        let b = Point2D::from_floats(1.0, 1.0);

        let result = complex_multiply(&cordic, a, b);
        let (x, y) = result.to_floats();

        // Result should be approximately (0, 2)
        assert!(x.abs() < 0.1);
        assert!((y - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_align_to_x_axis() {
        let cordic = Cordic::default();

        // Point at 45 degrees
        let point = Point2D::from_floats(1.0, 1.0);
        let angle = align_to_x_axis(&cordic, point);

        // Should be approximately π/4
        assert!((angle.to_num::<f64>() - std::f64::consts::FRAC_PI_4).abs() < 0.01);
    }

    #[test]
    fn test_angle_interpolate() {
        let cordic = Cordic::default();

        let a = I32F32::from_num(0.0);
        let b = I32F32::from_num(std::f64::consts::FRAC_PI_2);
        let t = I32F32::from_num(0.5);

        let result = angle_interpolate(&cordic, a, b, t);

        // Should be π/4 (halfway between 0 and π/2)
        assert!((result.to_num::<f64>() - std::f64::consts::FRAC_PI_4).abs() < 0.01);
    }
}
