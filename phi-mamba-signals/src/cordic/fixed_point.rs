//! Fixed-Point Arithmetic Utilities
//!
//! Helper functions for I32F32 fixed-point operations
//! Used throughout CORDIC computations for hardware efficiency

use fixed::types::I32F32;

/// Convert degrees to radians (fixed-point)
pub fn deg_to_rad(degrees: f64) -> I32F32 {
    I32F32::from_num(degrees * std::f64::consts::PI / 180.0)
}

/// Convert radians to degrees (fixed-point)
pub fn rad_to_deg(radians: I32F32) -> f64 {
    radians.to_num::<f64>() * 180.0 / std::f64::consts::PI
}

/// Normalize angle to [-π, π]
pub fn normalize_angle(mut angle: I32F32) -> I32F32 {
    let two_pi = I32F32::from_num(2.0 * std::f64::consts::PI);
    let pi = I32F32::from_num(std::f64::consts::PI);

    while angle > pi {
        angle -= two_pi;
    }
    while angle < -pi {
        angle += two_pi;
    }

    angle
}

/// Clamp value to range [min, max]
pub fn clamp(value: I32F32, min: I32F32, max: I32F32) -> I32F32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Linear interpolation
pub fn lerp(a: I32F32, b: I32F32, t: I32F32) -> I32F32 {
    a + (b - a) * t
}

/// Compute absolute value
pub fn abs(value: I32F32) -> I32F32 {
    if value < I32F32::from_num(0) {
        -value
    } else {
        value
    }
}

/// Sign function (-1, 0, or 1)
pub fn sign(value: I32F32) -> i8 {
    if value > I32F32::from_num(0) {
        1
    } else if value < I32F32::from_num(0) {
        -1
    } else {
        0
    }
}

/// Fast inverse square root approximation
/// Uses Quake III algorithm adapted for fixed-point
pub fn inv_sqrt_approx(x: I32F32) -> I32F32 {
    if x <= I32F32::from_num(0) {
        return I32F32::from_num(0);
    }

    // Convert to f32 for magic constant trick
    let x_f32 = x.to_num::<f32>();

    // Quake III fast inverse square root
    let i = x_f32.to_bits();
    let i = 0x5f3759df - (i >> 1);
    let y = f32::from_bits(i);

    // One Newton-Raphson iteration for accuracy
    let y = y * (1.5 - 0.5 * x_f32 * y * y);

    I32F32::from_num(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deg_to_rad() {
        let rad = deg_to_rad(180.0);
        assert!((rad.to_num::<f64>() - std::f64::consts::PI).abs() < 0.001);

        let rad = deg_to_rad(90.0);
        assert!((rad.to_num::<f64>() - std::f64::consts::FRAC_PI_2).abs() < 0.001);
    }

    #[test]
    fn test_normalize_angle() {
        let angle = I32F32::from_num(7.0); // > 2π
        let normalized = normalize_angle(angle);
        let pi = I32F32::from_num(std::f64::consts::PI);

        assert!(normalized >= -pi && normalized <= pi);
    }

    #[test]
    fn test_clamp() {
        let value = I32F32::from_num(5.0);
        let min = I32F32::from_num(0.0);
        let max = I32F32::from_num(3.0);

        let clamped = clamp(value, min, max);
        assert_eq!(clamped, max);
    }

    #[test]
    fn test_lerp() {
        let a = I32F32::from_num(0.0);
        let b = I32F32::from_num(10.0);
        let t = I32F32::from_num(0.5);

        let result = lerp(a, b, t);
        assert!((result.to_num::<f64>() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_sign() {
        assert_eq!(sign(I32F32::from_num(5.0)), 1);
        assert_eq!(sign(I32F32::from_num(-3.0)), -1);
        assert_eq!(sign(I32F32::from_num(0.0)), 0);
    }
}
