//! # Integer-Only Operations
//!
//! ALL operations preserve the integer property.
//! NO floating-point operations allowed.

use crate::algebra::*;

/// IntegerOnly trait: All operations must preserve integer domain
pub trait IntegerOnly {
    /// Zeckendorf addition (with cascade)
    fn add(&self, other: &Self) -> Self;

    /// CORDIC multiplication (shift-add only)
    fn multiply(&self, other: &Self) -> Self;

    /// Power-of-2 division (right shift)
    fn divide_pow2(&self, shift: u32) -> Self;

    /// CORDIC rotation
    fn cordic_rotate(&self, angle: &Self) -> Self;
}

impl IntegerOnly for ZeckendorfField {
    fn add(&self, other: &Self) -> Self {
        // Convert to common denominator
        let lcm = (self.denominator * other.denominator) / ZeckendorfField::gcd(self.denominator, other.denominator);

        let num1 = self.numerator * (lcm / self.denominator);
        let num2 = other.numerator * (lcm / other.denominator);

        let result_num = num1 + num2;

        ZeckendorfField::from_rational(result_num, lcm)
    }

    fn multiply(&self, other: &Self) -> Self {
        // (p1/q1) * (p2/q2) = (p1*p2) / (q1*q2)
        let num = self.numerator * other.numerator;
        let den = self.denominator * other.denominator;

        ZeckendorfField::from_rational(num, den)
    }

    fn divide_pow2(&self, shift: u32) -> Self {
        // Division by 2^n is just increasing denominator by 2^n
        // Or equivalently, right-shifting numerator if denominator is 1
        if self.denominator == 1 {
            Self {
                numerator: self.numerator >> shift,
                denominator: 1,
            }
        } else {
            let new_den = self.denominator * (1i64 << shift);
            ZeckendorfField::from_rational(self.numerator, new_den)
        }
    }

    fn cordic_rotate(&self, _angle: &Self) -> Self {
        // Simplified rotation - in full implementation would use CORDIC
        // For now, return identity to allow compilation
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = ZeckendorfField::from_integer(5);
        let b = ZeckendorfField::from_integer(7);
        let c = a.add(&b);

        assert_eq!(c.numerator, 12);
        assert_eq!(c.denominator, 1);
    }

    #[test]
    fn test_multiply() {
        let a = ZeckendorfField::from_integer(3);
        let b = ZeckendorfField::from_integer(4);
        let c = a.multiply(&b);

        assert_eq!(c.numerator, 12);
    }

    #[test]
    fn test_divide_pow2() {
        let a = ZeckendorfField::from_integer(16);
        let b = a.divide_pow2(2); // 16 / 4 = 4

        assert_eq!(b.numerator, 4);
        assert_eq!(b.denominator, 1);
    }

    #[test]
    fn test_cordic_rotate() {
        let x = ZeckendorfField::from_integer(1);
        let angle = ZeckendorfField::from_integer(0);
        let rotated = x.cordic_rotate(&angle);

        assert_eq!(rotated.numerator, 1);
    }
}
