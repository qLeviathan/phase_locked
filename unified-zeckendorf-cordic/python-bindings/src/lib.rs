//! Python bindings for unified-zeckendorf-cordic

use pyo3::prelude::*;
use unified_zeckendorf_cordic::{
    BitLattice, IntegerOnly, LucasRing, ZeckTokenizer, ZeckendorfField, UniversalTokenizer,
};

#[pyclass]
#[derive(Clone)]
struct PyZeckendorfField {
    inner: ZeckendorfField,
}

#[pymethods]
impl PyZeckendorfField {
    #[new]
    fn new(numerator: i64, denominator: Option<i64>) -> Self {
        let inner = if let Some(d) = denominator {
            ZeckendorfField::from_rational(numerator, d)
        } else {
            ZeckendorfField::from_integer(numerator)
        };
        Self { inner }
    }

    fn add(&self, other: &PyZeckendorfField) -> PyZeckendorfField {
        PyZeckendorfField {
            inner: self.inner.add(&other.inner),
        }
    }

    fn multiply(&self, other: &PyZeckendorfField) -> PyZeckendorfField {
        PyZeckendorfField {
            inner: self.inner.multiply(&other.inner),
        }
    }

    fn divide_pow2(&self, shift: u32) -> PyZeckendorfField {
        PyZeckendorfField {
            inner: self.inner.divide_pow2(shift),
        }
    }

    fn cordic_rotate(&self, angle: &PyZeckendorfField) -> PyZeckendorfField {
        PyZeckendorfField {
            inner: self.inner.cordic_rotate(&angle.inner),
        }
    }

    fn to_float(&self) -> f64 {
        self.inner.to_f64_display()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
}

#[pyclass]
#[derive(Clone)]
struct PyBitLattice {
    inner: BitLattice,
}

#[pymethods]
impl PyBitLattice {
    #[new]
    fn new(value: Option<u64>) -> Self {
        let inner = if let Some(v) = value {
            BitLattice::from_integer(v)
        } else {
            BitLattice::new()
        };
        Self { inner }
    }

    fn to_integer(&self) -> u64 {
        self.inner.to_integer()
    }

    fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    fn cascade(&mut self) {
        self.inner.cascade();
    }

    fn holes(&self) -> usize {
        self.inner.holes()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
}

#[pyclass]
struct PyZeckTokenizer {
    inner: ZeckTokenizer,
}

#[pymethods]
impl PyZeckTokenizer {
    #[new]
    fn new(vocab_size: Option<usize>) -> Self {
        let inner = if let Some(vs) = vocab_size {
            ZeckTokenizer { vocab_size: vs }
        } else {
            ZeckTokenizer::default()
        };
        Self { inner }
    }

    fn tokenize_numeric(&self, n: i64) -> PyBitLattice {
        PyBitLattice {
            inner: self.inner.tokenize_numeric(n),
        }
    }

    fn tokenize_text(&self, s: &str) -> Vec<PyBitLattice> {
        self.inner
            .tokenize_text(s)
            .into_iter()
            .map(|inner| PyBitLattice { inner })
            .collect()
    }

    fn tokenize_image(&self, pixels: Vec<u8>) -> Vec<PyBitLattice> {
        self.inner
            .tokenize_image(&pixels)
            .into_iter()
            .map(|inner| PyBitLattice { inner })
            .collect()
    }
}

/// Python module
#[pymodule]
fn zeckendorf_cordic(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyZeckendorfField>()?;
    m.add_class::<PyBitLattice>()?;
    m.add_class::<PyZeckTokenizer>()?;

    m.add_function(wrap_pyfunction!(py_fibonacci, m)?)?;
    m.add_function(wrap_pyfunction!(py_lucas, m)?)?;
    m.add_function(wrap_pyfunction!(py_zeckendorf_decomposition, m)?)?;
    m.add_function(wrap_pyfunction!(py_verify_axioms, m)?)?;

    Ok(())
}

#[pyfunction]
fn py_fibonacci(n: usize) -> u64 {
    unified_zeckendorf_cordic::sequences::fibonacci(n)
}

#[pyfunction]
fn py_lucas(n: usize) -> u64 {
    unified_zeckendorf_cordic::sequences::lucas(n)
}

#[pyfunction]
fn py_zeckendorf_decomposition(n: u64) -> Vec<u64> {
    unified_zeckendorf_cordic::sequences::zeckendorf_decomposition(n)
}

#[pyfunction]
fn py_verify_axioms() -> PyResult<bool> {
    match unified_zeckendorf_cordic::verify_axioms() {
        Ok(_) => Ok(true),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
    }
}
