use crate::pattern::meshgrid;
use ndarray::{Array1, Array2, Zip};
use std::f64::consts::PI;

/// Final color mapping function
/// Kind of represents a linear function mapping (0,1) to (0,255)
///
#[inline(always)]
pub fn f_vectorized(x: &Array2<f64>) -> Array2<u8> {
    x.mapv(|x| {
        ((255.0 * (-(-1000.0 * x).exp()).exp() * x.abs().powf((-(1000.0 * (x - 1.0)).exp()).exp()))
            .floor()
            % 256.0) as u8
    })
}

pub fn f(x: f64) -> u8 {
    (255.0 * (-(-1000.0 * x).exp()).exp() * x.abs().powf((-(1000.0 * (x - 1.0)).exp()).exp()))
        .floor()
        .clamp(0.0, 255.0) as u8
}

// Cached computations for h_v function
pub struct HVCache {
    w_term: Array2<f64>,
    exp_terms: Vec<f64>,
    a4_cache: Vec<Array2<f64>>,
    u_cache: Vec<Array2<f64>>,
}

impl HVCache {
    pub fn new(x: &Array2<f64>, y: &Array2<f64>, max_s: usize) -> Self {
        let w_term = w(x, y);
        let exp_terms: Vec<f64> = (0..max_s)
            .map(|r| (-(-1000.0 * (r as f64 - 0.5)).exp()).exp())
            .collect();

        let a4_cache: Vec<Array2<f64>> = (0..max_s)
            .map(|r| Array2::<f64>::ones(x.dim()) - &(a_v_s(x, y, 4.0, r as f64) * 1.25))
            .collect();

        let u_cache: Vec<Array2<f64>> = (0..max_s).map(|r| u_s(x, y, r as f64)).collect();

        HVCache {
            w_term,
            exp_terms,
            a4_cache,
            u_cache,
        }
    }
}

pub fn h_v_optimized(x: &Array2<f64>, y: &Array2<f64>, v: f64, cache: &HVCache) -> Array2<f64> {
    let shape = x.dim();
    let mut result = Array2::zeros(shape);

    // Pre-allocate arrays for common computations
    let mut product = Array2::<f64>::ones(shape);
    let ones = Array2::<f64>::ones(shape);

    for s in 1..31 {
        product.fill(1.0);

        for r in 0..s {
            let a1000_term = a_v_s(x, y, 1000.0, r as f64);

            // Use cached values
            let u_term = &cache.u_cache[r];
            let exp_term = cache.exp_terms[r];
            let a4_term = &cache.a4_cache[r];

            product =
                &product * &(&(&ones - &a1000_term) * &(&ones - &(exp_term * u_term)) * a4_term);
        }

        let fraction_term = (5.0 - 3.0 * (v - 1.0).powi(2) + &cache.w_term) / 10.0;
        let p_term = p_s(x, y, s as f64);
        let exp_term = (&(71.0_f64 / 10.0 - 10.0 * &p_term))
            .mapv(|x| (-x.abs()).exp())
            .mapv(|x| (-x).exp());
        let u_term = &cache.u_cache[s - 1];
        let a1000_term_s = a_v_s(x, y, 1000.0, s as f64);
        let l_term = l_v_s(x, y, v, s as f64);

        result = &result
            + &(&product
                * &(&(&fraction_term * &exp_term * u_term)
                    + &(&a1000_term_s * &(&ones - u_term) * &l_term)));
    }

    result
}

/// Complex pattern generation function H_v
pub fn h_v(x: &Array2<f64>, y: &Array2<f64>, v: f64) -> Array2<f64> {
    let shape = x.dim();
    let mut result = Array2::zeros(shape);

    for s in 1..31 {
        let mut product = Array2::<f64>::ones(shape);

        for r in 0..s {
            let a1000_term = a_v_s(x, y, 1000.0, r as f64);
            let u_term = u_s(x, y, r as f64);
            let exp_term = (-(-1000.0 * (r as f64 - 0.5)).exp()).exp();
            let a4_term = Array2::<f64>::ones(shape) - &(&a_v_s(x, y, 4.0, r as f64) * 1.25);

            product = &product
                * &(&(&Array2::<f64>::ones(shape) - &a1000_term)
                    * &(&Array2::<f64>::ones(shape) - &(exp_term * &u_term))
                    * &a4_term);
        }

        let w_term = w(x, y);
        let fraction_term = (5.0 - 3.0 * (v - 1.0).powi(2) + &w_term) / 10.0;
        let p_term = p_s(x, y, s as f64);
        let exp_term = (&(71.0_f64 / 10.0 - 10.0 * &p_term))
            .mapv(|x| (-x.abs()).exp())
            .mapv(|x| (-x).exp());
        let u_term = u_s(x, y, s as f64);
        let a1000_term_s = a_v_s(x, y, 1000.0, s as f64);
        let l_term = l_v_s(x, y, v, s as f64);

        result = &result
            + &(&product
                * &(&(&fraction_term * &exp_term * &u_term)
                    + &(&a1000_term_s * &(&Array2::<f64>::ones(shape) - &u_term) * &l_term)));
    }

    result
}

/// Complex lighting function L_v_s
pub fn l_v_s(x: &Array2<f64>, y: &Array2<f64>, v: f64, s: f64) -> Array2<f64> {
    let r_term = r_t_s(x, y, 0.0, s);
    let p_term = p_s(x, y, s);
    let c20_term = c_v_s(x, y, 20.0, s);
    let c10_term = c_v_s(x, y, 10.0, s);
    let a4_term = a_v_s(x, y, 4.0, s);
    let a1000_term = a_v_s(x, y, 1000.0, s);
    let b_term = b_s(x, y, s);

    let first_term = 0.1
        - (0.025
            * &r_term
                .mapv(|x| x.clamp(-1.0, 1.0).acos())
                .mapv(|x| (20.0 * x).cos())
            * &p_term.mapv(|x| (25.0 * x).cos()));

    let poly_term = 4.0 * v.powi(2) - 13.0 * v + 11.0;
    let cos_term = (7.0 * s + v * s).cos();

    let exp_term1 = 20.0
        * &(&c20_term - 0.5)
            .mapv(|x| -(-70.0 * x).exp())
            .mapv(|x| x.exp());
    let exp_term2 = 20.0
        * &(&c10_term - 0.5)
            .mapv(|x| -(-10.0 * x).exp())
            .mapv(|x| x.exp());

    let middle_terms = poly_term + cos_term + &exp_term1 + &exp_term2;
    let a_product = &a4_term * &a1000_term;

    &(&first_term * middle_terms) * &a_product + &b_term
}

/// Pattern function C_v_s
pub fn c_v_s(x: &Array2<f64>, y: &Array2<f64>, v: f64, s: f64) -> Array2<f64> {
    let r_term = r_t_s(x, y, 0.0, s);
    let p_term = p_s(x, y, s);
    let q_term = q_s(x, y, s);
    let w_term = w(x, y);

    let arccos_r = r_term.mapv(|x| x.clamp(-1.0, 1.0).acos());
    let p_scaled = &p_term * 12.5;
    let w_scaled = 0.7 + &w_term / 5.0;

    let exp1 = (&arccos_r.mapv(|x| (10.0 * x).cos()) * &p_scaled.mapv(|x| x.cos()) - &w_scaled)
        .mapv(|x| (v * x).exp());

    let exp2 = (&arccos_r.mapv(|x| (10.0 * x).cos()) * &p_scaled.mapv(|x| x.cos()) + &w_scaled)
        .mapv(|x| (-v * x).exp());

    let exp3 = (&arccos_r.mapv(|x| (10.0 * x).sin()) * &p_scaled.mapv(|x| x.sin()) - &w_scaled)
        .mapv(|x| (v * x).exp());

    let exp4 = (&arccos_r.mapv(|x| (10.0 * x).sin()) * &p_scaled.mapv(|x| x.sin()) + &w_scaled)
        .mapv(|x| (-v * x).exp());

    let term =
        1.5 * ((&q_term * &q_term) + &(&p_term - 0.25) * &(&p_term - 0.25) - 0.42 + &w_term / 5.0);
    let final_exp = term.mapv(|x| -x.exp()).mapv(|x| x.exp());

    (-(&exp1 + &exp2 + &exp3 + &exp4)).mapv(|x| x.exp()) * &final_exp
}

/// Pattern function B_s
pub fn b_s(x: &Array2<f64>, y: &Array2<f64>, s: f64) -> Array2<f64> {
    let r_term = r_t_s(x, y, 0.0, s);
    let p_term = p_s(x, y, s);

    let inner_cos = r_term
        .mapv(|x| x.clamp(-1.0, 1.0).acos() * 20.0)
        .mapv(|x| x.cos());
    let outer_cos = (&p_term * 25.0).mapv(|x| x.cos());

    let exponent = -70.0 * (&inner_cos * &outer_cos - 0.94);
    exponent.mapv(|x| -x.exp()).mapv(|x| x.exp())
}

/// Pattern function A_v_s
pub fn a_v_s(x: &Array2<f64>, y: &Array2<f64>, v: f64, s: f64) -> Array2<f64> {
    let p_term = p_s(x, y, s);
    let q_term = q_s(x, y, s);

    let exp1 = (-1000.0 * (s - 0.5)).exp();

    let q_squared = &q_term * &q_term;
    let p_squared = &p_term * &p_term;

    let middle_term: Array2<f64> = v
        * ((1.25 * &(Array2::ones(x.dim()) - &p_term) * &q_squared) + &p_squared - 0.55
            + (100.0 * (v - 100.0)).atan() / (10.0 * PI));

    let last_term = v * (&q_squared + &p_squared - 1.0);

    (-exp1 - &middle_term.mapv(|x| x.exp()) - &last_term.mapv(|x| x.exp())).mapv(|x| x.exp())
}

/// Pattern function U_s
pub fn u_s(x: &Array2<f64>, y: &Array2<f64>, s: f64) -> Array2<f64> {
    let m_term = m_s(x, y, s);
    let n_term = n_s(x, y, s);

    Array2::<f64>::ones(x.dim())
        - &(&(Array2::<f64>::ones(x.dim()) - &m_term) * &(Array2::<f64>::ones(x.dim()) - &n_term))
}

/// Pattern function M_s
pub fn m_s(x: &Array2<f64>, y: &Array2<f64>, s: f64) -> Array2<f64> {
    let p_term = p_s(x, y, s);
    let q_term = q_s(x, y, s);
    let r_term = r_t_s(x, y, 0.0, s);

    let inner_cos_term1 = (&q_term * 7.0 + 2.0 * s).mapv(|x| x.cos()) / 10.0;
    let term1 = &p_term - 0.57 - (0.15 + &inner_cos_term1);

    let arccos_term = r_term.mapv(|x| x.clamp(-1.0, 1.0).acos());

    let shape = x.dim();
    let mut inner_cos_term2 = Array2::zeros(shape);
    Zip::from(&mut inner_cos_term2)
        .and(x)
        .and(y)
        .par_for_each(|out, &x_val, &y_val| {
            *out = 0.3 * (45.0 * x_val + 47.0 * y_val + (17.0 * x_val).cos()).cos()
                + 2.0 * (5.0 * s).cos();
        });

    let cos_combined =
        (&arccos_term * (10.0 + 3.0 * (14.0 * s).cos()) + &inner_cos_term2).mapv(|x| x.cos());

    let exp1 = (-100.0 * &term1 * &cos_combined).mapv(|x| x.exp());
    let term2 = &p_term - 0.72 * (1.5 * &q_term).mapv(|x| x.powi(8));
    let exp2 = (1000.0 * term2).mapv(|x| x.exp());

    (-(&exp1 + &exp2)).mapv(|x| x.exp())
}

/// Pattern function N_s
pub fn n_s(x: &Array2<f64>, y: &Array2<f64>, s: f64) -> Array2<f64> {
    let p_term = p_s(x, y, s);
    let q_term = q_s(x, y, s);
    let r_term = r_t_s(x, y, 1.0, s);

    let shape = x.dim();
    let mut inner_cos_term1 = Array2::zeros(shape);
    Zip::from(&mut inner_cos_term1)
        .and(x)
        .and(&q_term)
        .par_for_each(|out, &x_val, &q_val| {
            *out = (8.0 * q_val * x_val + 5.0 * s).cos() / 10.0;
        });

    let term1 = &p_term - 0.74 - (0.15 + &inner_cos_term1);
    let arccos_term = r_term.mapv(|x| x.clamp(-1.0, 1.0).acos());

    let mut inner_cos_term2 = Array2::zeros(shape);
    Zip::from(&mut inner_cos_term2)
        .and(x)
        .and(y)
        .par_for_each(|out, &x_val, &y_val| {
            *out = 0.3 * (38.0 * x_val - 47.0 * y_val + (19.0 * x_val).cos()).cos()
                + 2.0 * (4.0 * s).cos();
        });

    let cos_combined =
        (&arccos_term * (10.0 + 3.0 * (16.0 * s).cos()) + &inner_cos_term2).mapv(|x| x.cos());

    let exp1 = (100.0 * &term1 * &cos_combined).mapv(|x| x.exp());
    let term2 = &p_term - 0.71 * (1.5 * &q_term).mapv(|x| x.powi(8));
    let exp2 = (-1000.0 * term2).mapv(|x| x.exp());

    (-(&exp1 + &exp2)).mapv(|x| x.exp())
}

/// Pattern function R_t_s
pub fn r_t_s(x: &Array2<f64>, y: &Array2<f64>, t: f64, s: f64) -> Array2<f64> {
    let e_term = e_t_s(x, y, t, s);
    let exponent = 1000.0 * (&e_term.mapv(|x| x.abs()) - 1.0);
    &e_term * &exponent.mapv(|x| -x.exp()).mapv(|x| x.exp())
}

/// Pattern function E_t_s
pub fn e_t_s(x: &Array2<f64>, y: &Array2<f64>, t: f64, s: f64) -> Array2<f64> {
    let q_term = q_s(x, y, s);
    let p_term = p_s(x, y, s);

    let sqrt_term1 = (5.0 * (20.0 - 20.0 * (1.0 - 2.0 * t) * &p_term - 27.0 * t).mapv(|x| x.abs()))
        .mapv(|x| x.sqrt());
    let inner_term = 200.0 - (20.0 * (1.0 - 2.0 * t) * &p_term + 27.0 * t).mapv(|x| x.powi(2));
    let sqrt_term2 = (4.0 * inner_term.mapv(|x| x.abs())).mapv(|x| x.sqrt());
    let denom_term = (1.0 + 50.0 * &sqrt_term2).mapv(|x| 1.0 / x);

    &(&q_term * &sqrt_term1) * &denom_term * (1000.0 / 20.0_f64.sqrt())
}

/// Pattern generation function P_s
/// Returns pattern values based on periodic variations of coordinates and pattern parameter s
pub fn p_s(x: &Array2<f64>, y: &Array2<f64>, s: f64) -> Array2<f64> {
    let shape = x.dim();
    let mut result = Array2::zeros(shape);

    // Precalculate some common values
    let sin_5s = (5.0 * s).sin();
    let cos_5s = (5.0 * s).cos();

    // Use Zip for better performance with element-wise operations
    Zip::from(&mut result)
        .and(x)
        .and(y)
        .par_for_each(|result, &x_val, &y_val| {
            // Calculate main component: 2 * sin(5s)x - 2 * cos(5s)y + 3 * cos(5s)
            let main_term = 2.0 * sin_5s * x_val - 2.0 * cos_5s * y_val + 3.0 * cos_5s;

            // Calculate modulation: 3 * cos(14x - 19y + 5s) / 200
            let modulation = 3.0 * (14.0 * x_val - 19.0 * y_val + 5.0 * s).cos() / 200.0;

            // Combine terms using arctan(tan()) structure for periodic wrapping
            *result = (main_term + modulation).tan().atan();
        });

    result
}

/// Pattern generation function Q_s
/// Creates periodic variations based on coordinates and pattern parameter s
pub fn q_s(x: &Array2<f64>, y: &Array2<f64>, s: f64) -> Array2<f64> {
    let shape = x.dim();
    let mut result = Array2::zeros(shape);

    // Precalculate common terms
    let cos_5s = (5.0 * s).cos();
    let sin_5s = (5.0 * s).sin();
    let cos_4s = 2.0 * (4.0 * s).cos();

    Zip::from(&mut result)
        .and(x)
        .and(y)
        .par_for_each(|result, &x_val, &y_val| {
            // Main periodic term: 2(cos(5s)x + sin(5s)y + 2cos(4s))
            let main_term = 2.0 * (cos_5s * x_val + sin_5s * y_val + cos_4s);

            // Fine detail modulation: 3cos(18x + 15y + 4s)/200
            let cos_term = 3.0 * (18.0 * x_val + 15.0 * y_val + 4.0 * s).cos() / 200.0;

            // Combine with periodic wrapping
            *result = (main_term + cos_term).tan().atan();
        });

    result
}

/// Texture detail function W
/// Creates fine surface patterns by summing multiple frequency components
pub fn w(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    let shape = x.dim();
    let mut result = Array2::zeros(shape);

    // Precalculate powers to avoid repeated computation
    let powers: Vec<f64> = (1..41).map(|s| (28.0_f64 / 25.0).powi(s)).collect();

    // Calculate sin/cos values for each frequency
    let sin_vals: Vec<(f64, f64)> = (1..41)
        .map(|s| ((5.0 * s as f64).sin(), (6.0 * s as f64).sin()))
        .collect();

    let cos_vals: Vec<f64> = (1..41).map(|s| (2.0 * s as f64).cos()).collect();

    let sin_2s: Vec<f64> = (1..41).map(|s| (2.0 * s as f64).sin()).collect();

    // Perform the main computation
    Zip::from(&mut result)
        .and(x)
        .and(y)
        .par_for_each(|result, &x_val, &y_val| {
            let mut sum = 0.0;

            for s in 0..40 {
                let power = powers[s];
                let (sin_5s, sin_6s) = sin_vals[s];
                let cos_2s = cos_vals[s];
                let sin_2s = sin_2s[s];

                // Calculate main terms
                let term1 = power * (cos_2s * x_val + sin_2s * y_val) + 2.0 * sin_5s;
                let term2 = power * (cos_2s * y_val - sin_2s * x_val) + 2.0 * sin_6s;

                // Calculate combined pattern
                let pattern = term1.cos().powi(2) * term2.cos().powi(2);

                // Add contribution from this frequency
                sum += (-(-3.0 * (pattern - 0.97)).exp()).exp();
            }

            *result = sum;
        });

    result
}

// Add this at the end of utils.rs, after all the function implementations

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array; // Add approx = "0.5" to your Cargo.toml

    fn assert_array1d_almost_equal(actual: &Array1<f64>, expected: &Array1<f64>, tol: f64) {
        assert_eq!(actual.shape(), expected.shape());
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < tol,
                "At index {}: actual={}, expected={}, diff={}",
                i,
                a,
                b,
                (a - b).abs()
            );
        }
    }

    fn assert_array_almost_equal(a: &Array2<f64>, b: &Array2<f64>, tolerance: f64) {
        assert_eq!(a.shape(), b.shape(), "Arrays have different shapes");

        for ((i, j), &value) in a.indexed_iter() {
            let diff = (value - b[[i, j]]).abs();
            assert!(
                diff < tolerance,
                "Arrays differ at position [{}, {}]: {} vs {} (diff: {})",
                i,
                j,
                value,
                b[[i, j]],
                diff
            );
        }
    }

    #[test]
    fn test_f() {
        // Test zero input
        assert_eq!(f(0.0), 0);

        // Test array input
        let test_array: Array1<f64> = array![-1.5, -1.0, 0.0, 0.5, 1.0, 1.5];
        let expected = array![0.0, 0.0, 0.0, 127.0, 255.0, 255.0];

        // Apply f function to each element
        let result: Array1<f64> = test_array.mapv(|x| f(x) as f64);

        // Check shape and bounds
        assert_eq!(result.shape(), test_array.shape());
        assert!(result.iter().all(|&x| x >= 0.0));
        assert!(result.iter().all(|&x| x <= 255.0));

        // Assert arrays are almost equal
        assert_array1d_almost_equal(&result, &expected, 1e-6);
    }

    #[test]
    fn test_m_s() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let s = 3.0;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 0.99995759, 1.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.]
        ];

        let result = m_s(&x_grid, &y_grid, s);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_h_v_optimized() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let v = 1.0;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Create cache for optimized version
        let cache = HVCache::new(&x_grid, &y_grid, 31);

        // Expected values (same as h_v test)
        let expected = array![
            [0.64896535, 0.6587483, 0.72895193, 0.70749145, 0.10552709],
            [0.62254445, 0.57186737, 0.7381196, 0.68524103, 0.63553497],
            [0.14702198, 0.67405708, 0.3522309, 0.70226551, 0.58097669],
            [0.68475501, 0.7350397, 0.10276321, 0.6485094, 0.58708438],
            [0.61895038, 0.64118258, 0.63125632, 0.60637943, 0.58010723]
        ];

        let result = h_v_optimized(&x_grid, &y_grid, v, &cache);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_h_v() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let v = 1.0;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [0.64896535, 0.6587483, 0.72895193, 0.70749145, 0.10552709],
            [0.62254445, 0.57186737, 0.7381196, 0.68524103, 0.63553497],
            [0.14702198, 0.67405708, 0.3522309, 0.70226551, 0.58097669],
            [0.68475501, 0.7350397, 0.10276321, 0.6485094, 0.58708438],
            [0.61895038, 0.64118258, 0.63125632, 0.60637943, 0.58010723]
        ];

        let result = h_v(&x_grid, &y_grid, v);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_u_s() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let s = 1.0;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [1., 1., 1., 1., 0.],
            [0.99669478, 1., 0., 1., 1.],
            [0., 0., 0., 1., 1.],
            [1., 1., 0., 1., 1.],
            [1., 0.99084226, 1., 0.99994164, 0.]
        ];

        let result = u_s(&x_grid, &y_grid, s);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_l_v_s() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let v = 0.0;
        let s = 3.0;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [
                0.00000000e+000,
                0.00000000e+000,
                0.00000000e+000,
                7.40140805e-113,
                0.00000000e+000
            ],
            [
                0.00000000e+000,
                0.00000000e+000,
                9.01695753e-001,
                3.31054999e-001,
                0.00000000e+000
            ],
            [
                0.00000000e+000,
                0.00000000e+000,
                0.00000000e+000,
                5.38877809e-001,
                5.62868899e-002
            ],
            [
                0.00000000e+000,
                0.00000000e+000,
                9.84917740e-001,
                0.00000000e+000,
                0.00000000e+000
            ],
            [
                0.00000000e+000,
                0.00000000e+000,
                6.65512335e-002,
                0.00000000e+000,
                6.39464138e-001
            ]
        ];

        let result = l_v_s(&x_grid, &y_grid, v, s);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_a_v_s() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let v = 10.0;
        let s = 3.0;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [
                0.00000000e+000,
                0.00000000e+000,
                0.00000000e+000,
                0.00000000e+000,
                5.57598586e-001
            ],
            [
                0.00000000e+000,
                6.38222163e-001,
                9.80054410e-001,
                0.00000000e+000,
                0.00000000e+000
            ],
            [
                0.00000000e+000,
                1.35334677e-022,
                1.82670604e-002,
                0.00000000e+000,
                0.00000000e+000
            ],
            [
                4.82518856e-001,
                0.00000000e+000,
                0.00000000e+000,
                7.41141995e-003,
                2.80475047e-001
            ],
            [
                6.65216903e-162,
                0.00000000e+000,
                0.00000000e+000,
                2.78856680e-006,
                9.59571497e-001
            ]
        ];

        let result = a_v_s(&x_grid, &y_grid, v, s);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_c_v_s() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let v = 10.0;
        let s = 3.0;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [
                1.39082265e-002,
                3.55189155e-017,
                3.41060823e-010,
                3.73292196e-009,
                2.25669849e-001
            ],
            [
                6.49950225e-052,
                3.67114263e-002,
                2.32674876e-001,
                1.35243906e-005,
                2.31450361e-023
            ],
            [
                4.65702359e-012,
                1.79286009e-002,
                1.89352893e-001,
                1.36501034e-007,
                4.06202820e-037
            ],
            [
                1.18004406e-001,
                2.18746913e-031,
                9.01210508e-107,
                5.12158536e-003,
                8.48493174e-002
            ],
            [
                1.06080439e-001,
                1.78862152e-064,
                2.73349964e-057,
                1.09154215e-001,
                2.97222568e-001
            ]
        ];

        let result = c_v_s(&x_grid, &y_grid, v, s);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_b_s() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let s = 3.0;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [
                0.00000000e+000,
                0.00000000e+000,
                0.00000000e+000,
                7.40140805e-113,
                0.00000000e+000
            ],
            [
                0.00000000e+000,
                0.00000000e+000,
                0.00000000e+000,
                3.31054999e-001,
                0.00000000e+000
            ],
            [
                0.00000000e+000,
                0.00000000e+000,
                0.00000000e+000,
                5.38877809e-001,
                5.62868899e-002
            ],
            [
                0.00000000e+000,
                0.00000000e+000,
                9.84917740e-001,
                0.00000000e+000,
                0.00000000e+000
            ],
            [
                0.00000000e+000,
                0.00000000e+000,
                6.65512335e-002,
                0.00000000e+000,
                0.00000000e+000
            ]
        ];

        let result = b_s(&x_grid, &y_grid, s);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_n_s() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let s = 3.0;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [
                9.99982212e-01,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                8.92290347e-31
            ],
            [
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00
            ],
            [
                0.00000000e+00,
                0.00000000e+00,
                4.10016451e-02,
                0.00000000e+00,
                0.00000000e+00
            ],
            [
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00
            ],
            [
                1.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                0.00000000e+00,
                5.02621223e-01
            ]
        ];

        let result = n_s(&x_grid, &y_grid, s);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_r_t_s() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let t = 0.5;
        let s = PI / 4.0;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [-0., 0., 0., -0.50018261, -0.],
            [0., 0., -0.47240285, -0., -0.],
            [0., -0.41545348, -0., -0., 0.],
            [-0.43517637, -0., -0., 0., 0.44633812],
            [-0., 0., 0., 0.46939089, -0.]
        ];

        let result = r_t_s(&x_grid, &y_grid, t, s);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_r_t_s_2() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let t = 0.5;
        let s = 0.1;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [0., -0., -0., 0., 0.],
            [0., -0., 0.22184502, 0., -0.],
            [-0., -0., 0., 0., -0.],
            [-0., 0.45670081, 0., -0., -0.],
            [-0.80449116, 0., 0., -0., 0.32917457]
        ];

        let result = r_t_s(&x_grid, &y_grid, t, s);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_e_t_s() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let t = 0.5;
        let s = PI / 4.0;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [-3.5358463, 3.7929558, 1.71212432, -0.50018261, -2.54631759],
            [3.76990304, 1.7095303, -0.47240285, -2.59434552, -4.70404606],
            [
                1.64782907,
                -0.41545348,
                -2.63642087,
                -4.68432317,
                2.59313384
            ],
            [
                -0.43517637,
                -2.59434552,
                -4.74127253,
                2.65483507,
                0.44633812
            ],
            [-2.54631759, 4.71399186, 2.6574291, 0.46939089, -1.64523674]
        ];

        let result = e_t_s(&x_grid, &y_grid, t, s);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_p_s() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let s = 1.0;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [0.18198075, -0.7684849, 1.42467361, 0.47310779, -0.48521152],
            [
                -0.07593162,
                -1.04282011,
                1.12932488,
                0.16249759,
                -0.79781949
            ],
            [
                -0.3864245,
                -1.33792403,
                0.85524149,
                -0.09527991,
                -1.05203142
            ],
            [
                -0.64232686,
                1.53349822,
                0.56416243,
                -0.40361706,
                -1.36548176
            ],
            [-0.95452101, 1.23437052, 0.28571324, -0.66395356, 1.52240629]
        ];

        let result = p_s(&x_grid, &y_grid, s);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_q_s() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let s = 1.0;

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [
                -1.27527117,
                -0.97426689,
                -0.69665955,
                -0.41930595,
                -0.11809303
            ],
            [0.9053912, 1.21724723, 1.47189559, -1.36135798, -1.0936497],
            [-0.03825514, 0.24761092, 0.51721352, 0.82429206, 1.07934313],
            [
                -0.98458166,
                -0.72758544,
                -0.42465653,
                -0.14943737,
                0.13034348
            ],
            [1.19154244, 1.45451402, -1.37599981, -1.12160728, -0.8120248]
        ];

        let result = q_s(&x_grid, &y_grid, s);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);
        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_w_values() {
        // Create linear space arrays
        let x_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);
        let y_1d: Array1<f64> = Array1::linspace(-1.0, 1.0, 5);

        // Create meshgrid
        let (x_grid, y_grid) = meshgrid(&x_1d, &y_1d);

        // Expected values
        let expected = array![
            [1.52277955, 1.58748552, 2.29526151, 2.76896464, 1.2076526],
            [1.22785052, 0.71867386, 2.38199789, 1.88118299, 1.35535153],
            [1.21651577, 1.82091608, 0.44520765, 2.02489012, 0.80976706],
            [1.84755924, 2.35234662, 0.90028912, 1.48518858, 0.87084379],
            [1.18950413, 1.44546374, 1.40390614, 1.06382403, 1.65016023]
        ];

        // Calculate W
        let result = w(&x_grid, &y_grid);

        // Print values for inspection
        println!("\nExpected values:");
        println!("{:#?}", expected);
        println!("\nCalculated values:");
        println!("{:#?}", result);
        println!("\nDifference:");
        println!("{:#?}", &result - &expected);

        // Assert arrays are almost equal
        assert_array_almost_equal(&result, &expected, 1e-6);

        assert_eq!(result.shape(), &[5, 5]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_combined_functions() {
        let x = array![[0.0]];
        let y = array![[0.0]];
        let s = 1.0;
        let v = 1.0;

        // Test that compositions of functions don't panic
        let p = p_s(&x, &y, s);
        let q = q_s(&x, &y, s);
        let w_val = w(&x, &y);
        let h = h_v(&x, &y, v);

        assert!(p[[0, 0]].is_finite());
        assert!(q[[0, 0]].is_finite());
        assert!(w_val[[0, 0]].is_finite());
        assert!(h[[0, 0]].is_finite());
    }
}
