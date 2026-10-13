use core_lm::geometry::fixed::Q32_31;
use core_lm::geometry::lorentz::LorentzVector;
use core_lm::geometry::torus::TorPhaseEncoder;

#[test]
fn test_sqrt() {
    let input = Q32_31::from_f64(4.0);
    let expected = Q32_31::from_f64(2.0);
    assert_eq!(input.sqrt(), expected, "sqrt(4.0) should be 2.0");

    let input_small = Q32_31::from_f64(0.25);
    let expected_small = Q32_31::from_f64(0.5);
    assert_eq!(
        input_small.sqrt(),
        expected_small,
        "sqrt(0.25) should be 0.5"
    );
}

#[test]
fn test_inv_sqrt() {
    let input = Q32_31::from_f64(4.0);
    let expected = Q32_31::from_f64(0.5);
    assert_eq!(input.inv_sqrt(), expected, "inv_sqrt(4.0) should be 0.5");
}

#[test]
fn test_fast_sin() {
    let pi = Q32_31::from_f64(std::f64::consts::PI);
    let pi_half = Q32_31::from_f64(std::f64::consts::FRAC_PI_2);

    // Tolerance for LUT interpolation
    let _tol = Q32_31::from_f64(0.01);

    let sin_pi = Q32_31::sin(pi);
    let diff_pi = (sin_pi - Q32_31::ZERO).to_f64().abs();
    assert!(
        diff_pi < 0.01,
        "sin(pi) should be ~0, got {:.4}",
        sin_pi.to_f64()
    );

    let sin_pi_half = Q32_31::sin(pi_half);
    let diff_pi_half = (sin_pi_half - Q32_31::ONE).to_f64().abs();
    assert!(
        diff_pi_half < 0.01,
        "sin(pi/2) should be ~1, got {:.4}",
        sin_pi_half.to_f64()
    );
}

#[test]
fn test_phi() {
    let encoder = TorPhaseEncoder::new();
    let theta = Q32_31::from_f64(0.0);
    let res = encoder.phi(theta);

    let inv_sqrt2 = Q32_31::from_f64(1.0 / 2.0f64.sqrt());
    assert_eq!(res[0], inv_sqrt2);
    assert_eq!(res[1], inv_sqrt2); // cos(0) = 1
    assert_eq!(res[2], Q32_31::ZERO); // sin(0) = 0
}

#[test]
fn test_boost_identity() {
    let origin = LorentzVector::origin();
    let n = [Q32_31::ONE, Q32_31::ZERO, Q32_31::ZERO];
    let beta = Q32_31::ZERO;
    let boosted = LorentzVector::boost(&origin, &n, beta);
    assert_eq!(boosted, origin, "Boost with beta=0 should be identity");
}

#[test]
fn test_inner_product_origin() {
    let origin = LorentzVector::origin();
    let ip = LorentzVector::inner_product(&origin, &origin);
    assert_eq!(
        ip,
        Q32_31::ONE,
        "Inner product of origin with itself should be 1.0"
    );
}
