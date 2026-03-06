fn test_model() {
    let _m = llmfit_core::models::Model {
        name: String::new(),
        repo: String::new(),
        params_b: 0.0,
        ..Default::default()
    };
    let _ = _m.not_a_real_field;
}
