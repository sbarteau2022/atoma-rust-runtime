fn main() {
    let mut engine = core_lm::Engine::new().expect("engine init");
    engine.seed_knowledge(vec![("health-check", "core_lm ready")]);
    match engine.generate("health-check", 1) {
        Ok(text) => println!("{text}"),
        Err(err) => {
            eprintln!("health-check failed: {err}");
            std::process::exit(1);
        }
    }
}
