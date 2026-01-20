mod utils;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(name: &str) {
    web_sys::console::log_1(&format!("something {name}").into());
    alert(&format!("Hello, {name}!!"));
}

#[wasm_bindgen]
pub fn set_panic_hook() {
    utils::set_panic_hook();
}
