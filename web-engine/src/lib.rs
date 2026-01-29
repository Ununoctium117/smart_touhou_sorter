use std::cell::RefCell;

use engine::{CharacterMetadata, CharacterSortingData};
use wasm_bindgen::prelude::*;

thread_local! {
    static SORTING_STATES: RefCell<Vec<engine::CharacterSortingData>> = const { RefCell::new(Vec::new()) };
}

/// Returns a handle for the created sorter.
#[wasm_bindgen]
pub fn create_sorter(character_ids: Vec<String>, num_candidate_lists: usize) -> usize {
    SORTING_STATES.with_borrow_mut(|states| {
        let metadata = character_ids
            .into_iter()
            .map(|id| CharacterMetadata {
                globally_unique_id: id,
            })
            .collect();

        states.push(CharacterSortingData::new(metadata, num_candidate_lists));
        states.len() - 1
    })
}

/// Estimates the progress for a sorter.
#[wasm_bindgen]
pub fn estimate_progress(handle: usize) -> f64 {
    SORTING_STATES.with_borrow(|states| states[handle].estimate_progress())
}

/// Saves the state of a sorter to a string, which can later be passed to `deserialize_sorter`.
#[wasm_bindgen]
pub fn serialize_sorter(handle: usize) -> String {
    SORTING_STATES.with_borrow(|states| serde_json::to_string(&states[handle]).unwrap())
}

#[wasm_bindgen]
pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}
