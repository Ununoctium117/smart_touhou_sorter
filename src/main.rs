//! This is based on the paper "Monte Carlo Sort for unreliable human comparisons" by Samuel L. Smith:
//! https://arxiv.org/abs/1612.08555.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
};

use anyhow::{Context, Result};
use clap::Parser as _;
use flate2::{Compression, write::GzEncoder};
use jiff::Timestamp;
use ordered_float::NotNan;
use rand::{Rng as _, RngCore, seq::SliceRandom as _};
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(test, derive(Default))]
struct CharacterMetadata {
    #[serde(rename = "g")]
    globally_unique_id: String,
    #[serde(rename = "d")]
    display_name: String,
    #[serde(rename = "u")]
    image_url: String,
}

/// Sorting IDs are numeric and must start at 0 with no gaps, to allow matchups to be encoded into indices.
#[derive(Debug, Serialize, Deserialize, Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[serde(transparent)]
struct SortingIndex(usize);

#[derive(Debug, Serialize, Deserialize, Hash, PartialEq, Eq, PartialOrd, Ord, Clone)]
struct Matchup {
    // invariant: a < b
    a: SortingIndex,
    b: SortingIndex,
}
impl Matchup {
    fn new(a: SortingIndex, b: SortingIndex) -> Self {
        assert_ne!(a, b);

        Self {
            a: a.min(b),
            b: a.max(b),
        }
    }

    // Matchups are ordered as follows (in this example, the max character ID is 4):
    // (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)

    fn to_index(&self, max_sorting_index: SortingIndex) -> MatchupIndex {
        let a = self.a.0;
        let b = self.b.0;
        let n = max_sorting_index.0;

        assert!(a <= n);
        assert!(b <= n);

        let block_starting_index = Self::find_block_starting_index(a, n);
        let block_offset = b - a - 1;

        let index = block_starting_index + block_offset;
        MatchupIndex(index)
    }

    fn from_index(index: MatchupIndex, max_sorting_index: SortingIndex) -> Self {
        let index = index.0;
        let n = max_sorting_index.0;

        // The naive decoder is a loop over blocks (where each block has the same a).
        // However, we have a closed-form quadratic function to find the starting index
        // of each block, so to find a (our first element), we just need to find the
        // largest a that satisfies:
        //      find_block_starting_index(a, n) <= index
        // Which means:
        //      a * n - (a * (a - 1) / 2) <= index
        // Rewriting into standard form:
        //      -0.5a * (a - 2n - 1) <= index
        //      -0.5a^2 + (-0.5a)(-2)n + (-1)(-0.5a) <= index
        //      -0.5a^2 + 2na/2 + a/2 <= index
        //      -1/2a^2 + (2n+1)/2 * a - index <= 0
        //      a^2 - (2n+1)a + 2 * index <= 0          (factoring out -1/2)
        // Solving using the standard quadratic formula:
        //      a = floor((2n + 1 - sqrt((2n+1)^2 - 8 * index)) / 2)
        let a = {
            let coeff_b = 2 * n + 1;
            // "8" comes from quadratic formula - 4 * a * c, with a = 1 and c = 2 * index.
            let Some(discriminant) = (coeff_b * coeff_b).checked_sub(8 * index) else {
                panic!("invalid index {index} with n {n}");
            };

            let sqrt_discriminant_lower_bound = discriminant.isqrt();
            let a_candidate = (coeff_b - sqrt_discriminant_lower_bound) / 2;

            // We've completed evaluating the quadratic formula, but our a candidate may be
            // one too high since we used the lower bound of the sqrt of the discriminant as an optimization.
            // Our actual index must be in the half-open range:
            //   [find_block_starting_index(true_a, n), find_block_starting_index(true_a + 1, n))
            // So if find_block_starting_index(a_candidate, n) is too small, then our candidate is too high.
            let decrement_needed = index < Self::find_block_starting_index(a_candidate, n);
            a_candidate - (decrement_needed as usize)
        };

        let b = (index - Self::find_block_starting_index(a, n)) + a + 1;

        Self {
            a: SortingIndex(a),
            b: SortingIndex(b),
        }
    }

    /// Used for index encoding/decoding. Finds the index of the block of matchups whose smaller element is a,
    /// where the max sorting id is n.
    fn find_block_starting_index(a: usize, n: usize) -> usize {
        // There are n pairs that start with 0, n-1 pairs that start with 1, n-2 that start with 2,
        // etc. This means that the number of pairs that come before any pair that starts with a is:
        //      sum from i=0 to a-1 of (n - i)
        // which is equal to (thanks wolframalpha):
        //      -0.5 * a * (a - 2 * n - 1)
        // Rewriting via algebraic manipulation to avoid floating point operations:
        //      a * n - ((a * (a - 1)) / 2)
        // (Note that a * (a - 1) is always even, so the division by 2 can be done with integer logic.)

        // We use wrapping_sub here to be explicit about the intent when a = 0, especially
        // in debug builds where the underflow would otherwise panic. When a = 0, wrapping_sub
        // produces u64::MAX, but it's multiplied by zero so the result of the calculation is
        // still correct.
        let difference = (a * a.wrapping_sub(1)) / 2;
        a * n - difference
    }
}

#[derive(Debug, Serialize, Deserialize, Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[serde(transparent)]
struct MatchupIndex(usize);

/// A positive number means that, for a given matchup, a < b.
/// A negative number means that b < a.
/// The magnitude of the number represents the "strength" of the decision,
/// which affects the probability of the evaluation being correct. An outcome
/// of 2.0 means that the evaluation is twice as likely to be correct as if the
/// outcome were 1.0.
#[derive(Debug, Serialize, Deserialize, PartialEq, PartialOrd, Clone, Copy)]
#[serde(transparent)]
struct EvaluationOutcome(f64);

// A unique identifier for a candidate list.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
struct CandidateListIndex(usize);

// A group of identical candidate lists.
type CandidateListGroup = SmallVec<[CandidateListIndex; 2]>;

// Maps from matchup to evaluation, and the time at which the evaluation occurred.
// This BTreeMap is basically used as a sparse vec.
type ComparisonHistory = BTreeMap<MatchupIndex, Vec<(Timestamp, EvaluationOutcome)>>;

trait DebugRng: RngCore + std::fmt::Debug {}
impl<T> DebugRng for T where T: RngCore + std::fmt::Debug {}
fn default_rng() -> Box<dyn DebugRng> {
    Box::new(rand::rng())
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedMatchupValue {
    #[serde(rename = "n")]
    n_ab: u64,
    #[serde(rename = "c")]
    convergence: NotNan<f64>,
}

// The different strategies we have to sample the probability distribution of possible orderings.
enum SamplingStrategy<'a> {
    // This strategy is most effective when the number of measurements is still small.
    Recursion {
        comparison_history: &'a ComparisonHistory,
        max_sorting_index: SortingIndex,
    },
    // This strategy's runtime does not depend on the total number of measurements,
    // but may be slower than the recursion sampler when the number of measurements is small.
    ByMaxElement {
        comparison_history: &'a ComparisonHistory,
        max_sorting_index: SortingIndex,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct SampleCandidateList {
    // Invariant: Each candidate list contains every SortingIndex in the metadata, and no others. This
    // implies that each candidate list has the same length which is equal to metadata.len().
    #[serde(rename = "o")]
    ordering: Vec<SortingIndex>,

    #[serde(rename = "h")]
    comparisons_seen: BTreeSet<(MatchupIndex, usize)>,

    #[serde(rename = "m")]
    matching_comparisons_seen: usize,
}
impl SampleCandidateList {
    fn new(metadata: &Vec<CharacterMetadata>, rng: &mut dyn RngCore) -> Self {
        let ordering = {
            let mut ordering = (0..metadata.len()).map(SortingIndex).collect::<Vec<_>>();
            ordering.shuffle(rng);
            ordering
        };

        Self {
            ordering,
            comparisons_seen: BTreeSet::new(),
            matching_comparisons_seen: 0,
        }
    }

    // Let's say we just got a measurement, i < j, that contradicts this sample. This method allows us to resample
    // just the section of this list that disagrees with the measurement.
    //
    // This list has structure a..bj..ic..d (although a..b and/or c..d may be empty). This method will resample only
    // the middle section j..i.
    fn resample_range(
        &mut self,
        matchup: &Matchup,
        sampling_strategy: SamplingStrategy<'_>,
        rng: &mut dyn RngCore,
    ) {
        let p = self.get_correct_measurement_probability();

        let mut range_to_resample = {
            let a_position = self.ordering.iter().position(|x| *x == matchup.a).unwrap();
            let b_position = self.ordering.iter().position(|x| *x == matchup.b).unwrap();

            if a_position < b_position {
                &mut self.ordering[a_position..=b_position]
            } else {
                &mut self.ordering[b_position..=a_position]
            }
        };

        match sampling_strategy {
            SamplingStrategy::Recursion {
                comparison_history,
                max_sorting_index,
            } => {
                fn recursion_helper(
                    slice_to_reorder: &mut [SortingIndex],
                    comparison_history: &ComparisonHistory,
                    comparisons_seen: &BTreeSet<(MatchupIndex, usize)>,
                    correctness_probability: f64,
                    max_sorting_index: SortingIndex,
                    rng: &mut dyn RngCore,
                ) {
                    if slice_to_reorder.len() <= 1 {
                        return;
                    }

                    // Within this slice, we randomly assign half the elements to the first half
                    // and half to the second half, then accept the partition with probability:
                    //      ((1-p)/p)^(n_dispute), where:
                    //          n_dispute is the number of measurements seen by this sample that do not match this partition, and
                    //          p is the error probability
                    let base_probability =
                        (1.0 - correctness_probability) / correctness_probability;
                    debug_assert!(base_probability <= 1.0, "{base_probability}");

                    slice_to_reorder.shuffle(rng);
                    let (mut left, mut right) =
                        slice_to_reorder.split_at_mut(slice_to_reorder.len() / 2);

                    let mut iteration = 0;
                    loop {
                        let n_dispute = comparisons_seen
                            .iter()
                            .filter(|(matchup_index, measurement_index)| {
                                let matchup =
                                    Matchup::from_index(*matchup_index, max_sorting_index);

                                let (_, evaluation_outcome) = comparison_history
                                    .get(matchup_index)
                                    .unwrap()
                                    .get(*measurement_index)
                                    .unwrap();

                                let left_contains_a = left.contains(&matchup.a);
                                let left_contains_b = left.contains(&matchup.b);
                                let right_contains_a = !left_contains_a;
                                let right_contains_b = !left_contains_b;

                                let a_before_b = evaluation_outcome.0.is_sign_positive();
                                let b_before_a = !a_before_b;

                                if a_before_b && left_contains_b && right_contains_a {
                                    // this measurement disputes this partitioning
                                    return true;
                                }

                                if b_before_a && left_contains_a && right_contains_b {
                                    // this measurement disputes this partitioning
                                    return true;
                                }

                                // no dispute
                                false
                            })
                            .count() as i32;

                        // if n_dispute is 0, then the probability should be 1
                        if rng.random_bool(base_probability.powi(n_dispute)) {
                            break;
                        } else {
                            slice_to_reorder.shuffle(rng);
                            (left, right) =
                                slice_to_reorder.split_at_mut(slice_to_reorder.len() / 2);
                        }

                        iteration += 1;
                        if iteration % 100 == 0 {
                            println!(
                                "iteration {iteration} trying to find an acceptable ordering; next attempt: {left:?} / {right:?}"
                            );
                        }
                    }

                    // Finally, we recurse on each of the two partitions.
                    recursion_helper(
                        left,
                        comparison_history,
                        comparisons_seen,
                        correctness_probability,
                        max_sorting_index,
                        rng,
                    );
                    recursion_helper(
                        right,
                        comparison_history,
                        comparisons_seen,
                        correctness_probability,
                        max_sorting_index,
                        rng,
                    );
                }

                recursion_helper(
                    range_to_resample,
                    comparison_history,
                    &self.comparisons_seen,
                    p,
                    max_sorting_index,
                    rng,
                );
            }

            SamplingStrategy::ByMaxElement {
                comparison_history,
                max_sorting_index,
            } => {
                // Cache the results of range_to_resample.contains(idx) for all indexes.
                let range_contains_map = {
                    let mut map = vec![false; max_sorting_index.0 + 1];
                    for sorting_idx in range_to_resample.iter() {
                        map[sorting_idx.0] = true;
                    }
                    map
                };

                // We sample the distribution by repeatedly finding the maximum element
                // out of all the unselected elements. So, for each element, we need to know
                // its n_dispute, the number of times it has been measured to be smaller than
                // some other element.
                let mut all_n_disputes = BTreeMap::<SortingIndex, i32>::new();
                for (matchup_index, comparison_history) in comparison_history {
                    let matchup = Matchup::from_index(*matchup_index, max_sorting_index);

                    if range_contains_map[matchup.a.0] && range_contains_map[matchup.b.0] {
                        for (_, comparison) in comparison_history {
                            if comparison.0.is_sign_positive() {
                                *all_n_disputes.entry(matchup.a).or_default() += 1;
                            } else {
                                *all_n_disputes.entry(matchup.b).or_default() += 1;
                            }
                        }
                    }
                }

                let base_beta = (1.0 - p) / p;
                let mut betas = range_to_resample
                    .iter()
                    .map(|sorting_idx| {
                        let n_dispute = *all_n_disputes.get(sorting_idx).unwrap_or(&0);
                        base_beta.powi(n_dispute)
                    })
                    .collect::<Vec<_>>(); // TODO: experiment with Vec vs LinkedList perf

                let mut beta_sum: f64 = betas.iter().rev().sum();

                while !range_to_resample.is_empty() {
                    // Note: f64's standard uniform distribution is [0, 1)
                    let threshold = rng.random::<f64>();

                    let mut total_beta_value = 0.0;
                    'find_next_max: for i in 0..betas.len() {
                        total_beta_value += betas[i] / beta_sum;
                        if total_beta_value >= threshold {
                            // We accept the new sample as the largest
                            let max_len = range_to_resample.len() - 1;

                            beta_sum -= betas[i];
                            range_to_resample.swap(i, max_len);
                            betas.swap(i, max_len);

                            (_, range_to_resample) = range_to_resample.split_last_mut().unwrap();
                            break 'find_next_max;
                        }
                    }
                }
            }
        }
    }

    fn agrees_with(&self, matchup: &Matchup, outcome: EvaluationOutcome) -> bool {
        for id in &self.ordering {
            if *id == matchup.a {
                return outcome.0.is_sign_positive();
            } else if *id == matchup.b {
                return outcome.0.is_sign_negative();
            }
        }

        panic!("incomplete candidate/sample list");
    }

    fn update(
        &mut self,
        matchup: &Matchup,
        outcome: EvaluationOutcome,
        history_index: usize,
        max_sorting_index: SortingIndex,
        comparison_history: &ComparisonHistory,
        rng: &mut dyn RngCore,
    ) {
        let matchup_index = matchup.to_index(max_sorting_index);

        self.comparisons_seen.insert((matchup_index, history_index));
        let agrees = self.agrees_with(matchup, outcome);
        if agrees {
            self.matching_comparisons_seen += 1;
        } else {
            // We are likely to reject this candidate, in which case part of it needs to be resampled.
            let p = self.get_correct_measurement_probability();
            if !rng.random_bool((1.0 - p) / p) {
                println!(
                    "Resampling candidate list: {matchup:?} / {:#?}",
                    self.ordering
                );
                self.resample_range(
                    matchup,
                    // TODO: select sampling strategy dynamically
                    SamplingStrategy::ByMaxElement {
                        comparison_history,
                        max_sorting_index,
                    },
                    rng,
                );
                println!("Done resampling: {:#?}", self.ordering);
            }
        }
    }

    // Invariant: Must be >= 0.5.
    fn get_correct_measurement_probability(&self) -> f64 {
        0.9
        // TODO: try to implement the correct logic from section 4 of the paper
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CharacterSortingData {
    #[serde(rename = "m")]
    metadata: Vec<CharacterMetadata>,

    #[serde(rename = "h")]
    comparison_history: ComparisonHistory,

    // These are the N candidate lists, initially generated by randomly sampling the distribution of
    // possible orderings. Since they are generated when we have no priors, the distribution is uniform
    // and these can be generated by shuffling the full list of characters.
    //
    // There should be on the order of 100 candidate lists, probably.
    #[serde(rename = "l")]
    candidate_lists: Vec<SampleCandidateList>,

    // This is a list of all of the values N_ab for each matchup. N_ab
    // represents the number of candidate lists in which a comes before b.
    // The vector is long - there are (max_sorting_index choose 2) elements -
    // but since we only store 16 bytes per entry, even if the max sorting id
    // is 250, this is still only ~600KiB of memory.
    // #[serde(skip)]
    #[serde(rename = "c")]
    cached_matchup_data: Vec<CachedMatchupValue>,

    // RNG used for monte carlo algorithms. Only exists to be mocked for UTs, so not serialized.
    #[serde(skip, default = "default_rng")]
    rng: Box<dyn DebugRng>,
}
impl CharacterSortingData {
    fn new(metadata: Vec<CharacterMetadata>, num_candidate_lists: usize) -> Self {
        Self::new_with_rng(metadata, num_candidate_lists, default_rng())
    }

    fn new_with_rng(
        metadata: Vec<CharacterMetadata>,
        num_candidate_lists: usize,
        mut rng: Box<dyn DebugRng>,
    ) -> Self {
        assert!(!metadata.is_empty());

        let candidate_lists = (0..num_candidate_lists)
            .map(|_| SampleCandidateList::new(&metadata, &mut rng))
            .collect::<Vec<_>>();

        let max_sorting_index = SortingIndex(metadata.len() - 1);
        let cached_matchup_data = Self::compute_matchup_cache(max_sorting_index, &candidate_lists);

        Self {
            metadata,
            comparison_history: BTreeMap::default(),
            candidate_lists,
            cached_matchup_data,
            rng,
        }
    }

    fn save(&self, path: &Option<PathBuf>) -> Result<()> {
        let Some(path) = path else {
            return Ok(());
        };

        serde_json::to_writer_pretty(BufWriter::new(File::create(path)?), self)?;

        let compressed_path = path.with_added_extension("gz");

        let mut encoder = GzEncoder::new(File::create(compressed_path)?, Compression::best());
        serde_json::to_writer(&mut encoder, self)?;
        encoder.finish()?;

        Ok(())
    }

    fn get_metadata(&self, sorting_id: SortingIndex) -> &CharacterMetadata {
        &self.metadata[sorting_id.0]
    }

    /// Determines if sorting is completed.
    ///
    /// Sorting is finished when enough of the candidate lists are identical. "Enough" is determined by the argument
    /// epsilon, which must be between 0.0 and 1.0; it specifies the fraction of candidate lists which are permitted
    /// to be different when sorting is finished. For example, with 100 candidate lists and an epsilon of 0.03, sorting
    /// is finished if 97/100 of the lists are identical. An epsilon of 0.0 requires *all* candidate lists to be
    /// identical, and an epsilon of 1.0 means that sorting is always considered finished.
    ///
    /// Returns either the final sort order (from the largest group of identical candidates) if sorting is finished,
    /// or None if the sort order is indeterminate.
    fn get_final_sort_order(&self, epsilon: f64) -> Option<Vec<SortingIndex>> {
        // If we just started, there will be N groups, each containing 1 element.
        // If we are nearly finished, there will be very few groups, with one containing nearly N elements.
        let unique_groups = self.get_unique_candidate_list_groups();

        let (largest_group_size, largest_group_rep) = unique_groups
            .iter()
            .map(|(rep, group)| (group.len(), rep))
            .max()
            .unwrap();

        let fraction_identical = largest_group_size as f64 / self.candidate_lists.len() as f64;
        if fraction_identical >= (1.0 - epsilon) {
            Some(self.candidate_lists[largest_group_rep.0].ordering.clone())
        } else {
            None
        }
    }

    /// Finds the groups of unique candidate lists. Each group is identified by a single representative candidate,
    /// which is mapped to a list of all of the candidate lists that are identical to it (including itself).
    fn get_unique_candidate_list_groups(&self) -> HashMap<CandidateListIndex, CandidateListGroup> {
        // Each group is identified by a representative candidate list's index.
        let mut unique_groups: HashMap<CandidateListIndex, CandidateListGroup> = Default::default();

        'candidate_lists: for (list_idx, candidate_list) in self.candidate_lists.iter().enumerate()
        {
            let list_idx = CandidateListIndex(list_idx);

            for (group_representative_idx, group_members) in unique_groups.iter_mut() {
                let ordering_a = candidate_list.ordering.as_slice();
                let ordering_b = self.candidate_lists[group_representative_idx.0]
                    .ordering
                    .as_slice();

                if ordering_a == ordering_b {
                    group_members.push(list_idx);
                    continue 'candidate_lists;
                }
            }

            // No existing group matched this candidate list, so add a new group with this list as the representative.
            unique_groups.insert(list_idx, smallvec![list_idx]);
        }

        unique_groups
    }

    fn compute_matchup_cache(
        max_sorting_index: SortingIndex,
        candidate_lists: &[SampleCandidateList],
    ) -> Vec<CachedMatchupValue> {
        let max_index = Self::get_max_matchup_index(max_sorting_index);

        (0..=max_index.0)
            .map(|index| {
                let matchup = Matchup::from_index(MatchupIndex(index), max_sorting_index);

                // TODO: This is a lot of repeated work across a lot of similar subproblems; can this be optimized?
                // At minimum it could be parallelized.
                let n_ab = Self::count_candidate_lists_in_which_a_before_b(
                    candidate_lists,
                    matchup.a,
                    matchup.b,
                );
                let convergence = Self::calculate_convergence(n_ab, candidate_lists.len());

                CachedMatchupValue { n_ab, convergence }
            })
            .collect()
    }

    fn get_max_matchup_index(max_sorting_index: SortingIndex) -> MatchupIndex {
        let n = max_sorting_index.0 + 1;
        let n_choose_2 = (n * (n - 1)) / 2;
        MatchupIndex(n_choose_2 - 1)
    }

    /// Convergence is defined as `(N_ab - N_ba)^2`, where `N_ab` is the number of candidate lists
    /// in which a occurs before b, and `N_ba` is the number of candidate lists in which b occurs before a.
    ///
    /// Convergence increases (approaches n^2, where n is the total number of candidate lists) when more lists
    /// agree that a occurs before b (N_ab => n) or agree that b occurs before a (N_ba => n).
    ///
    /// Convergence decreases (approaches zero) when lists are evenly split between contributing to N_ab and N_ba.
    fn calculate_convergence(n_ab: u64, num_candidate_lists: usize) -> NotNan<f64> {
        // N_ba = n - N_ab, where n is the number of candidate lists, so
        // we can just count the number of lists in which a occurs before b.
        // convergence = (N_ab - N_ba)^2
        // convergence = (N_ab - (n - N_ab))^2
        // convergence = (2 * N_ab - n)^2
        let sqrt_convergence = 2.0 * (n_ab as f64) - (num_candidate_lists as f64);
        NotNan::new(sqrt_convergence * sqrt_convergence).unwrap()
    }

    fn count_candidate_lists_in_which_a_before_b(
        candidate_lists: &[SampleCandidateList],
        a: SortingIndex,
        b: SortingIndex,
    ) -> u64 {
        candidate_lists
            .iter()
            .filter(|candidate_list| {
                for char_id in &candidate_list.ordering {
                    if *char_id == a {
                        return true;
                    } else if *char_id == b {
                        return false;
                    }
                }

                panic!("candidate list incomplete");
            })
            .count() as u64
    }

    /// Finds the matchups that are most valuable to ask the user about.
    ///
    /// The common behavior here is to return twice half_num_characters_requested
    /// characters, but note that this may return less if there are overlaps among
    /// the least converged matchups.
    ///
    /// There are no guarantees about the order of the returned character IDs.
    fn get_most_valuable_matchups(
        &mut self,
        half_num_characters_requested: usize,
    ) -> SmallVec<[SortingIndex; 4]> {
        // TODO: optimize using the algorithm in the appendix - only consider matchups between characters
        // that are adjacent in one or more candidate lists.

        let max_sorting_index = SortingIndex(self.metadata.len() - 1);

        // We have to mutate (sort) the matchup cache, so we have to copy it.
        // This is much cheaper than recalculating all the convergences though.
        let mut certainty_list: Vec<_> = self
            .cached_matchup_data
            .iter()
            .enumerate()
            .map(|(index, CachedMatchupValue { convergence, .. })| {
                (*convergence, MatchupIndex(index))
            })
            .collect();

        for candidate in &self.candidate_lists {
            println!("Candidate: {:?}", candidate.ordering);
        }
        for (convergence, index) in &certainty_list {
            let matchup = Matchup::from_index(*index, max_sorting_index);
            println!("Convergence: {matchup:?} = {convergence}");
        }

        // We don't need to fully sort the list, we just need to find the smallest N matchups in any order.
        let (unsorted_lower_selected_matchups, _, _) = certainty_list
            .select_nth_unstable_by_key(half_num_characters_requested, |(convergence, _)| {
                *convergence
            });

        let mut result = BTreeSet::new();

        for (_, selected_matchup_index) in unsorted_lower_selected_matchups {
            let selected_matchup = Matchup::from_index(*selected_matchup_index, max_sorting_index);
            result.insert(selected_matchup.a);
            result.insert(selected_matchup.b);
        }

        result.into_iter().collect()
    }

    // fn get_current_comparison_strength(
    //     &self,
    //     matchup: &Matchup,
    //     now: Option<Timestamp>,
    // ) -> EvaluationOutcome {
    //     let max_sorting_index = SortingIndex(self.metadata.len() - 1);
    //     let mut outcome = 0.0;

    //     for (timestamp, evaluation) in self
    //         .comparison_history
    //         .get(&matchup.to_index(max_sorting_index))
    //         .map(|x| x.iter())
    //         .into_iter()
    //         .flatten()
    //     {
    //         if let Some(now) = now {
    //             let seconds_older_than_one_week = (now - *timestamp)
    //                 .checked_sub((7.days(), SpanRelativeTo::days_are_24_hours()))
    //                 .unwrap()
    //                 .total((Unit::Second, SpanRelativeTo::days_are_24_hours()))
    //                 .unwrap()
    //                 .max(0.0);

    //             // Decay is calculated on a sigmoid curve, and in particular a logistic function:
    //             // decay(age) = L / (1 + e^(-k * (age - age_midpoint)))
    //             // L is the carrying capacity, the upper bound of the curve; in our case it is 1.0;
    //             // k affects the steepness of the curve;
    //             // and age_midpoint is the age at which we want the decay to equal 0.5L = 0.5.

    //             // We configure age_midpoint to be half a year (calculated using hours to avoid ambiguity
    //             // in the length of a day):
    //             let age_midpoint = (365 * 24 / 2).hours().total(Unit::Second).unwrap();

    //             // k is determined by just checking a bunch of values to find one that produces a nice-looking graph
    //             // on the scale of the number of seconds in a year.
    //             const K: f64 = 0.000_000_2;

    //             let logistic_denominator_exponent =
    //                 -K * (seconds_older_than_one_week - age_midpoint);
    //             let logistic_denominator = 1.0 + logistic_denominator_exponent.exp();
    //             let logistic_result = 1.0 / logistic_denominator;

    //             // The logistic function we calculated varies from 0 to 1, so we subtract it from 1 to get what we need.
    //             outcome += evaluation.0 * (1.0 - logistic_result);
    //         } else {
    //             outcome += evaluation.0;
    //         }
    //     }

    //     EvaluationOutcome(outcome)
    // }

    /// Records that a new measurement was performed by the user.
    fn record_new_measurement(&mut self, matchup: &Matchup, outcome: EvaluationOutcome) {
        let max_sorting_index = SortingIndex(self.metadata.len() - 1);
        let matchup_index = matchup.to_index(max_sorting_index);

        let entry = self.comparison_history.entry(matchup_index).or_default();
        entry.push((Timestamp::now(), outcome));
        let history_index = entry.len() - 1;

        for sample in self.candidate_lists.iter_mut() {
            sample.update(
                matchup,
                outcome,
                history_index,
                max_sorting_index,
                &self.comparison_history,
                &mut self.rng,
            );
        }

        self.cached_matchup_data =
            Self::compute_matchup_cache(max_sorting_index, &self.candidate_lists);
    }
}

#[derive(Debug, clap::Args)]
#[group(required = true, multiple = false)]
struct InputArgGroup {
    #[arg(short, long)]
    metadata_path: Option<PathBuf>,
    #[arg(short, long)]
    resume_data_path: Option<PathBuf>,
}

#[derive(Debug, clap::Parser)]
#[command(name = "smart_touhou_sorter")]
#[command(about = "Test harness")]
struct Cli {
    #[clap(flatten)]
    input: InputArgGroup,

    #[arg(short = 's')]
    save_path: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let mut sorting_data = if let Some(metadata_path) = args.input.metadata_path {
        let metadata: Vec<CharacterMetadata> = serde_json::from_reader(BufReader::new(
            File::open(&metadata_path).with_context(|| {
                format!(
                    "failed to open file {} for reading",
                    metadata_path.display()
                )
            })?,
        ))
        .with_context(|| {
            format!(
                "failed to read or parse metadata file {}",
                metadata_path.display()
            )
        })?;

        println!("Loaded {} characters from metadata file...", metadata.len());

        CharacterSortingData::new(metadata, 10) // TODO: configurable
    } else if let Some(resume_path) = args.input.resume_data_path {
        serde_json::from_reader(BufReader::new(File::open(&resume_path)?))?
    } else {
        panic!("no inputs");
    };

    let epsilon = 0.0; // TODO: configurable
    let result = loop {
        if let Some(result) = sorting_data.get_final_sort_order(epsilon) {
            break result;
        };

        let most_valuable_characters_to_compare = sorting_data.get_most_valuable_matchups(1);
        println!("Most valuable characters to compare:");
        for sorting_id in &most_valuable_characters_to_compare {
            println!(
                "\t{sorting_id:?}: {}",
                sorting_data.get_metadata(*sorting_id).display_name
            );
        }

        let selected_max = SortingIndex(loop {
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;

            match input.trim().parse() {
                Ok(input) => break input,
                Err(err) => println!("couldn't parse: {err}"),
            }
        });

        // assume we always have 2 characters to compare for now
        let matchup = Matchup::new(
            most_valuable_characters_to_compare[0],
            most_valuable_characters_to_compare[1],
        );
        let outcome = if selected_max == matchup.a {
            EvaluationOutcome(1.0)
        } else {
            EvaluationOutcome(-1.0)
        };

        sorting_data.record_new_measurement(&matchup, outcome);
        sorting_data.save(&args.save_path)?;
    };

    println!("\nResult: {result:#?}");

    Ok(())
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::{
        CachedMatchupValue, CharacterMetadata, CharacterSortingData, DebugRng, Matchup,
        MatchupIndex, SortingIndex,
    };

    const TEST_CHARACTER_LENGTH: usize = 250;

    static TEST_CHARACTERS: LazyLock<Vec<CharacterMetadata>> = LazyLock::new(|| {
        (0..TEST_CHARACTER_LENGTH)
            .map(|_| CharacterMetadata::default())
            .collect()
    });

    fn get_fixed_seed_rng() -> Box<dyn DebugRng> {
        Box::new(ChaCha8Rng::from_seed([4; 32]))
    }

    #[test]
    fn construct_test() {
        let mut sorting_data =
            CharacterSortingData::new_with_rng(TEST_CHARACTERS.clone(), 200, get_fixed_seed_rng());
        assert_eq!(sorting_data.get_unique_candidate_list_groups().len(), 200);
        assert!(sorting_data.get_final_sort_order(0.0).is_none());
        assert_eq!(
            sorting_data.get_final_sort_order(1.0).unwrap().len(),
            TEST_CHARACTER_LENGTH
        );

        // dbg!(sorting_data.get_most_valuable_matchups(4));
        // dbg!(sorting_data.get_most_valuable_matchups(5));
        // dbg!(sorting_data.get_most_valuable_matchups(6));

        // dbg!(sorting_data.cached_matchup_data.len() * std::mem::size_of::<CachedMatchupValue>());

        // assert!(false);
    }

    #[test]
    fn encode_matchup_test() {
        // 31375 = 251 choose 2
        let mut vec = Vec::with_capacity(31375);
        for i in 0..=250 {
            for j in (i + 1)..=250 {
                vec.push(
                    Matchup::new(SortingIndex(i), SortingIndex(j)).to_index(SortingIndex(250)),
                );
            }
        }

        assert_eq!(vec.len(), 31375);
        assert_eq!(vec, (0..31375).map(MatchupIndex).collect::<Vec<_>>());

        let mut i = 0;
        let mut j = 1;
        for k in 0..31_375 {
            assert_eq!(
                Matchup::from_index(MatchupIndex(k), SortingIndex(250)),
                Matchup {
                    a: SortingIndex(i),
                    b: SortingIndex(j)
                }
            );

            j += 1;
            if j > 250 {
                i += 1;
                j = i + 1;
            }
        }

        // The last valid matchup was (249, 250), but we increment at the end of the loop
        assert_eq!(i, 250);
        assert_eq!(j, 251);
    }
}
