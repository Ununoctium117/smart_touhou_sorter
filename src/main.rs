#![allow(unused)]

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
};

use itertools::Itertools as _;
use jiff::{SpanCompare, SpanTotal, Timestamp, ToSpan as _, Unit};
use ordered_float::NotNan;
use rand::{RngCore, seq::SliceRandom as _};
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

/// Character IDs are numeric and must start at 0 with no gaps, due to the way matchups are encoded
/// into indices.
#[derive(Debug, Serialize, Deserialize, Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[serde(transparent)]
struct SortingId(u64);

#[derive(Debug, Serialize, Deserialize, Hash, PartialEq, Eq, PartialOrd, Ord, Clone)]
struct Matchup {
    // invariant: a < b
    a: SortingId,
    b: SortingId,
}
impl Matchup {
    fn new(a: SortingId, b: SortingId) -> Self {
        assert_ne!(a, b);

        Self {
            a: a.min(b),
            b: a.max(b),
        }
    }

    // Matchups are ordered as follows (in this example, the max character ID is 4):
    // (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)

    fn to_index(&self, max_sorting_id: SortingId) -> MatchupIndex {
        let a = self.a.0;
        let b = self.b.0;
        let n = max_sorting_id.0;

        assert!(a <= n);
        assert!(b <= n);

        // To encode into an index, we note that there are n pairs that start with 0,
        // n-1 pairs that start with 1, n-2 pairs that start with 2, etc. So the number of
        // combinations that come before any pair starting with a is:

        // a * (a - 1) is *always* even, so the division by 2 can be done with integer logic.
        let block_starting_index = Self::find_block_starting_index(a, n);
        let block_offset = b - a - 1;

        let index = block_starting_index + block_offset;
        MatchupIndex(index)
    }

    fn from_index(index: MatchupIndex, max_sorting_id: SortingId) -> Self {
        let index = index.0;
        let n = max_sorting_id.0;

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
            a_candidate - decrement_needed as u64
        };

        let b = (index - Self::find_block_starting_index(a, n)) + a + 1;

        Self {
            a: SortingId(a),
            b: SortingId(b),
        }
    }

    /// Used for index encoding/decoding. Finds the index of the block of matchups whose smaller element is a,
    /// where the max sorting id is n.
    fn find_block_starting_index(a: u64, n: u64) -> u64 {
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
struct MatchupIndex(u64);

/// A positive number means that, for a given matchup, a < b.
/// A negative number means that b < a.
/// The magnitude of the number represents the "strength" of the decision,
/// which affects the probability of the evaluation being correct. An outcome
/// of 2.0 means that the evaluation is twice as likely to be correct as if the
/// outcome were 1.0.
#[derive(Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
#[serde(transparent)]
struct EvaluationOutcome(f64);

// A unique identifier for a candidate list.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
struct CandidateListIndex(usize);

// A group of identical candidate lists.
type CandidateListGroup = SmallVec<[CandidateListIndex; 2]>;

trait DebugRng: RngCore + std::fmt::Debug {}
impl<T> DebugRng for T where T: RngCore + std::fmt::Debug {}
fn default_rng() -> Box<dyn DebugRng> {
    Box::new(rand::rng())
}

#[derive(Debug)]
struct CachedMatchupValue {
    n_ab: u64,
    convergence: NotNan<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CharacterSortingData {
    #[serde(rename = "m")]
    metadata: BTreeMap<SortingId, CharacterMetadata>,

    #[serde(rename = "h")]
    // This could be a Vec indexed by the MatchupIndex instead, but that would be worse for
    // the initial state and for serialization; it's unlikely that the user ever provides data
    // for ALL matchups. We use the BTreeMap here as essentially a sparse Vec.
    comparison_history: BTreeMap<MatchupIndex, BTreeMap<Timestamp, EvaluationOutcome>>,

    // These are the N candidate lists, initially generated by randomly sampling the distribution of
    // possible orderings. Since they are generated when we have no priors, the distribution is uniform
    // and these can be generated by shuffling the full list of characters.
    // There should be on the order of 100 candidate lists, probably.
    // Invariant: Each candidate list contains every CharacterId in the metadata, and no others. This
    // implies that each candidate list has the same length which is equal to metadata.len().
    // TODO: explore alternate representations? We need to make a lot of comparisons of items' positions in each list
    #[serde(rename = "l")]
    candidate_lists: Vec<Vec<SortingId>>,

    // This is a list of all of the values N_ab for each matchup. N_ab
    // represents the number of candidate lists in which a comes before b.
    // The vector is long - there are (max_sorting_id choose 2) elements -
    // but since we only store 16 bytes per entry, even if the max sorting id
    // is 250, this is still only ~600KiB of memory.
    #[serde(skip)]
    cached_matchup_data: Option<Vec<CachedMatchupValue>>,

    // RNG used for monte carlo algorithms. Only exists to be mocked for UTs, so not serialized.
    #[serde(skip, default = "default_rng")]
    rng: Box<dyn DebugRng>,
}
impl CharacterSortingData {
    fn new(metadata: BTreeMap<SortingId, CharacterMetadata>, num_candidate_lists: usize) -> Self {
        Self::new_with_rng(metadata, num_candidate_lists, default_rng())
    }

    fn new_with_rng(
        metadata: BTreeMap<SortingId, CharacterMetadata>,
        num_candidate_lists: usize,
        mut rng: Box<dyn DebugRng>,
    ) -> Self {
        assert!(!metadata.is_empty());

        let mut candidate_lists: Vec<Vec<SortingId>> =
            vec![metadata.keys().copied().collect(); num_candidate_lists];

        for list in candidate_lists.iter_mut() {
            list.shuffle(&mut rng);
        }

        Self {
            metadata,
            comparison_history: BTreeMap::default(),
            candidate_lists,
            cached_matchup_data: None,
            rng,
        }
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
    fn get_final_sort_order(&self, epsilon: f64) -> Option<Vec<SortingId>> {
        // If we just started, there will be N groups, each containing 1 element.
        // If we are nearly finished, there will be very few groups, with one containing nearly N elements.
        let unique_groups = self.get_unique_candidate_list_groups();

        let (largest_group_size, largest_group_rep) = unique_groups
            .iter()
            .map(|(rep, group)| (group.len(), rep))
            .max()
            .unwrap();

        let fraction_identical = largest_group_size as f64 / self.candidate_lists.len() as f64;
        if fraction_identical > (1.0 - epsilon) {
            Some(self.candidate_lists[largest_group_rep.0].clone())
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
                if candidate_list == self.candidate_lists[group_representative_idx.0].as_slice() {
                    group_members.push(list_idx);
                    continue 'candidate_lists;
                }
            }

            // No existing group matched this candidate list, so add a new group with this list as the representative.
            unique_groups.insert(list_idx, smallvec![list_idx]);
        }

        unique_groups
    }

    /// Determines the convergence among all candidate lists for a specific character matchup.
    ///
    /// This method is O(n * c), where n is the total number of candidate lists and c is the character count.
    fn get_matchup_convergence(&mut self, matchup: &Matchup) -> NotNan<f64> {
        let index = matchup.to_index(self.get_max_sorting_id());
        self.get_matchup_cache()[index.0 as usize].convergence
    }

    fn get_matchup_cache(&mut self) -> &mut [CachedMatchupValue] {
        let (max_sorting_id, _) = self.metadata.last_key_value().unwrap();
        let candidate_lists = self.candidate_lists.as_slice();

        self.cached_matchup_data
            .get_or_insert_with(|| {
                let max_index = Self::get_max_matchup_index(*max_sorting_id);

                (0..=max_index.0)
                    .map(|index| {
                        let matchup = Matchup::from_index(MatchupIndex(index), *max_sorting_id);

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
            })
            .as_mut_slice()
    }

    fn get_max_sorting_id(&self) -> SortingId {
        let (max_sorting_id, _) = self.metadata.last_key_value().unwrap();
        *max_sorting_id
    }

    fn get_max_matchup_index(max_sorting_id: SortingId) -> MatchupIndex {
        MatchupIndex(
            max_sorting_id
                .0
                .checked_mul(max_sorting_id.0.checked_sub(1).unwrap())
                .unwrap()
                / 2,
        )
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
        candidate_lists: &[Vec<SortingId>],
        a: SortingId,
        b: SortingId,
    ) -> u64 {
        candidate_lists
            .iter()
            .filter(|candidate_list| {
                for char_id in *candidate_list {
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
    ) -> SmallVec<[SortingId; 4]> {
        let max_sorting_id = self.get_max_sorting_id();

        // We have to mutate (sort) the matchup cache, so we have to copy it.
        // This is much cheaper than recalculating all the convergences though.
        let mut certainty_list: Vec<_> = self
            .get_matchup_cache()
            .iter()
            .enumerate()
            .map(|(index, CachedMatchupValue { convergence, .. })| {
                let matchup = Matchup::from_index(MatchupIndex(index as u64), max_sorting_id);
                (*convergence, matchup)
            })
            .collect();

        // We don't need to fully sort the list, we just need to find the smallest N matchups in any order.
        let (unsorted_lower_selected_matchups, _, _) = certainty_list
            .select_nth_unstable_by_key(half_num_characters_requested, |(convergence, _)| {
                *convergence
            });

        let mut result = BTreeSet::new();

        for (_, selected_matchup) in unsorted_lower_selected_matchups {
            result.insert(selected_matchup.a);
            result.insert(selected_matchup.b);
        }

        result.into_iter().collect()
    }

    fn get_current_comparison_strength(
        &self,
        matchup: &Matchup,
        now: Timestamp,
        consider_age: bool,
    ) -> EvaluationOutcome {
        let max_sorting_id = self.get_max_sorting_id();
        let mut outcome = 0.0;

        for (timestamp, evaluation) in self
            .comparison_history
            .get(&matchup.to_index(max_sorting_id))
            .map(|x| x.iter())
            .into_iter()
            .flatten()
        {
            let decay_factor = if consider_age {
                let timestamp_diff = now - *timestamp;
                let older_than_one_week = matches!(
                    timestamp_diff
                        .compare(SpanCompare::from(7.days()).days_are_24_hours())
                        .unwrap(),
                    Ordering::Greater
                );
                let seconds_older_than_one_week = if older_than_one_week {
                    timestamp_diff
                        .checked_sub(7.days())
                        .unwrap()
                        .total(SpanTotal::from(Unit::Second).days_are_24_hours())
                        .unwrap()
                } else {
                    0.0
                };

                // Decay is calculated on a sigmoid curve, and in particular a logistic function:
                // decay(age) = L / (1 + e^(-k * (age - age_midpoint)))
                // L is the carrying capacity, the upper bound of the curve; in our case it is 1.0;
                // k affects the steepness of the curve;
                // and age_midpoint is the age at which we want the decay to equal 0.5L = 0.5.

                // We configure age_midpoint to be half a year (calculated using hours to avoid ambiguity
                // in the length of a day):
                let age_midpoint = (365 * 24 / 2).hours().total(Unit::Second).unwrap();

                // k is determined by just checking a bunch of values to find one that produces a nice-looking graph
                // on the scale of the number of seconds in a year.
                const K: f64 = 0.000_000_2;

                let logistic_denominator_exponent =
                    -K * (seconds_older_than_one_week - age_midpoint);
                let logistic_denominator = 1.0 + logistic_denominator_exponent.exp();
                let logistic_result = 1.0 / logistic_denominator;

                // The logistic function we calculated varies from 0 to 1, so we subtract it from 1 to get what we need.
                1.0 - logistic_result
            } else {
                1.0
            };

            outcome += evaluation.0 * decay_factor;
        }

        EvaluationOutcome(outcome)
    }
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod test {
    use std::{collections::BTreeMap, fs::File, sync::LazyLock};

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::{
        CachedMatchupValue, CharacterMetadata, CharacterSortingData, DebugRng, Matchup,
        MatchupIndex, SortingId,
    };

    const TEST_CHARACTER_LENGTH: usize = 250;

    static TEST_CHARACTERS: LazyLock<BTreeMap<SortingId, CharacterMetadata>> =
        LazyLock::new(|| {
            let mut result = BTreeMap::default();

            for id in 0..TEST_CHARACTER_LENGTH {
                result.insert(SortingId(id as u64), CharacterMetadata::default());
            }

            result
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

        dbg!(sorting_data.get_most_valuable_matchups(4));
        dbg!(sorting_data.get_most_valuable_matchups(5));
        dbg!(sorting_data.get_most_valuable_matchups(6));

        dbg!(
            sorting_data.cached_matchup_data.unwrap().len()
                * std::mem::size_of::<CachedMatchupValue>()
        );

        assert!(false);
    }

    #[test]
    fn encode_matchup_test() {
        // 31375 = 251 choose 2
        let mut vec = Vec::with_capacity(31375);
        for i in 0..=250 {
            for j in (i + 1)..=250 {
                vec.push(Matchup::new(SortingId(i), SortingId(j)).to_index(SortingId(250)));
            }
        }

        assert_eq!(vec.len(), 31375);
        assert_eq!(vec, (0..31375).map(MatchupIndex).collect::<Vec<_>>());

        let mut i = 0;
        let mut j = 1;
        for k in 0..31_375 {
            assert_eq!(
                Matchup::from_index(MatchupIndex(k), SortingId(250)),
                Matchup {
                    a: SortingId(i),
                    b: SortingId(j)
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
