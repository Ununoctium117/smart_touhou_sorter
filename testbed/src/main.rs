use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use clap::Parser as _;
use engine::{
    CharacterMetadata, CharacterSortingData, EvaluationOutcome, Matchup, SamplingStrategy,
};
use flate2::{Compression, write::GzEncoder};
use rand::{Rng as _, rng};
use serde::Serialize;

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

    #[arg(long = "bench")]
    bench: bool,

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

    if args.bench {
        #[derive(Serialize)]
        struct BenchmarkOutput {
            num_sorting_elements: usize,
            by_max_element: HashMap<usize, Vec<Duration>>,
            recursion: HashMap<usize, Vec<Duration>>,
        }

        let metadata = sorting_data.copy_all_metadata();

        // Goal: With various numbers of random measurements, try ~250 times to see how long it takes to
        // resample the list.
        let mut output = BenchmarkOutput {
            num_sorting_elements: metadata.len(),
            by_max_element: (0..1_000).map(|i| (i, Vec::with_capacity(250))).collect(),
            recursion: (0..1_000).map(|i| (i, Vec::with_capacity(250))).collect(),
        };
        for measurement_count in (0..500).step_by(50) {
            println!("m: {measurement_count}");
            let mut sorting_data = sorting_data.clone();

            // apply the right number of random measurements first
            let mut last_to_sort = None;
            for _ in 0..measurement_count {
                let to_sort = sorting_data.get_most_valuable_matchups(1, last_to_sort.clone());

                let matchup = Matchup::new(to_sort[0], to_sort[1]);
                let outcome = EvaluationOutcome(if rng().random_bool(0.5) { 1.0 } else { -1.0 });

                sorting_data.record_new_measurement(
                    &matchup,
                    outcome,
                    SamplingStrategy::ByMaxElement,
                );

                last_to_sort = Some(to_sort);
            }

            // now measure 250 resamplings with each strategy
            for sample in 0..250 {
                if sample % 25 == 1 {
                    println!("sample {sample}");
                }
                let mut sorting_data_recursion = sorting_data.clone();
                let mut sorting_data_max_element = sorting_data.clone();

                let to_sort = sorting_data.get_most_valuable_matchups(1, None);
                let matchup = Matchup::new(to_sort[0], to_sort[1]);
                let outcome = EvaluationOutcome(1.0);

                // recursion
                {
                    let start_time = Instant::now();

                    sorting_data_recursion.record_new_measurement(
                        &matchup,
                        outcome,
                        SamplingStrategy::Recursion,
                    );

                    let duration = Instant::now() - start_time;
                    output
                        .recursion
                        .get_mut(&measurement_count)
                        .unwrap()
                        .push(duration);
                }

                // by max element
                {
                    let start_time = Instant::now();

                    sorting_data_max_element.record_new_measurement(
                        &matchup,
                        outcome,
                        SamplingStrategy::ByMaxElement,
                    );

                    let duration = Instant::now() - start_time;
                    output
                        .recursion
                        .get_mut(&measurement_count)
                        .unwrap()
                        .push(duration);
                }
            }
        }

        serde_json::to_writer_pretty(
            BufWriter::new(File::create("bench_output.json").unwrap()),
            &output,
        )
        .unwrap();
    }

    let epsilon = 0.0; // TODO: configurable
    let result = loop {
        if let Some(result) = sorting_data.get_final_sort_order(epsilon) {
            break result;
        };

        let most_valuable_characters_to_compare = sorting_data.get_most_valuable_matchups(1, None);
        let selectors = ('a'..='z')
            .zip(most_valuable_characters_to_compare.iter())
            .collect::<HashMap<_, _>>();

        println!("Most valuable characters to compare:");
        for (selector, sorting_id) in selectors.iter() {
            println!(
                "\t{selector}: {}",
                sorting_data.get_metadata(**sorting_id).display_name
            );
        }

        let selected_max = loop {
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let Some(first_char) = input.trim().chars().next() else {
                println!("empty input");
                continue;
            };

            let Some(sorting_index) = selectors.get(&first_char) else {
                println!("not a selector: {first_char}");
                continue;
            };

            break sorting_index;
        };

        // assume we always have 2 characters to compare for now
        let matchup = Matchup::new(
            most_valuable_characters_to_compare[0],
            most_valuable_characters_to_compare[1],
        );
        let outcome = if **selected_max == matchup.a {
            EvaluationOutcome(1.0)
        } else {
            EvaluationOutcome(-1.0)
        };

        sorting_data.record_new_measurement(&matchup, outcome, SamplingStrategy::ByMaxElement);

        if let Some(ref save_path) = args.save_path {
            serde_json::to_writer_pretty(
                BufWriter::new(File::create(save_path).unwrap()),
                &sorting_data,
            )
            .unwrap();

            serde_json::to_writer(
                GzEncoder::new(
                    BufWriter::new(File::create(save_path.with_added_extension("gz")).unwrap()),
                    Compression::best(),
                ),
                &sorting_data,
            )
            .unwrap();
        }
    };

    println!("\nResult: {result:#?}");

    Ok(())
}
