use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
};

use anyhow::{Context, Result};
use clap::Parser as _;
use engine::{CharacterMetadata, CharacterSortingData, EvaluationOutcome, Matchup};
use flate2::{Compression, write::GzEncoder};

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

        // let count_to_download = metadata
        //     .iter()
        //     .filter(|c| c.image.starts_with("https://"))
        //     .count();

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

        sorting_data.record_new_measurement(&matchup, outcome);

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