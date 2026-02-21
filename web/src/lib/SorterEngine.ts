import {
    create_sorter,
    estimate_progress,
    get_best_comparisons,
    load_sorter,
    serialize_sorter_compressed,
} from "web-engine";

// TODO: Use a webworker to delegate background tasks so the UI can remain responsive... somehow

export type Manifest = {
    g: string;
    d: string;
    u: string;
    t: string[];
}[];

export type KeyedManifest = { [k: string]: number };

// conveniently wraps the WASM functions
export class SorterEngine {
    private readonly handle: number;

    private constructor(handle: number) {
        this.handle = handle;
    }

    public EstimateProgress(): number {
        return estimate_progress(this.handle);
    }

    public Save(saveKey: string): void {
        const serialized = serialize_sorter_compressed(this.handle);
        console.log(
            `Saving sorter state to ${saveKey}: ${serialized.length} bytes`,
        );
        localStorage.setItem(saveKey, serialized);
    }

    public GetBestComparisons(half_num: number, exclusions: string[]): string[] { 
        return get_best_comparisons(this.handle, half_num, exclusions);
    }

    // =====================
    // Instantiation methods
    // =====================

    public static Create(
        manifest: Manifest,
        numCandidateLists: number,
    ): SorterEngine {
        const ids = manifest.map((manifestEntry) => manifestEntry.g);
        return new SorterEngine(create_sorter(ids, numCandidateLists));
    }

    public static Load(saveKey: string): SorterEngine {
        const saveData = localStorage.getItem(saveKey);
        if (saveData === null) {
            throw "no such saveKey";
        }

        return new SorterEngine(load_sorter(saveData));
    }
}