import {
    create_sorter,
    estimate_progress,
    serialize_sorter_compressed,
    load_sorter,
} from "web-engine";

export type Manifest = {
    g: string;
    d: string;
    u: string;
    t: string[];
}[];

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