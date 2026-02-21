<script lang="ts">
    import type { KeyedManifest, Manifest, SorterEngine } from "./SorterEngine";

    interface Props {
        sorterEngine: SorterEngine;
        manifestName: string;
        manifest: Manifest;
        keyedManifest: KeyedManifest;
    }
    let { sorterEngine, manifestName, manifest, keyedManifest }: Props = $props();

    // This is intentional: We're using this to set the initial value only.
    // svelte-ignore state_referenced_locally
    let percentComplete = $state(sorterEngine.EstimateProgress());

    // This is intentional: We're using this to set the initial value only.
    // svelte-ignore state_referenced_locally
    let currentBestComparison: string[] = $state(
        sorterEngine.GetBestComparisons(2, []),
    );

    function submitComparison() {
        sorterEngine.Save("asdf");
        // TODO

        percentComplete = sorterEngine.EstimateProgress();
        currentBestComparison = sorterEngine.GetBestComparisons(2, []);
    }
</script>

<h3>Sorting: {manifestName}</h3>

<ul>
{#each currentBestComparison as comparisonCharId}
    <li><h4>{manifest[keyedManifest[comparisonCharId]].d}</h4></li>
{/each}
</ul>

<button onclick={submitComparison}> Submit Current Comparison </button>

<h4>Candidate list convergence: {(percentComplete * 100).toFixed(2)}%</h4>
