<script lang="ts">
    import type { KeyedManifest, Manifest, SorterEngine } from "./SorterEngine";
    import { draggable, Compartment, axis } from "@neodrag/svelte";

    interface Props {
        sorterEngine: SorterEngine;
        manifestName: string;
        manifest: Manifest;
        keyedManifest: KeyedManifest;
    }
    let { sorterEngine, manifestName, manifest, keyedManifest }: Props =
        $props();

    // This is intentional: We're using this to set the initial value only.
    // svelte-ignore state_referenced_locally
    let percentComplete = $state(sorterEngine.EstimateProgress());

    // This is intentional: We're using this to set the initial value only.
    // svelte-ignore state_referenced_locally
    let currentBestComparison: string[] = $state(
        sorterEngine.GetBestComparisons(2, []),
    );

    // These will be used to prevent seeing the same object repeatedly
    // in the comparison
    let lastComparedCharacters = $state([]);

    // Bindings for window width and height, used to determine if we're in vertical/horizontal
    // sorting mode.
    let windowWidth = $state(0);
    let windowHeight = $state(0);

    // Determines if we're in vertical sorting mode (for mobile devices).
    let verticalSorting = $derived(windowHeight > windowWidth);
    const draggableAxis = Compartment.of(() => axis(verticalSorting ? 'y' : 'x'));

    function submitComparison() {
        // TODO actually submit
        // TODO save after submitting and name it something better
        sorterEngine.Save("asdf");

        percentComplete = sorterEngine.EstimateProgress();
        currentBestComparison = sorterEngine.GetBestComparisons(2, []);
    }
</script>

<svelte:window bind:innerWidth={windowWidth} bind:innerHeight={windowHeight} />

<h3>Sorting: {manifestName}</h3>

<div id="interactableSorterArea">
    {#each currentBestComparison as comparisonCharId}
        <div {@attach draggable(() => [draggableAxis])}>
            <h4>{manifest[keyedManifest[comparisonCharId]].d}</h4>
        </div>
    {/each}
</div>

<button onclick={submitComparison}> Submit Current Comparison </button>

<h4>Candidate list convergence: {(percentComplete * 100).toFixed(2)}%</h4>
