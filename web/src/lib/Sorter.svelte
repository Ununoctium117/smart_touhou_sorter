<script lang="ts">
    import type { KeyedManifest, Manifest, SorterEngine } from "./SorterEngine";
    import {
        DndContext,
        type DragEndEvent,
    } from "@dnd-kit-svelte/core";
    import Droppable from "./DnD/Droppable.svelte";
    import { sensors } from "./DnD/DndUtils";
    import DraggableCharacter from "./DnD/DraggableCharacter.svelte";

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

    const containers = [0, 1, 2, 3];
    // Intentional: We're only using this to set the initial value.
    // svelte-ignore state_referenced_locally
    let containersForEachCharacter: (number | null)[] = $state(
        currentBestComparison.map(() => null),
    );

    // These will be used to prevent seeing the same object repeatedly
    // in the comparison
    let lastComparedCharacters = $state([]);

    // Bindings for window width and height, used to determine if we're in vertical/horizontal
    // sorting mode.
    let windowWidth = $state(0);
    let windowHeight = $state(0);
    let verticalSorting = $derived(windowHeight > windowWidth);

    function submitComparison() {
        // TODO actually submit
        // TODO save after submitting and name it something better
        sorterEngine.Save("asdf");

        percentComplete = sorterEngine.EstimateProgress();
        currentBestComparison = sorterEngine.GetBestComparisons(2, []);
        containersForEachCharacter = currentBestComparison.map(() => null);
    }

    function onDragEnd(evt: DragEndEvent) {
        // TODO
    }
</script>

<svelte:window bind:innerWidth={windowWidth} bind:innerHeight={windowHeight} />

{#snippet characterRender(characterId: string)}
    <DraggableCharacter
        id={characterId}
        characterDisplayName={manifest[keyedManifest[characterId]].d}
        characterImageUrl={manifest[keyedManifest[characterId]].u}
    ></DraggableCharacter>
{/snippet}

<h3>Sorting: {manifestName}</h3>

<DndContext {sensors} {onDragEnd}>
    <div id="interactableSorterArea">
        {#each containers as container}
            <Droppable id={container}>
                {#each currentBestComparison as characterId, index}
                    {#if containersForEachCharacter[index] === container}
                        {@render characterRender(characterId)}
                    {/if}
                {/each}
            </Droppable>
        {/each}
    </div>
    <div>
        {#each currentBestComparison as characterId, index}
            {#if containersForEachCharacter[index] === null}
                {@render characterRender(characterId)}
            {/if}
        {/each}
    </div>
</DndContext>

<button onclick={submitComparison}> Submit Current Comparison </button>

<h4>Candidate list convergence: {(percentComplete * 100).toFixed(2)}%</h4>

<style lang="scss">
    #interactableSorterArea {
        display: flex;
        justify-content: space-evenly;
        border: 1px white;

        @media (orientation: landscape) {
            min-width: 80vw;
            height: 50%;
            flex-direction: row;
        }

        @media (orientation: portrait) {
            width: 50%;
            min-height: 80vh;
            flex-direction: column;
        }
    }
</style>
