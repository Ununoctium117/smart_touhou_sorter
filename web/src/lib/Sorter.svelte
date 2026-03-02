<script lang="ts">
    import type { KeyedManifest, Manifest, SorterEngine } from "./SorterEngine";
    import { DragDropProvider } from "@dnd-kit-svelte/svelte";
    import { RestrictToWindowEdges } from "@dnd-kit-svelte/svelte/modifiers";
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

    const containers = [0, 1, 2, 3, 4];
    const containerLabels = [
        "Least Liked 😒",
        null,
        null,
        null,
        "Most Liked 💕",
    ];
    // Intentional: We're only using this to set the initial value.
    // svelte-ignore state_referenced_locally
    let containerForEachCharacter: (number | null)[] = $state(
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
        containerForEachCharacter = currentBestComparison.map(() => null);
    }
</script>

<svelte:window bind:innerWidth={windowWidth} bind:innerHeight={windowHeight} />

<h3>Sorting: {manifestName}</h3>

<div id="hintPanel">
    <h4>Drag characters to arrange them from left to right.</h4>
</div>

<DragDropProvider
    {sensors}
    modifiers={[RestrictToWindowEdges]}
    onDragEnd={(event) => {
        if (event.canceled) {
            return;
        }
        const sourceId = event.operation.source?.id;
        const sourceIndex = currentBestComparison.findIndex(
            (charId) => charId == sourceId,
        );
        const targetContainer = event.operation.target?.id;

        if (typeof targetContainer === "number") {
            containerForEachCharacter[sourceIndex] = targetContainer;
        } else {
            containerForEachCharacter[sourceIndex] = null;
        }
    }}
>
    <div id="interactableSorterArea">
        {#each containers as container, index}
            <div
                class={verticalSorting
                    ? "sorterTargetVertical"
                    : "sorterTargetHorizontal"}
            >
                <div class="dropTargetLabel">
                    {containerLabels[index] || "\xA0"}
                </div>
                <Droppable id={container} class="sorterDropTargetInner">
                    {#each currentBestComparison as characterId, index}
                        {#if containerForEachCharacter[index] === container}
                            {@render characterRender(characterId)}
                        {/if}
                    {:else}
                        <span>Drop Here</span>
                    {/each}
                </Droppable>
            </div>
        {/each}
    </div>
    <div id="characterDefaultPosition">
        {#each currentBestComparison as characterId, index}
            {#if containerForEachCharacter[index] === null}
                {@render characterRender(characterId)}
            {/if}
        {/each}

        {#if containerForEachCharacter.findIndex((e) => e === null) === -1}
            <span class="submitHint"
                >Submit comparison to load more characters!</span
            >
        {/if}
    </div>
</DragDropProvider>

{#snippet characterRender(characterId: string)}
    <DraggableCharacter
        id={characterId}
        characterDisplayName={manifest[keyedManifest[characterId]].d}
        characterImageUrl={manifest[keyedManifest[characterId]].u}
    ></DraggableCharacter>
{/snippet}

<button id="submitButton" onclick={submitComparison}>
    Submit Current Comparison
</button>

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

        :global(.sorterDropTargetInner) {
            border: 2px solid gray;
            min-width: 15vw;
            min-height: 15vh;
            flex: 1;
            margin: 15px;
            padding: 15px;

            display: flex;
            flex-direction: column;
        }

        .dropTargetLabel {
            font-weight: lighter;
            font-size: smaller;
        }
    }

    .sorterTargetVertical {
        display: flex;
        flex-direction: row;
    }

    .sorterTargetHorizontal {
        display: flex;
        flex-direction: column;
    }

    #characterDefaultPosition {
        flex: 1;
        align-self: stretch;

        display: flex;
        justify-content: space-evenly;
        border: 1px solid white;

        margin: 15px;
    }

    #hintPanel {
        text-align: left;
        align-self: flex-start;
        margin: 15px;
    }

    .submitHint {
        font-size: smaller;
        font-weight: lighter;
        margin: 15px;
    }
</style>
