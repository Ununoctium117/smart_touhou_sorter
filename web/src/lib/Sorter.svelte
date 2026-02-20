<script lang="ts">
    import type { SorterEngine } from "./SorterEngine";

    interface Props {
        engine: SorterEngine;
    }
    let { engine }: Props = $props();

    let getPercentComplete = $state(() => engine.EstimateProgress());

    function submitComparison() {
        engine.Save('asdf');
        // TODO
    }
</script>

<h1>TEMP</h1>

<button onclick={submitComparison}> Submit Current Comparison </button>

<h4>Candidate list convergence: {(getPercentComplete() * 100).toFixed(2)}%</h4>
