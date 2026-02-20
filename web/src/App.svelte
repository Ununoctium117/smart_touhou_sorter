<script lang="ts">
  import Sorter from "./lib/Sorter.svelte";
  import { SorterEngine, type Manifest } from "./lib/SorterEngine";
  import manifests from "./assets/manifests.json";

  let sorterEngine: Promise<SorterEngine> = $state(
    (async () => {
      // TODO: Allow user to pick a manifest
      const manifestUrl = manifests[0].url;

      // TODO: Allow changing manifest, configuring number of candidate lists
      const manifest: Manifest = await (await fetch(manifestUrl)).json();
      return SorterEngine.Create(manifest, 100);
    })(),
  );
</script>

<main>
  <h1>Vite + Svelte</h1>

  <div class="card">
    {#await sorterEngine}
      <!-- pending... -->
       <h4>Loading...</h4>
    {:then sorterEngine} 
      <Sorter engine={sorterEngine} />
    {/await}
  </div>
</main>

<style>
</style>
