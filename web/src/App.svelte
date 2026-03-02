<script lang="ts">
  import Sorter from "./lib/Sorter.svelte";
  import {
    SorterEngine,
    type KeyedManifest,
    type Manifest,
  } from "./lib/SorterEngine";
  import manifests from "./assets/manifests.json";

  let manifestPromise: Promise<Manifest> = $state(
    (async () => {
      // TODO: Allow user to pick a manifest
      const manifestUrl = manifests[0].url;
      return await (await fetch(manifestUrl)).json();
    })(),
  );
  // TODO: Allow user to pick a manifest
  let manifestName = $state(manifests[0].name);

  let keyedManifest: Promise<KeyedManifest> = $state(
    (async () => {
      const manifest = await manifestPromise;
      const keyedManifest: KeyedManifest = {};
      for (let i = 0; i < manifest.length; i++) {
        keyedManifest[manifest[i].g] = i;
      }
      return keyedManifest;
    })(),
  );

  let sorterEngine: Promise<SorterEngine> = $state(
    (async () => {
      let manifest = await manifestPromise;
      // TODO: Allow changing manifest, configuring number of candidate lists
      return SorterEngine.Create(manifest, 100);
    })(),
  );

  let sorterReady = Promise.all([manifestPromise, keyedManifest, sorterEngine]);
</script>

<main>
  {#await sorterReady}
    <!-- pending... -->
    <h4>Loading sorter: {manifestName}...</h4>
  {:then [manifest, keyedManifest, sorterEngine]}
    <Sorter {sorterEngine} {manifestName} {manifest} {keyedManifest} />
  {/await}
</main>

<style lang="scss">
  main {
    display: flex;
    flex-direction: column; // even in portrait mode we want a columnar layout
    justify-content: space-around;
    align-items: center;
  }
</style>
