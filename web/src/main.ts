import { mount } from 'svelte';
import './app.css';
import App from './App.svelte';
import { set_panic_hook, create_sorter, estimate_progress, serialize_sorter_compressed, load_sorter } from 'web-engine';
import manifests from './assets/manifests.json';

set_panic_hook();

type Manifest = {
  g: string,
  d: string,
  u: string,
  t: string[],
}[];

// temp: load from manifest URL
const touhouManifestUrl = manifests[0].url;
const touhouManifest: Manifest = await (await fetch(touhouManifestUrl)).json();
const manifestIds = touhouManifest.map((manifestEntry) => manifestEntry.g);

const sorter_handle_1 = create_sorter(manifestIds, 100);
const serialized = serialize_sorter_compressed(sorter_handle_1);
localStorage.setItem('test-save', serialized);

const sorter_handle = load_sorter(serialized);

console.log('handle', sorter_handle);
console.log('progress', estimate_progress(sorter_handle));
console.log('serLen', serialized.length);

const app = mount(App, {
  target: document.getElementById('app')!,
});

export default app;