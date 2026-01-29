import { mount } from 'svelte';
import './app.css';
import App from './App.svelte';
import { set_panic_hook, create_sorter, estimate_progress, serialize_sorter } from 'web-engine';

set_panic_hook();

const sorter_handle = create_sorter(['a', 'b', 'c'], 100);
console.log('handle', sorter_handle);
console.log('progress', estimate_progress(sorter_handle));
console.log('ser', serialize_sorter(sorter_handle));

const app = mount(App, {
  target: document.getElementById('app')!,
});

export default app;