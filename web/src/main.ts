import { mount } from 'svelte';
import './app.css';
import App from './App.svelte';
import { set_panic_hook } from 'web-engine';

set_panic_hook();

const app = mount(App, {
  target: document.getElementById('app')!,
});

export default app;