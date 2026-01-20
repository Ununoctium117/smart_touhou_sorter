import { mount } from 'svelte'
import './app.css'
import App from './App.svelte'
import { greet, set_panic_hook } from 'web-engine';

set_panic_hook();
greet('asdf');

const app = mount(App, {
  target: document.getElementById('app')!,
})

export default app
