import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// Inter, bundled rather than fetched from Google Fonts: this app's whole
// promise is that nothing leaves the machine, and a webfont request would be
// a network call on every page load.
import '@fontsource-variable/inter'
import './index.css'
import App from './App.tsx'
import { apply, load } from './lib/appearance.ts'

// Before the first render, not inside a component: applying it in an effect
// would paint the default and then correct itself, which is a flash of the
// wrong theme on every load. `load` falls back to the system preference when
// nothing has been chosen yet.
apply(load())

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
