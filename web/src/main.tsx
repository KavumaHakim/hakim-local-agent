import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// Inter, bundled rather than fetched from Google Fonts: this app's whole
// promise is that nothing leaves the machine, and a webfont request would be
// a network call on every page load.
import '@fontsource-variable/inter'
import './index.css'
import App from './App.tsx'

// The system's starting value. After this the rail's toggle owns it, so the
// attribute is set once here and never read from the media query again.
document.documentElement.dataset.theme = window.matchMedia(
  '(prefers-color-scheme: light)',
).matches
  ? 'light'
  : 'dark'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
