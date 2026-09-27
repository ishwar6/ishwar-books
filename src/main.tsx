import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import 'katex/dist/katex.min.css'
import './styles/base.css'
import './styles/prose.css'
import './styles/layout.css'
import Layout from './components/Layout'
import Home from './pages/Home'
import Writings from './pages/Writings'
import Post from './pages/Post'
import Books from './pages/Books'
import Book from './pages/Book'
import Chapter from './pages/Chapter'
import Projects from './pages/Projects'
import About from './pages/About'
import NotFound from './pages/NotFound'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="writings" element={<Writings />} />
          <Route path="writings/:slug" element={<Post />} />
          <Route path="books" element={<Books />} />
          <Route path="books/:book" element={<Book />} />
          <Route path="books/:book/:chapter" element={<Chapter />} />
          <Route path="projects" element={<Projects />} />
          <Route path="about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
