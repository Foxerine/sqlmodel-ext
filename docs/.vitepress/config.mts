import { withMermaid } from 'vitepress-plugin-mermaid'
import llmstxt from 'vitepress-plugin-llms'
import { copyOrDownloadAsMarkdownButtons } from 'vitepress-plugin-llms'
import { zh } from './zh.mts'
import { en } from './en.mts'

export default withMermaid({
  title: 'sqlmodel-ext',

  vite: {
    plugins: [llmstxt()],
    optimizeDeps: {
      include: ['mermaid', 'dayjs'],
    },
    ssr: {
      noExternal: ['vitepress-plugin-mermaid', 'mermaid'],
    },
  },

  markdown: {
    config(md) {
      md.use(copyOrDownloadAsMarkdownButtons)
    },
  },

  locales: {
    root: {
      label: '简体中文',
      ...zh,
    },
    en: {
      label: 'English',
      ...en,
    },
  },

  themeConfig: {
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Foxerine/sqlmodel-ext' },
    ],
    search: {
      provider: 'local',
    },
  },
})
