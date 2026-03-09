import { withMermaid } from 'vitepress-plugin-mermaid'
import { zh } from './zh.mts'
import { en } from './en.mts'

export default withMermaid({
  title: 'sqlmodel-ext',

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
