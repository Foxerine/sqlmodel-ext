import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { zh } from './zh.mjs'
import { en } from './en.mjs'

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

  markdown: {
    container: {
      tipLabel: '提示',
      warningLabel: '注意',
      dangerLabel: '危险',
      infoLabel: '信息',
      detailsLabel: '详细信息',
    },
  },
})
