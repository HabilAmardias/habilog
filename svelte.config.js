import adapter from '@sveltejs/adapter-auto';
import {escapeSvelte, mdsvex} from 'mdsvex';
import rehypeKatexSvelte from 'rehype-katex-svelte';
import remarkMath from 'remark-math';
import { createHighlighter } from 'shiki';

const mdsvexOptions = {
	extensions: ['.md'],
	highlight:{
		highlighter: async(code,lang='text') =>{
			const highlighter = await createHighlighter({
				themes:['github-dark-default'],
				langs:['javascript','typescript','python','r']
			})
			await highlighter.loadLanguage('javascript','typescript','python','r')
			const html = escapeSvelte(highlighter.codeToHtml(code,{lang,theme:'github-dark-default'}))
			return `{@html \`${html}\` }`
		}
	},
	remarkPlugins:[remarkMath],
	rehypePlugins:[rehypeKatexSvelte]
};
/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: [mdsvex(mdsvexOptions)],
	extensions:['.svelte','.md'],
	kit: {
		// adapter-auto only supports some environments, see https://kit.svelte.dev/docs/adapter-auto for a list.
		// If your environment is not supported, or you settled on a specific environment, switch out the adapter.
		// See https://kit.svelte.dev/docs/adapters for more information about adapters.
		adapter: adapter()
	}
};

export default config;
