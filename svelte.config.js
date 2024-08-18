import adapter from '@sveltejs/adapter-auto';
import {escapeSvelte, mdsvex} from 'mdsvex';
import { createHighlighter } from 'shiki';

const mdsvexOptions = {
	extensions: ['.md'],
	highlight:{
		highlighter: async(code,lang='text') =>{
			const highlighter = await createHighlighter({
				themes:['material-theme-lighter'],
				langs:['javascript','typescript','python','r']
			})
			await highlighter.loadLanguage('javascript','typescript','python','r')
			const html=escapeSvelte(highlighter.codeToHtml(code,{lang,theme:'material-theme-lighter'}))
			return `{@html \`${html}\` }`
		}
	},
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
