import { injectSpeedInsights } from '@vercel/speed-insights/sveltekit';
import { dev } from '$app/environment';
import { inject } from '@vercel/analytics';
 
inject({ mode: dev ? 'development' : 'production' });
injectSpeedInsights()

/**@type {import('./$types').LayoutLoad} */
export async function load({url}){
    return{
        url:url.pathname
    }
}

