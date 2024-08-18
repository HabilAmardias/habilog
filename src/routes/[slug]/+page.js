import { error } from "@sveltejs/kit";
export async function load({ params }) {
    try{
        const res = await import(`../../posts/${params.slug}.md`)
        return {
            content: res.default,
            metadata: res.metadata
        }
    } catch (e){
        console.log(e)
        error(404, `Could not find ${params.slug}`)
    }
}