<script>
	import { formatDate } from '$lib/utils.js';
    import * as config from '$lib/config.js'
    export let data
</script>

<svelte:head>
    <title>
        {data.metadata.title} | {config.title}
    </title>
</svelte:head>

<article>
    <section>
        <h1>{data.metadata.title}</h1>
        <p class="date">{formatDate(data.metadata.date)}</p>
    </section>
    <section class="tags">
        {#each data.metadata.tags as tag}
            <span class="surface-3">{tag}</span>
        {/each}
    </section>
    <section>
        <svelte:component this={data.content}/>
    </section>
</article>

<style>
    .tags {
		display: flex;
		gap: var(--size-2);
		margin-top: var(--size-3);
        margin-bottom: var(--size-7);
	}
    .tags > * {
		padding: var(--size-2) var(--size-3);
		border-radius: var(--radius-round);
	}
    article{
        width: 80%;
    }
    h1{
        margin-bottom: 0;
        color: var(--brand-light);
    }
    span{
        background-color: var(--gray-8);
        color: var(--brand-light);
        font-size: var(--size-3);
    }
    @media(max-width:820px){
        article{
            width: 90%;
        }
    }

    @media(max-width:512px){
        article{
            width: 100%;
        }
    }
</style>