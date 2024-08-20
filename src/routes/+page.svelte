<script>
    import * as config from '$lib/config.js'
	import { formatDate, paginate } from '$lib/utils.js';
    export let data
	let currentPage=0
	let totalPosts
	$: posts=paginate(data.posts)
	$: totalPosts=posts.length
</script>

<svelte:head>
    <title>Home | {config.title}</title>
</svelte:head>

<div class="container">
    <ul class="posts">
        {#each posts[currentPage] as post}
			<a class="post" href={post.slug}>
				<li>
					<span class="title">{post.title}</span>
					<p class="date">{formatDate(post.date,'medium','en')}</p>
					<p class="description">{post.desc}</p>
				</li>
			</a>
        {/each}
    </ul>
	<section class="paginate">
		<button disabled={currentPage === 0} on:click={() => currentPage--}>
			Previous
		</button>
		{#each Array(totalPosts) as _,index}
			<button class:active={index === currentPage} on:click={() => currentPage = index}>
				{index+1}
			</button>
		{/each}
		<button disabled={currentPage === posts.length - 1} on:click={() => currentPage++}>
			Next
		</button>
	</section>
</div>

<style>
	.container{
		height: 100%;
		display: flex;
		flex-direction: column;
		width: 100%;
		align-items: center;
	}

	.posts {
		display: grid;
		gap: var(--size-3);
		grid-template-columns: repeat(3,2fr);
		grid-template-rows: repeat(3,2fr);
		width: 100%;
		height: 100%;
	}

	.post {
		max-inline-size: var(--size-content-3);
		background: var(--surface-1);
		border: 1px solid var(--surface-1);
		padding: var(--size-2);
		border-radius: var(--radius-3);
		box-shadow: var(--shadow-2);
		text-decoration: none;
  		transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
		width: 100%;
		height: 100%;
		box-sizing: border-box;
	}
	
	.post:hover{
		box-shadow: 0 14px 28px rgba(0, 0, 0, 0.25), 0 10px 10px rgba(0, 0, 0, 0.22);
		border-color: var(--brand-light);
	}

	.title {
		font-size: var(--font-size-fluid-1);
		text-transform: capitalize;
		color: var(--brand-light);
		
	}

	.date {
		color: var(--text-2);
		font-size: var(--size-3);
	}

	.description {
		margin-top: var(--size-3);
		font-size: var(--size-3);
		color: var(--text-1-light);
	}

	.paginate{
		margin-top: auto;
		display: flex;
		justify-content: center;
		gap: var(--size-2);
	}

	.paginate>button{
		background-color: white;
		padding-inline: var(--size-3);
		padding-block: var(--size-2);
		border-radius: var(--radius-3);
	}

	.paginate>button:hover{
		background-color:var(--brand-light);
		color: white;
		
	}

	.paginate>button.active{
		background-color: var(--brand-light);
		color: white;
	}

	.paginate>button:disabled{
		cursor: not-allowed;
		background: transparent;
		border: none;
	}

	
	@media(max-width:768px){
		.posts{
			grid-template-columns: repeat(1,1fr);
			justify-content: center;
		}
		.post{
			justify-self: center;
		}
	}
</style>