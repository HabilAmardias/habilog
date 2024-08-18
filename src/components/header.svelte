<script>
    import * as config from '$lib/config';
	import { Linkedin, Github } from 'lucide-svelte';
	import { onMount } from 'svelte';
	let isSticky=false
	onMount(()=>{
		const scrollHandler=()=>{
			isSticky=window.scrollY>0
		}
		scrollHandler()
		window.addEventListener('scroll',scrollHandler)
		return ()=>{
			window.removeEventListener('scroll',scrollHandler)
		}
	})
</script>
<nav class:sticky={isSticky}>
    <a href="/" class="title">{config.title}</a>
    <ul class="links">
        <li>
            <a href="https://www.linkedin.com/in/muhammad-habil-amardias/">
				<Linkedin color='white'/>
			</a>
        </li>
		<li>
			<a href="https://github.com/HabilAmardias">
				<Github color='white'/>
			</a>
		</li>
    </ul>
</nav>

<style>
	.title{
		color: var(--brand-light);
		font-size: var(--font-size-3);
		text-decoration: none;
		font-weight: var(--font-weight-7);
	}
	.title:hover{
		color: var(--gray-3)
	}
    nav {
		padding-block: var(--size-1);
		border-bottom: 1px solid var(--surface-4);
		margin-block: var(--size-3);
		background-color: var(--gray-8);
		border-radius: var(--radius-3);
		padding-inline: var(--size-3);
		width: 500px;
		height: 80px;
		justify-self: center;
		box-sizing: border-box;
		display: flex;
		justify-content: space-between;
		align-items: center;
	}
	.sticky {
		position: fixed;
		top: 0;
		z-index: 100;
		width: 500px;
	}

	.links {
		margin-block: var(--size-7);
		display: flex;
		gap: var(--size-3);
	}

	li>a {
		color: var(--gray-5);
		text-decoration: none;
		font-weight: bold;
	}

	a:hover{
		color: var(--brand-light);
	}

	@media (min-width: 768px) {
		nav {
			display: flex;
			align-items: center;
			justify-content: space-between;
			width: 700px;
			padding-block: var(--size-3);
		}

		.links {
			display: flex;
			gap: var(--size-3);
			margin-block: 0;
		}

		.sticky{
			width: 700px;
		}
	}

	@media(max-width:576px){
		nav, .sticky{
			width: 300px;
			height: 50px;
		}
	}
</style>