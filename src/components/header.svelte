<script>
    import * as config from '$lib/config';
	import { Linkedin, Github } from 'lucide-svelte';
</script>
<nav class="nav">
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
		color: var(--brand);
		font-size: var(--font-size-3);
		text-decoration: none;
		font-weight: var(--font-weight-7);
	}
	.title:hover{
		color: var(--text-2)
	}
    nav {
		padding-block: var(--size-1);
		border-bottom: 1px solid var(--surface-3);
		margin-block: var(--size-3);
		background-color: var(--surface-3);
		border-radius: var(--radius-3);
		padding-inline: var(--size-3);
		width: 400px;
		height: 50px;
		justify-self: center;
		box-sizing: border-box;
		display: flex;
		justify-content: space-between;
		align-items: center;
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
		color: var(--brand);
	}
	@media(max-width:576px){
		nav{
			width: 300px;
		}
	}
</style>