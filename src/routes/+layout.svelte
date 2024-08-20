<script>
    import Transition from "../components/transition.svelte";
    import Header from "../components/header.svelte";
	import Footer from "../components/footer.svelte";
    import '../app.css'
	import 'katex/dist/katex.min.css'
	export let data
</script>

<div class="layout">
    <Header/>
    <main>
		<Transition url={data.url}>
			<slot/>
		</Transition>
    </main>
	<Footer/>
</div>

<style>
	.layout {
		height: 100%;
		max-inline-size: 1440px;
		display: grid;
		grid-template-rows: repeat(3,1fr);
		margin-inline: auto;
		padding-inline: var(--size-7);
		box-sizing: border-box;
		justify-content: center;
	}

	main {
		padding-block: var(--size-3);
		display: flex;
		box-sizing: border-box;
	}

	@media (min-width: 1440px) {
		.layout {
			padding-inline: 0;
		}
	}

</style>