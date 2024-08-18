<script>
    import Transition from "../components/transition.svelte";
    import Header from "../components/header.svelte";
    import '../app.css'
	export let data
</script>

<div class="layout">
    <Header/>
    <main>
		<Transition url={data.url}>
			<slot/>
		</Transition>
    </main>
</div>

<style>
	.layout {
		height: 100%;
		max-inline-size: 1440px;
		display: grid;
		grid-template-rows: auto 1fr auto;
		margin-inline: auto;
		padding-inline: var(--size-7);
	}

	main {
		padding-block: var(--size-3);
		display: flex;
	}

	@media (min-width: 1440px) {
		.layout {
			padding-inline: 0;
		}
	}

</style>