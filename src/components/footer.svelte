<script>
    import * as config from '$lib/config'
</script>

<footer>
    <p>{config.title}&copy{new Date().getFullYear()}</p>
</footer>

<style>
	footer {
		background-color: var(--surface-3);
		padding-block: var(--size-7);
		border-top: 1px solid var(--surface-3);
	}

	p {
		color: var(--text-2);
	}
</style>