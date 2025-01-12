<script>
    import * as config from '$lib/config'
</script>

<footer>
    <p class="title">{config.title}&copy{new Date().getFullYear()}</p>
</footer>

<style>
	footer {
		background-color: var(--surface-3);
		padding-block: var(--size-1);
		border-top: 1px solid var(--surface-3);
		padding-inline: var(--size-3);
		border-top-left-radius: var(--radius-3);
		border-top-right-radius: var(--radius-3);
		width: 80%;
		justify-self: center;
		margin-block-start: var(--spacing-64);
		display: flex;
		align-items: center;
	}

	.title {
		color: var(--brand);
		font-size: var(--font-size-3);
	}
</style>