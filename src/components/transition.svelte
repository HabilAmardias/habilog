<script>
    import { fade } from "svelte/transition";
    export let url
</script>

{#key url}
    <div class="transition" in:fade>
        <slot/>
    </div>
{/key}

<style>
	.transition {
		height: 100%;
        width: 100%;
        display: flex;
        justify-content: center;
	}
</style>