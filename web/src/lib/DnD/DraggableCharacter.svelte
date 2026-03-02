<script lang="ts">
    import type { UseDraggableArguments } from "@dnd-kit-svelte/core";
    import { useDraggable } from "@dnd-kit-svelte/svelte";

    interface DraggableCharacterProps extends UseDraggableArguments {
        characterDisplayName: string;
        characterImageUrl: string;
    }
    const {
        id,
        characterDisplayName,
        characterImageUrl,
    }: DraggableCharacterProps = $props();

    const { ref } = $derived(useDraggable({ id }));
</script>

<div {@attach ref} class="draggableCharacter">
    <img src="{characterImageUrl}" alt="{characterDisplayName}"/>
    <h4>{characterDisplayName}</h4>
</div>

<style lang="scss">
    .draggableCharacter {
        padding: 10px;

        display: flex;
        flex-direction: column;
    }
</style>
