<script lang="ts">
    import {
        useDraggable,
        type UseDraggableArguments,
    } from "@dnd-kit-svelte/core";
    import { CSS } from "@dnd-kit-svelte/utilities";

    interface DraggableCharacterProps extends UseDraggableArguments {
        characterDisplayName: string;
        characterImageUrl: string;
    }
    const {
        id,
        characterDisplayName,
        characterImageUrl,
    }: DraggableCharacterProps = $props();

    const { transform, listeners, attributes, node } = useDraggable({
        id: "draggable",
    });
    const style = $derived(
        transform.current
            ? `transform: ${CSS.Translate.toString(transform.current)}`
            : "",
    );
</script>

<div
    {style}
    bind:this={node.current}
    {...listeners.current}
    {...attributes.current}
>
    <h4>{characterDisplayName}</h4>
</div>
