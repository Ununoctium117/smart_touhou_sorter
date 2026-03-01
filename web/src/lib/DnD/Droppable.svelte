<script lang="ts">
    import { useDroppable, type UseDroppableInput } from "@dnd-kit-svelte/svelte";
    import type { Snippet } from "svelte";
    import type { ClassValue } from "svelte/elements";

    interface DroppableProps extends UseDroppableInput {
        children: Snippet;
        class?: ClassValue;
    }

    let { children, class: className, ...rest }: DroppableProps = $props();
    const droppable = $derived(useDroppable(rest));
</script>

<div class={className} bind:this={droppable.droppable.element}>
    {@render children()}
</div>